import { FEATURES_PER_TOKEN, MAX_TOKENS, featurizeQuery } from "./featurize";
import {
  DequantizedWeights,
  HIDDEN,
  NUM_ROLES,
  SieveModel,
  compileTokens,
  forwardTokensCPU,
  validateCompiledQuery,
} from "./model";
import { FilterClause, SchemaDef } from "./schema";

export const SIEVE_BATCH_WGSL = /* wgsl */ `
struct BatchConfig {
  batch_size: u32,
  max_tokens: u32,
  hidden: u32,
  num_roles: u32,
};

@group(0) @binding(0) var<uniform> cfg: BatchConfig;
@group(0) @binding(1) var<storage, read> feature_ids: array<i32>; // [B, T, 8]
@group(0) @binding(2) var<storage, read> lengths: array<u32>;     // [B]
@group(0) @binding(3) var<storage, read> weights_buf: array<f32>; // flat packed weights
@group(0) @binding(4) var<storage, read_write> out_logits: array<f32>; // [B, T, 13]

// Weight buffer offsets (total 29,613 floats):
const OFF_EMBED: u32 = 0u;          // 52 * 32 = 1664
const OFF_DW_W: u32 = 1664u;        // 32 * 5  = 160
const OFF_DW_B: u32 = 1824u;        // 32
const OFF_LFC1_W: u32 = 1856u;      // 64 * 192 = 12288
const OFF_LFC1_B: u32 = 14144u;     // 64
const OFF_LFC2_W: u32 = 14208u;     // 32 * 64 = 2048
const OFF_LFC2_B: u32 = 16256u;     // 32
const OFF_FG_W: u32 = 16288u;       // 32 * 32 = 1024
const OFF_FG_B: u32 = 17312u;       // 32
const OFF_FC_W: u32 = 17344u;       // 32 * 32 = 1024
const OFF_FC_B: u32 = 18368u;       // 32
const OFF_BG_W: u32 = 18400u;       // 32 * 32 = 1024
const OFF_BG_B: u32 = 19424u;       // 32
const OFF_BC_W: u32 = 19456u;       // 32 * 32 = 1024
const OFF_BC_B: u32 = 20480u;       // 32
const OFF_HFC1_W: u32 = 20512u;     // 64 * 128 = 8192
const OFF_HFC1_B: u32 = 28704u;     // 64
const OFF_HFC2_W: u32 = 28768u;     // 13 * 64 = 832
const OFF_HFC2_B: u32 = 29600u;     // 13

fn sigmoid_f(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= cfg.batch_size) {
    return;
  }

  let T = min(lengths[b], 24u);
  if (T == 0u) {
    return;
  }

  var x_emb: array<f32, 768>;   // 24 * 32
  var x_local: array<f32, 768>; // 24 * 32
  var fwd_st: array<f32, 768>;  // 24 * 32
  var bwd_st: array<f32, 768>;  // 24 * 32

  let base_feat_b = b * 24u * 8u;

  // 1. Summed sparse-feature embedding
  for (var t: u32 = 0u; t < T; t = t + 1u) {
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
      var sum_e: f32 = 0.0;
      for (var f: u32 = 0u; f < 8u; f = f + 1u) {
        let fid = feature_ids[base_feat_b + t * 8u + f];
        if (fid > 0 && fid < 52) {
          sum_e = sum_e + weights_buf[OFF_EMBED + u32(fid) * 32u + c];
        }
      }
      x_emb[t * 32u + c] = sum_e;
    }
  }

  // 2. Depthwise conv1d + neighbor gathers + local MLP
  for (var t: u32 = 0u; t < T; t = t + 1u) {
    var local_cat: array<f32, 192>;
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
      local_cat[c] = x_emb[t * 32u + c];
      var conv_s: f32 = weights_buf[OFF_DW_B + c];
      for (var k: u32 = 0u; k < 5u; k = k + 1u) {
        let st = i32(t) + i32(k) - 2;
        if (st >= 0 && st < i32(T)) {
          conv_s = conv_s + weights_buf[OFF_DW_W + c * 5u + k] * x_emb[u32(st) * 32u + c];
        }
      }
      local_cat[32u + c] = conv_s;
      local_cat[64u + c] = select(0.0, x_emb[(t - 1u) * 32u + c], t >= 1u);
      local_cat[96u + c] = select(0.0, x_emb[(t + 1u) * 32u + c], t + 1u < T);
      local_cat[128u + c] = select(0.0, x_emb[(t - 2u) * 32u + c], t >= 2u);
      local_cat[160u + c] = select(0.0, x_emb[(t + 2u) * 32u + c], t + 2u < T);
    }

    var mid: array<f32, 64>;
    for (var o: u32 = 0u; o < 64u; o = o + 1u) {
      var s: f32 = weights_buf[OFF_LFC1_B + o];
      let w_off = OFF_LFC1_W + o * 192u;
      for (var i: u32 = 0u; i < 192u; i = i + 1u) {
        s = s + weights_buf[w_off + i] * local_cat[i];
      }
      mid[o] = max(s, 0.0);
    }

    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
      var s2: f32 = weights_buf[OFF_LFC2_B + c];
      let w_off2 = OFF_LFC2_W + c * 64u;
      for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        s2 = s2 + weights_buf[w_off2 + i] * mid[i];
      }
      x_local[t * 32u + c] = x_emb[t * 32u + c] + s2;
    }
  }

  // 3. Forward gated affine scan
  for (var t: u32 = 0u; t < T; t = t + 1u) {
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
      var g_sum: f32 = weights_buf[OFF_FG_B + c];
      var c_sum: f32 = weights_buf[OFF_FC_B + c];
      let wg = OFF_FG_W + c * 32u;
      let wc = OFF_FC_W + c * 32u;
      for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let xv = x_local[t * 32u + i];
        g_sum = g_sum + weights_buf[wg + i] * xv;
        c_sum = c_sum + weights_buf[wc + i] * xv;
      }
      let prev = select(0.0, fwd_st[(t - 1u) * 32u + c], t > 0u);
      fwd_st[t * 32u + c] = sigmoid_f(g_sum) * prev + tanh(c_sum);
    }
  }

  // 4. Backward gated affine scan
  for (var step: u32 = 0u; step < T; step = step + 1u) {
    let t = T - 1u - step;
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
      var g_sum: f32 = weights_buf[OFF_BG_B + c];
      var c_sum: f32 = weights_buf[OFF_BC_B + c];
      let wg = OFF_BG_W + c * 32u;
      let wc = OFF_BC_W + c * 32u;
      for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let xv = x_local[t * 32u + i];
        g_sum = g_sum + weights_buf[wg + i] * xv;
        c_sum = c_sum + weights_buf[wc + i] * xv;
      }
      let next_v = select(0.0, bwd_st[(t + 1u) * 32u + c], t + 1u < T);
      bwd_st[t * 32u + c] = sigmoid_f(g_sum) * next_v + tanh(c_sum);
    }
  }

  // 5. Global mean context
  var global_ctx: array<f32, 32>;
  let inv_t = 1.0 / f32(T);
  for (var c: u32 = 0u; c < 32u; c = c + 1u) {
    var gs: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {
      gs = gs + x_local[t * 32u + c];
    }
    global_ctx[c] = gs * inv_t;
  }

  // 6. Two-layer classification head -> 13 logits per token
  let out_base_b = b * 24u * 13u;
  for (var t: u32 = 0u; t < T; t = t + 1u) {
    var h_in: array<f32, 128>;
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
      h_in[c] = x_local[t * 32u + c];
      h_in[32u + c] = fwd_st[t * 32u + c];
      h_in[64u + c] = bwd_st[t * 32u + c];
      h_in[96u + c] = global_ctx[c];
    }

    var h_mid: array<f32, 64>;
    for (var o: u32 = 0u; o < 64u; o = o + 1u) {
      var s: f32 = weights_buf[OFF_HFC1_B + o];
      let w_off = OFF_HFC1_W + o * 128u;
      for (var i: u32 = 0u; i < 128u; i = i + 1u) {
        s = s + weights_buf[w_off + i] * h_in[i];
      }
      h_mid[o] = max(s, 0.0);
    }

    for (var r: u32 = 0u; r < 13u; r = r + 1u) {
      var s2: f32 = weights_buf[OFF_HFC2_B + r];
      let w_off2 = OFF_HFC2_W + r * 64u;
      for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        s2 = s2 + weights_buf[w_off2 + i] * h_mid[i];
      }
      out_logits[out_base_b + t * 13u + r] = s2;
    }
  }
}
`;

export function packWeightsFlat(w: DequantizedWeights): Float32Array {
  const arrays = [
    w.embedWeight,
    w.dwConvWeight,
    w.dwConvBias,
    w.localFc1Weight,
    w.localFc1Bias,
    w.localFc2Weight,
    w.localFc2Bias,
    w.fwdGateWeight,
    w.fwdGateBias,
    w.fwdCandWeight,
    w.fwdCandBias,
    w.bwdGateWeight,
    w.bwdGateBias,
    w.bwdCandWeight,
    w.bwdCandBias,
    w.headFc1Weight,
    w.headFc1Bias,
    w.headFc2Weight,
    w.headFc2Bias,
  ];
  const totalLen = arrays.reduce((acc, a) => acc + a.length, 0);
  const flat = new Float32Array(totalLen);
  let offset = 0;
  for (const a of arrays) {
    flat.set(a, offset);
    offset += a.length;
  }
  return flat;
}

export interface BatchEvaluationResult {
  batchSize: number;
  backend: "webgpu" | "cpu-fallback";
  totalMs: number;
  cpuComparisonMs: number;
  parityVerified: boolean;
  maxLogitDiff: number;
  clausesPerQuery: FilterClause[][];
  issuesPerQuery: string[][];
}

export class SieveWebGPUBatchEngine {
  private readonly model: SieveModel;
  private readonly flatWeights: Float32Array;
  private device: any = null;
  private pipeline: any = null;
  private weightsBuffer: any = null;

  constructor(model: SieveModel = new SieveModel()) {
    this.model = model;
    this.flatWeights = packWeightsFlat(model.weights);
  }

  async initWebGPU(): Promise<boolean> {
    if (this.device && this.pipeline) return true;
    const nav = typeof navigator !== "undefined" ? (navigator as any) : undefined;
    if (!nav || !nav.gpu) return false;

    try {
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) return false;
      this.device = await adapter.requestDevice();
      const module = this.device.createShaderModule({ code: SIEVE_BATCH_WGSL });
      this.pipeline = await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });

      const GPUBufferUsage = (globalThis as any).GPUBufferUsage;
      this.weightsBuffer = this.device.createBuffer({
        size: this.flatWeights.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.weightsBuffer, 0, this.flatWeights);
      return true;
    } catch {
      this.device = null;
      this.pipeline = null;
      return false;
    }
  }

  async evaluateBatch(
    queries: readonly string[],
    schema: SchemaDef,
    checkParitySample: number = 8,
  ): Promise<BatchEvaluationResult> {
    const B = queries.length;
    const metasList = queries.map((q) => featurizeQuery(q, schema, false));
    const featureIds = new Int32Array(B * MAX_TOKENS * FEATURES_PER_TOKEN);
    const lengths = new Uint32Array(B);

    for (let b = 0; b < B; b++) {
      const metas = metasList[b];
      const T = Math.min(metas.length, MAX_TOKENS);
      lengths[b] = T;
      const baseB = b * MAX_TOKENS * FEATURES_PER_TOKEN;
      for (let t = 0; t < T; t++) {
        for (let f = 0; f < FEATURES_PER_TOKEN; f++) {
          featureIds[baseB + t * FEATURES_PER_TOKEN + f] = metas[t].featureIds[f];
        }
      }
    }

    // Measure CPU batch evaluation
    const cpuStart = typeof performance !== "undefined" ? performance.now() : Date.now();
    const cpuClauses: FilterClause[][] = new Array(B);
    const cpuIssues: string[][] = new Array(B);
    const cpuSampleLogits: Float32Array[] = [];

    for (let b = 0; b < B; b++) {
      const T = lengths[b];
      if (T === 0) {
        cpuClauses[b] = [];
        cpuIssues[b] = queries[b].trim() ? ["No filter could be applied"] : [];
        continue;
      }
      const slice = featureIds.subarray(
        b * MAX_TOKENS * FEATURES_PER_TOKEN,
        b * MAX_TOKENS * FEATURES_PER_TOKEN + T * FEATURES_PER_TOKEN,
      );
      const fwd = forwardTokensCPU(slice, T, this.model.weights);
      if (b < checkParitySample) {
        cpuSampleLogits.push(fwd.roleLogits);
      }
      const bFlags = new Uint8Array(T);
      for (let t = 0; t < T; t++) bFlags[t] = fwd.boundaryProbs[t] >= 0.5 ? 1 : 0;
      cpuClauses[b] = compileTokens(metasList[b], fwd.roleIds, bFlags, schema);
      cpuIssues[b] = validateCompiledQuery(queries[b], schema, metasList[b], cpuClauses[b]);
      if (cpuIssues[b].length) cpuClauses[b] = [];
    }
    const cpuComparisonMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - cpuStart;

    const hasGpu = await this.initWebGPU();
    if (!hasGpu || !this.device || !this.pipeline) {
      return {
        batchSize: B,
        backend: "cpu-fallback",
        totalMs: Number(cpuComparisonMs.toFixed(3)),
        cpuComparisonMs: Number(cpuComparisonMs.toFixed(3)),
        parityVerified: true,
        maxLogitDiff: 0,
        clausesPerQuery: cpuClauses,
        issuesPerQuery: cpuIssues,
      };
    }

    // Run WebGPU Compute Kernel
    const GPUBufferUsage = (globalThis as any).GPUBufferUsage;
    const GPUMapMode = (globalThis as any).GPUMapMode;
    const gpuStart = performance.now();

    const cfgData = new Uint32Array([B, MAX_TOKENS, HIDDEN, NUM_ROLES]);
    const cfgBuf = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const featBuf = this.device.createBuffer({
      size: featureIds.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const lenBuf = this.device.createBuffer({
      size: lengths.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outBytes = B * MAX_TOKENS * (NUM_ROLES + 1) * 4;
    const outBuf = this.device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuf = this.device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(cfgBuf, 0, cfgData);
    this.device.queue.writeBuffer(featBuf, 0, featureIds);
    this.device.queue.writeBuffer(lenBuf, 0, lengths);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cfgBuf } },
        { binding: 1, resource: { buffer: featBuf } },
        { binding: 2, resource: { buffer: lenBuf } },
        { binding: 3, resource: { buffer: this.weightsBuffer } },
        { binding: 4, resource: { buffer: outBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(B / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
    this.device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const gpuLogits = new Float32Array(readBuf.getMappedRange()).slice();
    readBuf.unmap();
    const gpuTotalMs = performance.now() - gpuStart;

    cfgBuf.destroy();
    featBuf.destroy();
    lenBuf.destroy();
    outBuf.destroy();
    readBuf.destroy();

    // Automatic CPU parity check on sample queries
    let maxDiff = 0;
    const gpuClauses: FilterClause[][] = new Array(B);
    for (let b = 0; b < B; b++) {
      const T = lengths[b];
      const roleIds = new Int32Array(T);
      const bFlags = new Uint8Array(T);
      const baseOut = b * MAX_TOKENS * 13;

      for (let t = 0; t < T; t++) {
        let bestR = 0;
        let maxL = -Infinity;
        for (let r = 0; r < NUM_ROLES; r++) {
          const l = gpuLogits[baseOut + t * 13 + r];
          if (b < cpuSampleLogits.length) {
            const diff = Math.abs(l - cpuSampleLogits[b][t * NUM_ROLES + r]);
            if (diff > maxDiff) maxDiff = diff;
          }
          if (l > maxL) {
            maxL = l;
            bestR = r;
          }
        }
        roleIds[t] = bestR;
        const bLogit = gpuLogits[baseOut + t * 13 + NUM_ROLES];
        bFlags[t] = 1 / (1 + Math.exp(-bLogit)) >= 0.5 ? 1 : 0;
      }
      gpuClauses[b] = compileTokens(metasList[b], roleIds, bFlags, schema);
    }

    const gpuIssues = gpuClauses.map((clauses, b) => validateCompiledQuery(queries[b], schema, metasList[b], clauses));
    for (let b = 0; b < B; b++) if (gpuIssues[b].length) gpuClauses[b] = [];
    return {
      batchSize: B,
      backend: "webgpu",
      totalMs: Number(gpuTotalMs.toFixed(3)),
      cpuComparisonMs: Number(cpuComparisonMs.toFixed(3)),
      parityVerified: maxDiff < 1e-3,
      maxLogitDiff: Number(maxDiff.toFixed(6)),
      clausesPerQuery: gpuClauses,
      issuesPerQuery: gpuIssues,
    };
  }
}
