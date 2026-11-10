import {
  FEATURES_PER_TOKEN,
  LEX_AFTER_SET,
  LEX_AND_SET,
  LEX_BEFORE_SET,
  LEX_BOOL_FALSE_SET,
  LEX_BOOL_TRUE_SET,
  LEX_CONTAINS_SET,
  LEX_EQ_SET,
  LEX_GT_SET,
  LEX_LT_SET,
  LEX_NEGATE_SET,
  LEX_NEQ_SET,
  LEX_FILLER_SET,
  MAX_TOKENS,
  TokenFeatureMeta,
  featurizeQuery,
} from "./featurize";
import { FieldDef, FilterClause, SchemaDef } from "./schema";
import { isIsoDate } from "./autoSchema";
import rawWeightsJson from "./weights.json";

export const ROLES = [
  "NONE",
  "FIELD",
  "VALUE",
  "OP_EQ",
  "OP_NEQ",
  "OP_GT",
  "OP_LT",
  "OP_CONTAINS",
  "OP_BEFORE",
  "OP_AFTER",
  "AND",
  "NEGATE",
] as const;

export type RoleName = (typeof ROLES)[number];
export const NUM_ROLES = ROLES.length; // 12
export const HIDDEN = 32;

const ROLE_TO_CMP: Partial<Record<RoleName, FilterClause["cmp"]>> = {
  OP_EQ: "eq",
  OP_NEQ: "neq",
  OP_GT: "gt",
  OP_LT: "lt",
  OP_CONTAINS: "contains",
  OP_BEFORE: "before",
  OP_AFTER: "after",
};

export interface QuantizedTensorSpec {
  shape: number[];
  scale: number;
  b64: string;
}

export interface WeightsPayload {
  architecture: string;
  hidden: number;
  num_features: number;
  features_per_token: number;
  quantization_bits: number;
  roles: string[];
  param_count: number;
  packed_int6_bytes: number;
  tensors: Record<string, QuantizedTensorSpec>;
  parity_traces?: Array<{
    schema: string;
    query: string;
    tokens: string[];
    feature_ids: number[][];
    role_ids: number[];
    roles: string[];
    role_probs_top1: number[];
    boundary_probs: number[];
    fwd_gate_mean: number[];
    bwd_gate_mean: number[];
    clauses: FilterClause[];
    baseline_clauses: FilterClause[];
  }>;
}

export interface DequantizedWeights {
  hidden: number;
  numFeatures: number;
  paramCount: number;
  packedInt6Bytes: number;
  embedWeight: Float32Array;       // [52, 32]
  dwConvWeight: Float32Array;      // [32, 5]
  dwConvBias: Float32Array;        // [32]
  localFc1Weight: Float32Array;    // [64, 192]
  localFc1Bias: Float32Array;      // [64]
  localFc2Weight: Float32Array;    // [32, 64]
  localFc2Bias: Float32Array;      // [32]
  fwdGateWeight: Float32Array;     // [32, 32]
  fwdGateBias: Float32Array;       // [32]
  fwdCandWeight: Float32Array;     // [32, 32]
  fwdCandBias: Float32Array;       // [32]
  bwdGateWeight: Float32Array;     // [32, 32]
  bwdGateBias: Float32Array;       // [32]
  bwdCandWeight: Float32Array;     // [32, 32]
  bwdCandBias: Float32Array;       // [32]
  headFc1Weight: Float32Array;     // [64, 128]
  headFc1Bias: Float32Array;       // [64]
  headFc2Weight: Float32Array;     // [13, 64]
  headFc2Bias: Float32Array;       // [13]
}

function decodeBase64Bytes(b64: string): Uint8Array {
  if (typeof Buffer !== "undefined") {
    return new Uint8Array(Buffer.from(b64, "base64"));
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function unpackInt6Tensor(spec: QuantizedTensorSpec): Float32Array {
  const count = spec.shape.reduce((a, b) => a * b, 1);
  const packed = decodeBase64Bytes(spec.b64);
  const out = new Float32Array(count);
  const scale = Math.fround(spec.scale);
  let acc = 0;
  let bitsInAcc = 0;
  let idx = 0;

  for (let i = 0; i < packed.length && idx < count; i++) {
    acc = (acc << 8) | packed[i];
    bitsInAcc += 8;
    while (bitsInAcc >= 6 && idx < count) {
      bitsInAcc -= 6;
      const u = (acc >> bitsInAcc) & 0x3f;
      const q = u - 32;
      out[idx++] = Math.fround(q * scale);
    }
  }
  return out;
}

export function loadDequantizedWeights(payload: WeightsPayload = rawWeightsJson as unknown as WeightsPayload): DequantizedWeights {
  const t = payload.tensors;
  const embedWeight = unpackInt6Tensor(t.embed_weight);
  for (let c = 0; c < payload.hidden; c++) embedWeight[c] = 0;

  return {
    hidden: payload.hidden,
    numFeatures: payload.num_features,
    paramCount: payload.param_count,
    packedInt6Bytes: payload.packed_int6_bytes,
    embedWeight,
    dwConvWeight: unpackInt6Tensor(t.dw_conv_weight),
    dwConvBias: unpackInt6Tensor(t.dw_conv_bias),
    localFc1Weight: unpackInt6Tensor(t.local_fc1_weight),
    localFc1Bias: unpackInt6Tensor(t.local_fc1_bias),
    localFc2Weight: unpackInt6Tensor(t.local_fc2_weight),
    localFc2Bias: unpackInt6Tensor(t.local_fc2_bias),
    fwdGateWeight: unpackInt6Tensor(t.fwd_gate_weight),
    fwdGateBias: unpackInt6Tensor(t.fwd_gate_bias),
    fwdCandWeight: unpackInt6Tensor(t.fwd_cand_weight),
    fwdCandBias: unpackInt6Tensor(t.fwd_cand_bias),
    bwdGateWeight: unpackInt6Tensor(t.bwd_gate_weight),
    bwdGateBias: unpackInt6Tensor(t.bwd_gate_bias),
    bwdCandWeight: unpackInt6Tensor(t.bwd_cand_weight),
    bwdCandBias: unpackInt6Tensor(t.bwd_cand_bias),
    headFc1Weight: unpackInt6Tensor(t.head_fc1_weight),
    headFc1Bias: unpackInt6Tensor(t.head_fc1_bias),
    headFc2Weight: unpackInt6Tensor(t.head_fc2_weight),
    headFc2Bias: unpackInt6Tensor(t.head_fc2_bias),
  };
}

function linearLayer(
  x: Float32Array,
  xOffset: number,
  inDim: number,
  weight: Float32Array,
  bias: Float32Array,
  out: Float32Array,
  outOffset: number,
  outDim: number,
  relu: boolean = false,
): void {
  for (let o = 0; o < outDim; o++) {
    let sum = bias[o];
    const wRow = o * inDim;
    for (let i = 0; i < inDim; i++) {
      sum += weight[wRow + i] * x[xOffset + i];
    }
    out[outOffset + o] = relu && sum < 0 ? 0 : sum;
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export interface TokenTrace {
  text: string;
  norm: string;
  start: number;
  end: number;
  featureIds: number[];
  roleId: number;
  role: RoleName;
  confidence: number;
  roleProbs: number[];
  boundaryProb: number;
  isBoundary: boolean;
  fwdGateMean: number;
  bwdGateMean: number;
  matchedField: string | null;
  matchedEnumField: string | null;
  matchedEnumValue: string | null;
  fieldMatchQuality: string;
}

export interface NeuralParseResult {
  query: string;
  schemaName: string;
  tokens: TokenTrace[];
  clauses: FilterClause[];
  accepted: boolean;
  issues: string[];
  latencyMs: number;
  fwdStates: Float32Array; // [T, 32]
  bwdStates: Float32Array; // [T, 32]
}

export function forwardTokensCPU(
  featureIdsFlat: Int32Array | number[],
  numTokens: number,
  weights: DequantizedWeights,
): {
  roleLogits: Float32Array;
  roleProbs: Float32Array;
  roleIds: Int32Array;
  boundaryLogits: Float32Array;
  boundaryProbs: Float32Array;
  fwdGateMean: Float32Array;
  bwdGateMean: Float32Array;
  fwdStates: Float32Array;
  bwdStates: Float32Array;
  headHidden: Float32Array;
} {
  const T = numTokens;
  const H = weights.hidden; // 32

  const xEmb = new Float32Array(T * H);
  for (let t = 0; t < T; t++) {
    const baseF = t * FEATURES_PER_TOKEN;
    const baseH = t * H;
    for (let f = 0; f < FEATURES_PER_TOKEN; f++) {
      const fid = featureIdsFlat[baseF + f];
      if (fid > 0 && fid < weights.numFeatures) {
        const embRow = fid * H;
        for (let c = 0; c < H; c++) {
          xEmb[baseH + c] += weights.embedWeight[embRow + c];
        }
      }
    }
  }

  // Depthwise Conv1d (kernel=5, pad=2)
  const xConv = new Float32Array(T * H);
  for (let t = 0; t < T; t++) {
    const baseH = t * H;
    for (let c = 0; c < H; c++) {
      let sum = weights.dwConvBias[c];
      const wBase = c * 5;
      for (let k = 0; k < 5; k++) {
        const srcT = t + k - 2;
        if (srcT >= 0 && srcT < T) {
          sum += weights.dwConvWeight[wBase + k] * xEmb[srcT * H + c];
        }
      }
      xConv[baseH + c] = sum;
    }
  }

  // Local mixing over [center, conv, left1, right1, left2, right2] -> 192 -> 64 -> 32
  const xLocal = new Float32Array(T * H);
  const localCat = new Float32Array(H * 6);
  const localMid = new Float32Array(H * 2);
  const localOut = new Float32Array(H);

  for (let t = 0; t < T; t++) {
    localCat.fill(0);
    localCat.set(xEmb.subarray(t * H, (t + 1) * H), 0);
    localCat.set(xConv.subarray(t * H, (t + 1) * H), H);
    if (t - 1 >= 0) localCat.set(xEmb.subarray((t - 1) * H, t * H), H * 2);
    if (t + 1 < T) localCat.set(xEmb.subarray((t + 1) * H, (t + 2) * H), H * 3);
    if (t - 2 >= 0) localCat.set(xEmb.subarray((t - 2) * H, (t - 1) * H), H * 4);
    if (t + 2 < T) localCat.set(xEmb.subarray((t + 2) * H, (t + 3) * H), H * 5);

    linearLayer(localCat, 0, H * 6, weights.localFc1Weight, weights.localFc1Bias, localMid, 0, H * 2, true);
    linearLayer(localMid, 0, H * 2, weights.localFc2Weight, weights.localFc2Bias, localOut, 0, H, false);

    const baseH = t * H;
    for (let c = 0; c < H; c++) {
      xLocal[baseH + c] = xEmb[baseH + c] + localOut[c];
    }
  }

  // Forward & Backward gated affine scans: state[t] = gate[t] * state[t-1] + candidate[t]
  const fg = new Float32Array(T * H);
  const fc = new Float32Array(T * H);
  const bg = new Float32Array(T * H);
  const bc = new Float32Array(T * H);

  for (let t = 0; t < T; t++) {
    linearLayer(xLocal, t * H, H, weights.fwdGateWeight, weights.fwdGateBias, fg, t * H, H, false);
    linearLayer(xLocal, t * H, H, weights.fwdCandWeight, weights.fwdCandBias, fc, t * H, H, false);
    linearLayer(xLocal, t * H, H, weights.bwdGateWeight, weights.bwdGateBias, bg, t * H, H, false);
    linearLayer(xLocal, t * H, H, weights.bwdCandWeight, weights.bwdCandBias, bc, t * H, H, false);
    const base = t * H;
    for (let c = 0; c < H; c++) {
      fg[base + c] = sigmoid(fg[base + c]);
      fc[base + c] = Math.tanh(fc[base + c]);
      bg[base + c] = sigmoid(bg[base + c]);
      bc[base + c] = Math.tanh(bc[base + c]);
    }
  }

  const fwdStates = new Float32Array(T * H);
  const bwdStates = new Float32Array(T * H);
  const fwdGateMean = new Float32Array(T);
  const bwdGateMean = new Float32Array(T);

  for (let t = 0; t < T; t++) {
    const base = t * H;
    const prevBase = (t - 1) * H;
    let gSum = 0;
    for (let c = 0; c < H; c++) {
      const prev = t > 0 ? fwdStates[prevBase + c] : 0;
      const g = fg[base + c];
      gSum += g;
      fwdStates[base + c] = g * prev + fc[base + c];
    }
    fwdGateMean[t] = gSum / H;
  }

  for (let t = T - 1; t >= 0; t--) {
    const base = t * H;
    const nextBase = (t + 1) * H;
    let gSum = 0;
    for (let c = 0; c < H; c++) {
      const next = t + 1 < T ? bwdStates[nextBase + c] : 0;
      const g = bg[base + c];
      gSum += g;
      bwdStates[base + c] = g * next + bc[base + c];
    }
    bwdGateMean[t] = gSum / H;
  }

  // Mean-pooled global context over valid tokens T
  const globalCtx = new Float32Array(H);
  if (T > 0) {
    for (let t = 0; t < T; t++) {
      const base = t * H;
      for (let c = 0; c < H; c++) globalCtx[c] += xLocal[base + c];
    }
    for (let c = 0; c < H; c++) globalCtx[c] /= T;
  }

  // Two-layer classification head
  const roleLogits = new Float32Array(T * NUM_ROLES);
  const roleProbs = new Float32Array(T * NUM_ROLES);
  const roleIds = new Int32Array(T);
  const boundaryLogits = new Float32Array(T);
  const boundaryProbs = new Float32Array(T);
  const headHidden = new Float32Array(T * H * 2);

  const headIn = new Float32Array(H * 4);
  const headMid = new Float32Array(H * 2);
  const headOut = new Float32Array(NUM_ROLES + 1);

  for (let t = 0; t < T; t++) {
    headIn.set(xLocal.subarray(t * H, (t + 1) * H), 0);
    headIn.set(fwdStates.subarray(t * H, (t + 1) * H), H);
    headIn.set(bwdStates.subarray(t * H, (t + 1) * H), H * 2);
    headIn.set(globalCtx, H * 3);

    linearLayer(headIn, 0, H * 4, weights.headFc1Weight, weights.headFc1Bias, headMid, 0, H * 2, true);
    headHidden.set(headMid, t * H * 2);
    linearLayer(headMid, 0, H * 2, weights.headFc2Weight, weights.headFc2Bias, headOut, 0, NUM_ROLES + 1, false);

    let maxLogit = -Infinity;
    let bestRole = 0;
    for (let r = 0; r < NUM_ROLES; r++) {
      const l = headOut[r];
      roleLogits[t * NUM_ROLES + r] = l;
      if (l > maxLogit) {
        maxLogit = l;
        bestRole = r;
      }
    }
    roleIds[t] = bestRole;

    let expSum = 0;
    for (let r = 0; r < NUM_ROLES; r++) {
      const e = Math.exp(roleLogits[t * NUM_ROLES + r] - maxLogit);
      roleProbs[t * NUM_ROLES + r] = e;
      expSum += e;
    }
    for (let r = 0; r < NUM_ROLES; r++) {
      roleProbs[t * NUM_ROLES + r] /= expSum;
    }

    const bLogit = headOut[NUM_ROLES];
    boundaryLogits[t] = bLogit;
    boundaryProbs[t] = sigmoid(bLogit);
  }

  return {
    roleLogits,
    roleProbs,
    roleIds,
    boundaryLogits,
    boundaryProbs,
    fwdGateMean,
    bwdGateMean,
    fwdStates,
    bwdStates,
    headHidden,
  };
}

export function cleanNumericValue(raw: string, fieldName: string): string | null {
  if (raw.trim().startsWith("$") && !/(?:_usd|price|cost|revenue|budget|mrr|arr)$/.test(fieldName)) return null;
  const s = raw.trim().toLowerCase().replace(/^\$/, "");
  const m = /^(-?\d+(?:\.\d+)?)(ms|kb|mb|gb|s|m|h|usd|k)?$/.exec(s);
  if (!m) return null;
  const [, numStr, suffix] = m;
  const unit = fieldName.endsWith("_ms") ? "ms" : fieldName.endsWith("_min") ? "m" : fieldName.endsWith("_kb") ? "kb" : fieldName.endsWith("_usd") ? "usd" : "";
  const scale: Record<string, Record<string, number>> = {
    ms: { ms: 1, s: 1000, m: 60000, h: 3600000 },
    m: { ms: 1 / 60000, s: 1 / 60, m: 1, h: 60 },
    kb: { kb: 1, mb: 1024, gb: 1048576 },
    usd: { usd: 1, k: 1000 },
    "": { k: 1000 },
  };
  if (suffix && scale[unit]?.[suffix] === undefined) return null;
  return String(Number(numStr) * (suffix ? scale[unit][suffix] : 1));
}

function inferOperatorFromToken(norm: string): FilterClause["cmp"] | null {
  if (norm === ">=" || norm === "gte" || norm === "at_least" || norm === "atleast" || norm === "min") return "gte";
  if (norm === "<=" || norm === "lte" || norm === "at_most" || norm === "atmost" || norm === "max") return "lte";
  if (LEX_NEQ_SET.has(norm)) return "neq";
  if (LEX_GT_SET.has(norm)) return "gt";
  if (LEX_LT_SET.has(norm)) return "lt";
  if (LEX_CONTAINS_SET.has(norm)) return "contains";
  if (LEX_BEFORE_SET.has(norm)) return "before";
  if (LEX_AFTER_SET.has(norm)) return "after";
  if (LEX_EQ_SET.has(norm)) return "eq";
  return null;
}

function resolveClauseSpan(
  span: Array<{ meta: TokenFeatureMeta; role: RoleName }>,
  schema: SchemaDef,
): FilterClause | null {
  if (span.length === 0) return null;

  const fieldMap = new Map<string, FieldDef>(schema.fields.map((f) => [f.name, f]));
  let negated = false;
  let cmpOp: FilterClause["cmp"] | null = null;
  let explicitField: string | null = null;
  const valueMetas: TokenFeatureMeta[] = [];

  for (const { meta, role } of span) {
    if (/^['"]/.test(meta.token.text)) { valueMetas.push(meta); continue; }
    if (meta.boolNegPrefix) negated = true;
    if (role === "NEGATE" || LEX_NEGATE_SET.has(meta.token.norm)) {
      negated = true;
      continue;
    }
    const mappedCmp = inferOperatorFromToken(meta.token.norm) ?? ROLE_TO_CMP[role];
    if (mappedCmp) {
      if (cmpOp === null || cmpOp === "eq") cmpOp = mappedCmp;
      continue;
    }
    if (role === "FIELD") {
      if (explicitField === null && meta.matchedField !== null) {
        explicitField = meta.matchedField;
      } else if (explicitField === null && meta.matchedEnumField !== null) {
        explicitField = meta.matchedEnumField;
        valueMetas.push(meta);
      }
      continue;
    }
    if (role === "VALUE") {
      valueMetas.push(meta);
    }
  }

  if (cmpOp === null) {
    for (const { meta, role } of span) {
      if (role !== "FIELD" && role !== "VALUE") {
        const inferred = inferOperatorFromToken(meta.token.norm);
        if (inferred !== null) {
          cmpOp = inferred;
          break;
        }
      }
    }
  }

  if (explicitField === null) {
    for (const { meta, role } of span) {
      if (meta.matchedField !== null && role !== "NONE") {
        explicitField = meta.matchedField;
        break;
      }
    }
  }

  if (explicitField !== null && fieldMap.has(explicitField)) {
    const fDef = fieldMap.get(explicitField)!;
    let finalCmp: FilterClause["cmp"] = cmpOp ?? "eq";

    if (fDef.kind === "BOOL") {
      let boolVal = true;
      for (const vm of valueMetas) {
        if (LEX_BOOL_FALSE_SET.has(vm.token.norm) || vm.token.norm === "0") boolVal = false;
        else if (LEX_BOOL_TRUE_SET.has(vm.token.norm) || vm.token.norm === "1") boolVal = true;
      }
      if (finalCmp === "neq") {
        finalCmp = "eq";
        negated = !negated;
      }
      if (!boolVal) negated = !negated;
      return { field: fDef.name, cmp: "eq", value: "true", negated };
    }

    if (fDef.kind === "ENUM") {
      let chosenVal: string | null = null;
      for (const vm of valueMetas) {
        if (vm.matchedEnumField === fDef.name && vm.matchedEnumValue !== null) {
          chosenVal = vm.matchedEnumValue;
          break;
        }
      }
      if (chosenVal === null) {
        for (const { meta } of span) {
          if (meta.matchedEnumField === fDef.name && meta.matchedEnumValue !== null) {
            chosenVal = meta.matchedEnumValue;
            break;
          }
        }
      }
      if (chosenVal === null && valueMetas.length > 0) {
        const rawCand = valueMetas[0].token.norm;
        if (fDef.enumValues.includes(rawCand)) chosenVal = rawCand;
      }
      if (chosenVal === null) return null;
      return { field: fDef.name, cmp: finalCmp, value: chosenVal, negated };
    }

    if (fDef.kind === "NUMBER") {
      for (const vm of valueMetas) {
        const cleaned = cleanNumericValue(vm.token.norm, fDef.name);
        if (cleaned !== null) {
          return { field: fDef.name, cmp: finalCmp, value: cleaned, negated };
        }
      }
      for (const { meta, role } of span) {
        if (role !== "FIELD") {
          const cleaned = cleanNumericValue(meta.token.norm, fDef.name);
          if (cleaned !== null) {
            return { field: fDef.name, cmp: finalCmp, value: cleaned, negated };
          }
        }
      }
      return null;
    }

    if (fDef.kind === "DATE") {
      if (valueMetas.length === 0) return null;
      const valStr = valueMetas[0].token.norm;
      if (!isIsoDate(valStr)) return null;
      return { field: fDef.name, cmp: finalCmp, value: valStr, negated };
    }

    if (fDef.kind === "TEXT") {
      const filteredVals = valueMetas
        .map((vm) => vm.token.norm)
        .filter((norm, i) => valueMetas[i].matchedField !== fDef.name && !LEX_EQ_SET.has(norm));
      if (filteredVals.length === 0) return null;
      return { field: fDef.name, cmp: finalCmp, value: filteredVals.join(" "), negated };
    }
  }

  for (const vm of valueMetas) {
    if (vm.matchedEnumField !== null && vm.matchedEnumValue !== null) {
      return {
        field: vm.matchedEnumField,
        cmp: cmpOp ?? "eq",
        value: vm.matchedEnumValue,
        negated,
      };
    }
  }

  return null;
}

export function compileTokens(
  metas: readonly TokenFeatureMeta[],
  roleIds: ArrayLike<number>,
  boundaryFlags: ArrayLike<number | boolean>,
  schema: SchemaDef,
): FilterClause[] {
  if (metas.length === 0) return [];

  const spans: Array<Array<{ meta: TokenFeatureMeta; role: RoleName }>> = [];
  let current: Array<{ meta: TokenFeatureMeta; role: RoleName }> = [];

  const currentHasContent = (): boolean =>
    current.some((x) => x.role === "FIELD" || x.role === "VALUE");

  const currentIsComplete = (): boolean => {
    const hasField = current.some((x) => x.role === "FIELD");
    const hasVal = current.some((x) => x.role === "VALUE");
    if (hasField && hasVal) return true;
    if (hasField && current.some((x) => x.role === "FIELD" && x.meta.matchedFieldKind === "BOOL")) return true;
    if (!hasField && hasVal && current.some((x) => x.role === "VALUE" && x.meta.matchedEnumField !== null)) return true;
    return false;
  };

  for (let idx = 0; idx < metas.length; idx++) {
    const meta = metas[idx];
    const rId = roleIds[idx] ?? 0;
    let role: RoleName = /^['"]/.test(meta.token.text) ? "VALUE" : ROLES[rId] ?? "NONE";
    if (!/^['"]/.test(meta.token.text) && !meta.matchedField && !meta.matchedEnumField) {
      const lexicalCmp = inferOperatorFromToken(meta.token.norm);
      if (lexicalCmp) role = ({ eq: "OP_EQ", neq: "OP_NEQ", gt: "OP_GT", gte: "OP_GT", lt: "OP_LT", lte: "OP_LT", contains: "OP_CONTAINS", before: "OP_BEFORE", after: "OP_AFTER" } as const)[lexicalCmp];
      else if (LEX_NEGATE_SET.has(meta.token.norm)) role = "NEGATE";
    }
    const isBoundary = Boolean(boundaryFlags[idx]);

    if (!/^['"]/.test(meta.token.text) && (role === "AND" || [",", ";", "and", "&&", "&"].includes(meta.token.norm))) {
      if (current.length > 0) {
        spans.push(current);
        current = [];
      }
      continue;
    }

    if (role === "NONE") continue;

    if (current.length > 0) {
      if (isBoundary && currentHasContent()) {
        spans.push(current);
        current = [];
      } else if (role === "FIELD" && currentIsComplete()) {
        spans.push(current);
        current = [];
      } else if (role === "NEGATE" && currentIsComplete()) {
        spans.push(current);
        current = [];
      } else if (
        role === "VALUE" &&
        meta.matchedEnumField !== null &&
        currentIsComplete() &&
        !current.some((x) => x.role === "FIELD" && x.meta.matchedField === meta.matchedEnumField)
      ) {
        spans.push(current);
        current = [];
      }
    }

    current.push({ meta, role });
  }

  if (current.length > 0) spans.push(current);

  const clauses: FilterClause[] = [];
  const seen = new Set<string>();
  for (const span of spans) {
    const c = resolveClauseSpan(span, schema);
    if (c !== null) {
      const key = `${c.field}|${c.cmp}|${c.value}|${c.negated}`;
      if (!seen.has(key)) {
        seen.add(key);
        clauses.push(c);
      }
    }
  }
  return clauses;
}

/** Reject a partial AST whenever the query contains meaning the compiler cannot represent. */
export function validateCompiledQuery(query: string, schema: SchemaDef, metas: readonly TokenFeatureMeta[], clauses: readonly FilterClause[]): string[] {
  if (!query.trim()) return [];
  const issues: string[] = [];
  if (/\S/.test(query.slice(metas.at(-1)?.token.end ?? 0))) issues.push(`Query exceeds ${MAX_TOKENS} tokens or contains unsupported syntax`);
  let lastEnd = 0;
  for (const m of metas) {
    if (/\S/.test(query.slice(lastEnd, m.token.start))) issues.push("Unsupported query syntax");
    lastEnd = m.token.end;
  }
  if (metas.some((m) => !/^['"]/.test(m.token.text) && ["or", "either"].includes(m.token.norm))) issues.push("OR and grouping are unsupported");
  if (clauses.length === 0) issues.push("No filter could be applied");
  const fieldByName = new Map(schema.fields.map((f) => [f.name, f]));
  const groups: TokenFeatureMeta[][] = [];
  let group: TokenFeatureMeta[] = [];
  for (let i = 0; i < metas.length; i++) {
    const m = metas[i];
    if (LEX_AND_SET.has(m.token.norm) && !/^['"]/.test(m.token.text)) {
      if (i === 0 || i === metas.length - 1 || group.length === 0) issues.push("Dangling connector");
      if (group.length) groups.push(group);
      group = [];
    }
    else group.push(m);
  }
  if (group.length) groups.push(group);
  for (const segment of groups) {
    const fields = new Set(segment.map((m) => m.matchedField ?? m.matchedEnumField).filter(Boolean));
    const operators = segment.flatMap((m, i) => !/^['"]/.test(m.token.text) && inferOperatorFromToken(m.token.norm) ? [{ index: i, cmp: inferOperatorFromToken(m.token.norm)! }] : []);
    if (operators.length > 1) issues.push("Conflicting comparators");
    for (const operator of operators) {
      const fieldTokens = segment.flatMap((m, i) => m.matchedField || m.matchedEnumField ? [{ index: i, name: (m.matchedField ?? m.matchedEnumField)! }] : []);
      const nearest = fieldTokens.sort((a, b) => Math.abs(a.index - operator.index) - Math.abs(b.index - operator.index))[0];
      const kind = nearest && fieldByName.get(nearest.name)?.kind;
      const allowed = kind === "NUMBER" ? ["eq", "neq", "gt", "gte", "lt", "lte"] :
        kind === "BOOL" || kind === "ENUM" ? ["eq", "neq"] :
        kind === "TEXT" ? ["eq", "neq", "contains"] :
        kind === "DATE" ? ["eq", "neq", "before", "after", "gt", "gte", "lt", "lte"] : [];
      if (!allowed.includes(operator.cmp)) issues.push(`Comparator ${operator.cmp} is unsupported for ${kind ?? "this field"}`);
    }
    const numericOrTextField = segment.some((m) => m.matchedField && fieldByName.get(m.matchedField)?.kind !== "BOOL" && m.matchedEnumField === null);
    const hasValue = segment.some((m) => m.matchedEnumField || clauses.some((c) => {
      const f = fieldByName.get(c.field);
      return f?.kind === "NUMBER" ? cleanNumericValue(m.token.norm, f.name) === c.value :
        f?.kind === "DATE" ? m.token.norm === c.value :
        f?.kind === "TEXT" ? c.value === m.token.norm || c.value.split(/\s+/).includes(m.token.norm) : false;
    }));
    if (numericOrTextField && !hasValue) issues.push("Unapplied field value");
    if (segment.filter((m) => !/^['"]/.test(m.token.text) && LEX_NEGATE_SET.has(m.token.norm)).length > 1) issues.push("Ambiguous repeated negation");
    for (const field of fields) if (!clauses.some((c) => c.field === field)) issues.push(`Unapplied field: ${field}`);
    if (segment.some((m) => !/^['"]/.test(m.token.text) && LEX_NEGATE_SET.has(m.token.norm)) && !clauses.some((c) => c.negated && segment.some((m) => m.matchedField === c.field || m.matchedEnumField === c.field))) issues.push("Unapplied negation");
    if (segment.some((m) => !/^['"]/.test(m.token.text) && inferOperatorFromToken(m.token.norm)) && !clauses.some((c) => segment.some((m) => m.matchedField === c.field || m.matchedEnumField === c.field))) issues.push("Unapplied comparator");
    if (fields.size === 0 && segment.some((m) => !LEX_FILLER_SET.has(m.token.norm))) issues.push("Unapplied query segment");
  }
  for (const m of metas) {
    const word = m.token.norm;
    if (m.matchedEnumField && m.matchedEnumValue && !clauses.some((c) => c.field === m.matchedEnumField && c.value === m.matchedEnumValue)) {
      issues.push(`Unapplied value: ${m.token.text}`);
      continue;
    }
    if (/^['"]/.test(m.token.text)) {
      if (!clauses.some((c) => fieldByName.get(c.field)?.kind === "TEXT" && c.value.toLowerCase().includes(word))) issues.push(`Unrecognized value: ${m.token.text}`);
      continue;
    }
    if (m.matchedField || m.matchedEnumField || word === schema.name || LEX_FILLER_SET.has(word) || LEX_AND_SET.has(word) ||
      LEX_EQ_SET.has(word) || LEX_NEQ_SET.has(word) || LEX_GT_SET.has(word) || LEX_LT_SET.has(word) ||
      LEX_CONTAINS_SET.has(word) || LEX_BEFORE_SET.has(word) || LEX_AFTER_SET.has(word) ||
      LEX_NEGATE_SET.has(word) || LEX_BOOL_TRUE_SET.has(word) || LEX_BOOL_FALSE_SET.has(word)) continue;
    if (clauses.some((c) => {
      const f = fieldByName.get(c.field);
      return f?.kind === "TEXT" ? c.value.toLowerCase().split(/\s+/).includes(word) :
        f?.kind === "NUMBER" ? cleanNumericValue(word, f.name) === c.value :
        f?.kind === "DATE" ? c.value === word : false;
    })) continue;
    issues.push(`Unrecognized query word: ${m.token.text}`);
  }
  return issues;
}

export class SieveModel {
  readonly weights: DequantizedWeights;
  readonly rawPayload: WeightsPayload;
  private readonly originalHeadFc2Weight: Float32Array;

  constructor(payload: WeightsPayload = rawWeightsJson as unknown as WeightsPayload) {
    this.rawPayload = payload;
    this.weights = loadDequantizedWeights(payload);
    this.originalHeadFc2Weight = new Float32Array(this.weights.headFc2Weight);
  }

  resetWeights(): void {
    this.weights.headFc2Weight.set(this.originalHeadFc2Weight);
  }

  /**
   * Experimental blind magnitude perturbation of headFc2Weight.
   */
  applyBlindWeightPerturbation(fraction: number = 0.15, scale: number = -1.25): void {
    const w = this.weights.headFc2Weight;
    const indices = Array.from({ length: w.length }, (_, i) => i);
    indices.sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]));
    const k = Math.max(1, Math.floor(w.length * fraction));
    for (let i = 0; i < k; i++) {
      w[indices[i]] *= scale;
    }
  }

  /**
   * Experimental supervised rank-1 output-head edit using calibration examples.
   * This low-level API can alter role predictions; accepted parses still require lexical validation.
   */
  applyRank1HeadSurgery(
    calibrationSamples: ReadonlyArray<{ query: string; schema: SchemaDef; tokenIndex: number }>,
    targetRole: RoleName = "OP_GT",
    alpha: number = 5.0,
  ): { editedFloats: number; pctOfModel: number; elapsedMs: number } {
    if (calibrationSamples.length === 0 || !Number.isFinite(alpha) || alpha <= 0 ||
      !ROLES.includes(targetRole) || targetRole === "NEGATE" || targetRole === "OP_BEFORE") {
      throw new Error("Provide calibration samples, a positive finite gain, and a distinct target role");
    }
    for (const sample of calibrationSamples) {
      if (!Number.isInteger(sample.tokenIndex) || sample.tokenIndex < 0 || sample.tokenIndex >= featurizeQuery(sample.query, sample.schema, false).length) {
        throw new Error(`Invalid calibration token index: ${sample.tokenIndex}`);
      }
    }
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const dim = this.weights.hidden * 2; // 64
    const targetVectors: Float64Array[] = [];
    const bgVectors: Float64Array[] = [];

    for (const sample of calibrationSamples) {
      const metas = featurizeQuery(sample.query, sample.schema, false);
      const T = metas.length;
      const flatFeatures = new Int32Array(T * FEATURES_PER_TOKEN);
      for (let t = 0; t < T; t++) {
        for (let f = 0; f < FEATURES_PER_TOKEN; f++) {
          flatFeatures[t * FEATURES_PER_TOKEN + f] = metas[t].featureIds[f];
        }
      }
      const fwd = forwardTokensCPU(flatFeatures, T, this.weights);
      for (let t = 0; t < T; t++) {
        const slice = fwd.headHidden.subarray(t * dim, (t + 1) * dim);
        const v = new Float64Array(dim);
        for (let c = 0; c < dim; c++) v[c] = slice[c];
        if (t === sample.tokenIndex) {
          targetVectors.push(v);
        } else {
          bgVectors.push(v);
        }
      }
    }

    // Also include standard filler/stop-word activations in bgVectors so NONE/OP_LT/NEGATE tokens are orthogonalized
    if (calibrationSamples.length > 0) {
      const refSchema = calibrationSamples[0].schema;
      const numField = refSchema.fields.find((f) => f.kind === "NUMBER")?.name ?? "revisions";
      const fillerQueries = [
        `show me tickets where the ${refSchema.fields[0]?.name ?? "status"} was ${refSchema.fields[0]?.enumValues[0] ?? "active"} and not ${refSchema.fields[1]?.enumValues[0] ?? "closed"}`,
        `the ${refSchema.fields[0]?.name ?? "status"} is ${refSchema.fields[0]?.enumValues[0] ?? "active"} and ${numField} under 10 and ${numField} over 5`,
        `${refSchema.fields[0]?.name ?? "status"}: ${refSchema.fields[0]?.enumValues[0] ?? "active"}, ${numField} > 50, not ${refSchema.fields[1]?.enumValues[0] ?? "closed"}`,
        "please list all records that have the value and are active",
      ];
      for (const fq of fillerQueries) {
        const metas = featurizeQuery(fq, refSchema, false);
        const T = metas.length;
        if (T === 0) continue;
        const flatFeatures = new Int32Array(T * FEATURES_PER_TOKEN);
        for (let t = 0; t < T; t++) {
          for (let f = 0; f < FEATURES_PER_TOKEN; f++) {
            flatFeatures[t * FEATURES_PER_TOKEN + f] = metas[t].featureIds[f];
          }
        }
        const fwd = forwardTokensCPU(flatFeatures, T, this.weights);
        for (let t = 0; t < T; t++) {
          const slice = fwd.headHidden.subarray(t * dim, (t + 1) * dim);
          const v = new Float64Array(dim);
          for (let c = 0; c < dim; c++) v[c] = slice[c];
          bgVectors.push(v);
        }
      }
    }

    // Build an orthonormal basis for background non-target token activations via modified Gram-Schmidt
    const basis: Float64Array[] = [];
    for (const bg of bgVectors) {
      const u = new Float64Array(bg);
      for (const b of basis) {
        let dot = 0;
        for (let c = 0; c < dim; c++) dot += u[c] * b[c];
        for (let c = 0; c < dim; c++) u[c] -= dot * b[c];
      }
      let norm = 0;
      for (let c = 0; c < dim; c++) norm += u[c] * u[c];
      norm = Math.sqrt(norm);
      if (norm > 1e-3 && basis.length < 40) {
        for (let c = 0; c < dim; c++) u[c] /= norm;
        basis.push(u);
      }
    }

    // Orthogonalize each target vector against the background subspace & accumulate bounded rank-1 update
    const deltaRow = new Float64Array(dim);
    const targetBasis: Float64Array[] = [];
    for (const hTarget of targetVectors) {
      const vSurg = new Float64Array(hTarget);
      for (const b of basis) {
        let dot = 0;
        for (let c = 0; c < dim; c++) dot += vSurg[c] * b[c];
        for (let c = 0; c < dim; c++) vSurg[c] -= dot * b[c];
      }
      for (const tb of targetBasis) {
        let dot = 0;
        for (let c = 0; c < dim; c++) dot += vSurg[c] * tb[c];
        for (let c = 0; c < dim; c++) vSurg[c] -= dot * tb[c];
      }
      let vNorm = 0;
      for (let c = 0; c < dim; c++) vNorm += vSurg[c] * vSurg[c];
      vNorm = Math.sqrt(vNorm);
      if (vNorm > 1e-4) {
        for (let c = 0; c < dim; c++) vSurg[c] /= vNorm;
        targetBasis.push(vSurg);
        let dotTarget = 0;
        for (let c = 0; c < dim; c++) dotTarget += hTarget[c] * vSurg[c];
        // Ridge-bounded closed-form gain: never amplifies small residuals
        const sign = dotTarget >= 0 ? 1.0 : -1.0;
        const gain = sign * alpha * 2.1;
        for (let c = 0; c < dim; c++) {
          deltaRow[c] += gain * vSurg[c];
        }
      }
    }

    const targetRoleId = ROLES.indexOf(targetRole);
    const negateRoleId = ROLES.indexOf("NEGATE");
    const beforeRoleId = ROLES.indexOf("OP_BEFORE");
    const editedFloats = deltaRow.some((value) => value !== 0) ? dim * 3 : 0;
    if (editedFloats) {
      const targetRowOffset = targetRoleId * dim;
      const negateRowOffset = negateRoleId * dim;
      const beforeRowOffset = beforeRoleId * dim;
      for (let c = 0; c < dim; c++) {
        this.weights.headFc2Weight[targetRowOffset + c] += deltaRow[c];
        this.weights.headFc2Weight[negateRowOffset + c] -= 0.7 * deltaRow[c];
        this.weights.headFc2Weight[beforeRowOffset + c] -= 0.7 * deltaRow[c];
      }
    }

    const elapsedMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
    return {
      editedFloats,
      pctOfModel: Number(((editedFloats * 100) / this.weights.paramCount).toFixed(3)),
      elapsedMs: Number(elapsedMs.toFixed(3)),
    };
  }

  parse(query: string, schema: SchemaDef): NeuralParseResult {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const metas = featurizeQuery(query, schema, false);
    const T = metas.length;

    if (T === 0) {
      return {
        query,
        schemaName: schema.name,
        tokens: [],
        clauses: [],
        accepted: !query.trim(),
        issues: query.trim() ? ["No filter could be applied"] : [],
        latencyMs: 0,
        fwdStates: new Float32Array(0),
        bwdStates: new Float32Array(0),
      };
    }

    const flatFeatures = new Int32Array(T * FEATURES_PER_TOKEN);
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < FEATURES_PER_TOKEN; f++) {
        flatFeatures[t * FEATURES_PER_TOKEN + f] = metas[t].featureIds[f];
      }
    }

    const fwd = forwardTokensCPU(flatFeatures, T, this.weights);
    const boundaryFlags = new Uint8Array(T);
    for (let t = 0; t < T; t++) {
      boundaryFlags[t] = fwd.boundaryProbs[t] >= 0.5 ? 1 : 0;
    }

    const compiled = compileTokens(metas, fwd.roleIds, boundaryFlags, schema);
    const issues = validateCompiledQuery(query, schema, metas, compiled);
    const clauses = issues.length ? [] : compiled;
    const latencyMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;

    const tokens: TokenTrace[] = metas.map((m, t) => {
      const roleId = fwd.roleIds[t];
      const probs = Array.from(fwd.roleProbs.subarray(t * NUM_ROLES, (t + 1) * NUM_ROLES));
      return {
        text: m.token.text,
        norm: m.token.norm,
        start: m.token.start,
        end: m.token.end,
        featureIds: m.featureIds,
        roleId,
        role: ROLES[roleId],
        confidence: Number(probs[roleId].toFixed(5)),
        roleProbs: probs.map((p) => Number(p.toFixed(5))),
        boundaryProb: Number(fwd.boundaryProbs[t].toFixed(5)),
        isBoundary: boundaryFlags[t] === 1,
        fwdGateMean: Number(fwd.fwdGateMean[t].toFixed(5)),
        bwdGateMean: Number(fwd.bwdGateMean[t].toFixed(5)),
        matchedField: m.matchedField,
        matchedEnumField: m.matchedEnumField,
        matchedEnumValue: m.matchedEnumValue,
        fieldMatchQuality: m.fieldMatchQuality,
      };
    });

    return {
      query,
      schemaName: schema.name,
      tokens,
      clauses,
      accepted: issues.length === 0,
      issues,
      latencyMs: Number(latencyMs.toFixed(3)),
      fwdStates: fwd.fwdStates,
      bwdStates: fwd.bwdStates,
    };
  }
}
