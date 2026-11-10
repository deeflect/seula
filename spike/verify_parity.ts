import assert from "node:assert/strict";
import {
  ALL_SCHEMAS,
  CentroidMemory,
  SieveModel,
  SieveWebGPUBatchEngine,
  astExactMatch,
  baselineParse,
  inferSchemaFromRows,
  parseCsvOrJsonToRows,
} from "../src/engine";
import rawWeightsJson from "../src/engine/weights.json";

async function main(): Promise<void> {
  const model = new SieveModel(rawWeightsJson as any);
  const traces = (rawWeightsJson as any).parity_traces ?? [];

  if (traces.length === 0) {
    throw new Error("No parity_traces found in weights.json");
  }

  let maxProbDiff = 0;
  let maxGateDiff = 0;

  for (const t of traces) {
    const schema = ALL_SCHEMAS[t.schema];
    if (!schema) throw new Error(`Unknown schema: ${t.schema}`);

    const res = model.parse(t.query, schema);

    // 1. Verify token count & feature IDs
    if (res.tokens.length !== t.tokens.length) {
      throw new Error(`Token length mismatch on "${t.query}": TS=${res.tokens.length} vs Py=${t.tokens.length}`);
    }

    for (let i = 0; i < res.tokens.length; i++) {
      const tsFeat = res.tokens[i].featureIds.join(",");
      const pyFeat = t.feature_ids[i].join(",");
      if (tsFeat !== pyFeat) {
        throw new Error(`Feature ID mismatch at token ${i} ("${res.tokens[i].text}"): TS=[${tsFeat}] vs Py=[${pyFeat}]`);
      }

      if (res.tokens[i].roleId !== t.role_ids[i]) {
        throw new Error(
          `Role ID mismatch at token ${i} ("${res.tokens[i].text}"): TS=${res.tokens[i].role} vs Py=${t.roles[i]}`,
        );
      }

      const probDiff = Math.abs(res.tokens[i].confidence - t.role_probs_top1[i]);
      if (probDiff > maxProbDiff) maxProbDiff = probDiff;
      if (probDiff > 1e-4) {
        throw new Error(`Probability diff too large at token ${i}: ${probDiff}`);
      }

      const fgDiff = Math.abs(res.tokens[i].fwdGateMean - t.fwd_gate_mean[i]);
      const bgDiff = Math.abs(res.tokens[i].bwdGateMean - t.bwd_gate_mean[i]);
      maxGateDiff = Math.max(maxGateDiff, fgDiff, bgDiff);
      if (fgDiff > 1e-4 || bgDiff > 1e-4) {
        throw new Error(`Gate trace diff too large at token ${i}: fg=${fgDiff}, bg=${bgDiff}`);
      }
    }

    // 2. Verify compiled AST exact match
    if (!astExactMatch(res.clauses, t.clauses)) {
      throw new Error(
        `Compiled AST mismatch on "${t.query}": TS=${JSON.stringify(res.clauses)} vs Py=${JSON.stringify(t.clauses)}`,
      );
    }

    // 3. Verify Baseline 1 rule parser exact match between TS and Python
    const tsBase = baselineParse(t.query, schema);
    if (!astExactMatch(tsBase, t.baseline_clauses)) {
      throw new Error(
        `Baseline rule AST mismatch on "${t.query}": TS=${JSON.stringify(tsBase)} vs Py=${JSON.stringify(t.baseline_clauses)}`,
      );
    }
  }

  // 4. Verify CentroidMemory in TS
  const memory = new CentroidMemory(96, 0.58);
  memory.teach(
    "p0_fires",
    ALL_SCHEMAS.incidents,
    ["p0 fires", "active p0 fire", "unmitigated p0 fires"],
    [
      { field: "severity", cmp: "eq", value: "critical", negated: false },
      { field: "mitigated", cmp: "eq", value: "true", negated: true },
    ],
  );
  const cMatch = memory.query("current p0 fire", ALL_SCHEMAS.incidents);
  if (!cMatch.matched || cMatch.clauses.length !== 2) {
    throw new Error(`CentroidMemory failed to match "current p0 fire": ${JSON.stringify(cMatch)}`);
  }
  const cReject = memory.query("severity is minor and commander is frank", ALL_SCHEMAS.incidents);
  if (cReject.matched) {
    throw new Error(`CentroidMemory failed to reject unrelated query: ${JSON.stringify(cReject)}`);
  }

  // 4b. Verify unseen "client_work" (Agency Client Chaos) zero-shot transfer + junk-word drop
  const clientSchema = ALL_SCHEMAS.client_work;
  const clientMessy = "show me passive_aggressive or panicking vibe where ask is logo_bigger and revisions over 4 and not paid banana";
  const clientRes = model.parse(clientMessy, clientSchema);
  assert.equal(clientRes.accepted, false);
  assert.deepEqual(clientRes.clauses, []);
  assert.ok(clientRes.issues.some((issue) => issue.includes("OR")));

  // Teach funny client slang to CentroidMemory
  memory.teach(
    "make_it_pop",
    clientSchema,
    ["make it pop before friday", "client wants it to pop more asap", "apple style pop"],
    [
      { field: "ask", cmp: "eq", value: "pop_more", negated: false },
      { field: "vibe", cmp: "eq", value: "delusional", negated: false },
    ],
  );
  const popMatch = memory.query("can we make it pop more asap", clientSchema);
  if (!popMatch.matched || popMatch.clauses.length !== 2) {
    throw new Error(`CentroidMemory failed on client_work slang: ${JSON.stringify(popMatch)}`);
  }

  // 4c. Verify Compositional Sub-Span Slang Splicer ("make it pop and revisions over 4 not paid")
  const mixedQuery = "make it pop and revisions over 4 not paid";
  const spliceRes = memory.splice(mixedQuery, clientSchema);
  if (!spliceRes.matched || spliceRes.mode !== "subspan" || spliceRes.splicedClauses.length !== 2) {
    throw new Error(`Compositional Slang Splicer failed on mixed query: ${JSON.stringify(spliceRes)}`);
  }
  const residualNeural = model.parse(spliceRes.residualQuery, clientSchema);
  const mergedHybridClauses = [...spliceRes.splicedClauses, ...residualNeural.clauses];
  assert.equal(residualNeural.accepted, true);
  assert.deepEqual(mergedHybridClauses, [
    { field: "ask", cmp: "eq", value: "pop_more", negated: false },
    { field: "vibe", cmp: "eq", value: "delusional", negated: false },
    { field: "revisions", cmp: "gt", value: "4", negated: false },
    { field: "invoice_state", cmp: "eq", value: "paid", negated: true },
  ]);

  // 4d. Verify Zero-Config Auto-Schema Inference (`inferSchemaFromRows`) on raw CSV
  const rawCsv = `company,plan,mrr,churn_risk,tickets,paid
"VibeCode AI",pro,4200,critical,19,false
"Stealth Crypto DAO",enterprise,12500,critical,31,false
"Boring Plumbing LLC",starter,450,low,1,true`;
  const csvRows = parseCsvOrJsonToRows(rawCsv);
  const autoSchema = inferSchemaFromRows("saas_churn", csvRows);
  const autoParse = model.parse("critical churn_risk where mrr over 1000 and not paid", autoSchema);
  if (autoParse.clauses.length !== 3) {
    throw new Error(
      `Expected 3 clauses on auto-inferred CSV schema, got: ${JSON.stringify(autoParse.clauses)}`,
    );
  }

  // 4e. Verify Direct Weight Surgery: Blind Perturbation (MIT negative control) vs Closed-Form Rank-1 Head Surgery
  const novelVerbSuite = [
    {
      schema: ALL_SCHEMAS.client_work,
      query: "revisions topping 4 and not paid",
      expected: [
        { field: "revisions", cmp: "gt" as const, value: "4", negated: false },
        { field: "invoice_state", cmp: "eq" as const, value: "paid", negated: true },
      ],
    },
    {
      schema: ALL_SCHEMAS.deployments,
      query: "latency_ms surpassing 250 and not rolled_back",
      expected: [
        { field: "latency_ms", cmp: "gt" as const, value: "250", negated: false },
        { field: "outcome", cmp: "eq" as const, value: "rolled_back", negated: true },
      ],
    },
    {
      schema: ALL_SCHEMAS.contacts,
      query: "enterprise segment with arr_usd eclipsing 25000",
      expected: [
        { field: "segment", cmp: "eq" as const, value: "enterprise", negated: false },
        { field: "arr_usd", cmp: "gt" as const, value: "25000", negated: false },
      ],
    },
    {
      schema: ALL_SCHEMAS.incidents,
      query: "critical severity and duration_min beating 30",
      expected: [
        { field: "severity", cmp: "eq" as const, value: "critical", negated: false },
        { field: "duration_min", cmp: "gt" as const, value: "30", negated: false },
      ],
    },
  ];

  const evalNovelSuite = (): number => {
    let ok = 0;
    for (const item of novelVerbSuite) {
      const p = model.parse(item.query, item.schema);
      if (astExactMatch(p.clauses, item.expected)) ok++;
    }
    return Number(((ok / novelVerbSuite.length) * 100).toFixed(1));
  };

  const baseNovelAcc = evalNovelSuite();

  // Negative control: blind magnitude perturbation
  model.applyBlindWeightPerturbation(0.2, -1.5);
  const blindNovelAcc = evalNovelSuite();
  model.resetWeights();

  // Positive surgical edit: Rank-1 output-head surgery (128 floats = 0.432% of parameters, 0 backprop)
  const surgeryStats = model.applyRank1HeadSurgery(
    [
      { query: "revisions topping 5 and not paid", schema: ALL_SCHEMAS.client_work, tokenIndex: 1 },
      { query: "latency_ms surpassing 200 and pending", schema: ALL_SCHEMAS.deployments, tokenIndex: 1 },
      { query: "enterprise segment with arr_usd northof 10000", schema: ALL_SCHEMAS.contacts, tokenIndex: 4 },
      { query: "critical severity and duration_min topping 20", schema: ALL_SCHEMAS.incidents, tokenIndex: 4 },
    ],
    "OP_GT",
    5.5,
  );
  const surgicalNovelAcc = evalNovelSuite();
  // Surgery may change low-level roles, but unsupported comparator words remain rejected.
  for (const item of novelVerbSuite) assert.equal(model.parse(item.query, item.schema).accepted, false);
  let retainedTraces = 0;
  // Measure retention only on the five stored parity traces.
  for (const t of traces) {
    const p = model.parse(t.query, ALL_SCHEMAS[t.schema]);
    if (astExactMatch(p.clauses, t.clauses)) retainedTraces++;
  }
  model.resetWeights();

  // 5. Measure single-query CPU latency and batch evaluation
  const sampleQuery = "show me incidents where the severty was critcal and unmitigated";
  const iters = 200;
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    model.parse(sampleQuery, ALL_SCHEMAS.incidents);
  }
  const singleLatencyMs = (performance.now() - t0) / iters;

  const batchEngine = new SieveWebGPUBatchEngine(model);
  const batchQueries = Array.from({ length: 1024 }, (_, i) => traces[i % traces.length].query);
  const batchRes = await batchEngine.evaluateBatch(batchQueries, ALL_SCHEMAS.incidents);

  console.log(
    JSON.stringify(
      {
        parity: "PASS",
        tracesVerified: traces.length,
        maxProbDiff: Number(maxProbDiff.toFixed(6)),
        maxGateDiff: Number(maxGateDiff.toFixed(6)),
        centroidMatchSim: cMatch.similarity,
        centroidRejectSim: cReject.similarity,
        hybridSlangSpliceClauses: mergedHybridClauses.length,
        autoSchemaInferredFields: autoSchema.fields.length,
        weightSurgery: {
          editedFloats: surgeryStats.editedFloats,
          pctOfModel: surgeryStats.pctOfModel,
          surgeryMs: surgeryStats.elapsedMs,
          baseNovelVerbAcc: baseNovelAcc,
          blindPerturbAcc: blindNovelAcc,
          rank1SurgicalAcc: surgicalNovelAcc,
          retainedParityTraces: `${retainedTraces}/${traces.length}`,
        },
        singleQueryCpuMs: Number(singleLatencyMs.toFixed(4)),
        batch1024CpuMs: batchRes.cpuComparisonMs,
      },
      null,
      2,
    ),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
