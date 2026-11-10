import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { cpus } from "node:os";
import { ALL_SCHEMAS, FEATURES_PER_TOKEN, SieveModel, compileTokens, featurizeQuery, forwardTokensCPU, tokenizeQuery } from "../src/engine";
import type { FilterClause, SchemaDef } from "../src/engine/schema";

const model = new SieveModel();
const calibration = [
  { query: "revisions topping 5 and not paid", schema: ALL_SCHEMAS.client_work, tokenIndex: 1 },
  { query: "latency_ms surpassing 200 and pending", schema: ALL_SCHEMAS.deployments, tokenIndex: 1 },
  { query: "enterprise segment with arr_usd northof 10000", schema: ALL_SCHEMAS.contacts, tokenIndex: 4 },
  { query: "critical severity and duration_min topping 20", schema: ALL_SCHEMAS.incidents, tokenIndex: 4 },
];
const cases: Array<{ query: string; schema: SchemaDef; expected: FilterClause[]; kind: string }> = [
  ["revisions topping 7", "client_work", "revisions", "gt", "7", "positive"],
  ["latency_ms surpassing 350", "deployments", "latency_ms", "gt", "350", "positive"],
  ["arr_usd eclipsing 25000", "contacts", "arr_usd", "gt", "25000", "positive"],
  ["duration_min beating 45", "incidents", "duration_min", "gt", "45", "positive"],
  ["revisions beneath 7", "client_work", "revisions", "lt", "7", "opposite"],
  ["latency_ms equaling 350", "deployments", "latency_ms", "eq", "350", "opposite"],
  ["revisions banana 7", "client_work", "revisions", "gt", "7", "unrelated"],
  ['founder = "topping"', "client_work", "founder", "eq", "topping", "quoted"],
].map(([query, schema, field, cmp, value, kind]) => ({
  query, schema: ALL_SCHEMAS[schema], expected: [{ field, cmp: cmp as FilterClause["cmp"], value, negated: false }], kind,
}));

const raw = (query: string, schema: SchemaDef): FilterClause[] => {
  const metas = featurizeQuery(query, schema, false);
  const flat = new Int32Array(metas.flatMap((m) => m.featureIds));
  assert.equal(flat.length, metas.length * FEATURES_PER_TOKEN);
  const fwd = forwardTokensCPU(flat, metas.length, model.weights);
  return compileTokens(metas, fwd.roleIds, Array.from(fwd.boundaryProbs, (p) => p >= 0.5), schema);
};
// Explicit supervised word meanings. Only replace a comparator between a matched NUMBER field and a numeric value.
const meanings: Record<string, string> = {
  topping: "over", surpassing: "over", eclipsing: "over", beating: "over",
  beneath: "under", equaling: "equals",
};
const adapt = (query: string, schema: SchemaDef): string => {
  const tokens = tokenizeQuery(query);
  const metas = featurizeQuery(query, schema, false);
  const replacements = tokens.flatMap((token, i) => {
    const word = meanings[token.norm];
    const field = schema.fields.find((f) => f.name === metas[i - 1]?.matchedField && f.kind === "NUMBER");
    return word && !/^['"]/.test(token.text) && field && /^-?\d+(?:\.\d+)?$/.test(tokens[i + 1]?.norm ?? "")
      ? [{ start: token.start, end: token.end, word }] : [];
  });
  for (const r of replacements.reverse()) query = query.slice(0, r.start) + r.word + query.slice(r.end);
  return query;
};

const collision = ["topping", "beneath"].map((word) => featurizeQuery(`revisions ${word} 4`, ALL_SCHEMAS.client_work, false)[1].featureIds);
assert.deepEqual(collision[0], collision[1]);
const matchedQueries = ["revisions topping 7", "revisions beneath 7"];
assert.deepEqual(...matchedQueries.map((query) => featurizeQuery(query, ALL_SCHEMAS.client_work, false).map((m) => m.featureIds)) as [number[][], number[][]]);
const matchedModel = new SieveModel();
const matchedCalibration = (word: string) => [{ query: `revisions ${word} 5`, schema: ALL_SCHEMAS.client_work, tokenIndex: 1 }];
const matchedStats = [
  matchedModel.applyRank1HeadSurgery(matchedCalibration("topping"), "OP_GT", 5.5),
  matchedModel.applyRank1HeadSurgery(matchedCalibration("beneath"), "OP_LT", 5.5),
];
const matchedRaw = matchedQueries.map((query) => {
  const metas = featurizeQuery(query, ALL_SCHEMAS.client_work, false);
  const fwd = forwardTokensCPU(new Int32Array(metas.flatMap((m) => m.featureIds)), metas.length, matchedModel.weights);
  return compileTokens(metas, fwd.roleIds, Array.from(fwd.boundaryProbs, (p) => p >= 0.5), ALL_SCHEMAS.client_work);
});
const retentionQueries = [
  { query: "revisions over 4", schema: ALL_SCHEMAS.client_work },
  { query: "latency_ms under 300", schema: ALL_SCHEMAS.deployments },
  { query: "status = open", schema: ALL_SCHEMAS.issues },
  { query: "not paid", schema: ALL_SCHEMAS.client_work },
];
const retentionBefore = retentionQueries.map(({ query, schema }) => new SieveModel().parse(query, schema).clauses);
const original = cases.map((item) => ({ ...item, baseRaw: raw(item.query, item.schema), baseGuarded: model.parse(item.query, item.schema) }));
const surgery = model.applyRank1HeadSurgery(calibration, "OP_GT", 5.5);
const retentionAfter = retentionQueries.map(({ query, schema }) => model.parse(query, schema).clauses);
const results = original.map(({ query, schema, expected, kind, baseRaw, baseGuarded }) => {
  const editedRaw = raw(query, schema);
  const editedGuarded = model.parse(query, schema);
  const adaptedQuery = adapt(query, schema);
  const adapted = new SieveModel().parse(adaptedQuery, schema);
  const exact = (clauses: FilterClause[]) => JSON.stringify(clauses) === JSON.stringify(expected);
  return { query, schema: schema.name, kind, expected, baseRaw, baseRawExact: exact(baseRaw),
    baseGuardedAccepted: baseGuarded.accepted, editedRaw, editedRawExact: exact(editedRaw),
    editedGuardedAccepted: editedGuarded.accepted, adaptedQuery, adapterAccepted: adapted.accepted,
    adapterClauses: adapted.clauses, adapterExact: adapted.accepted && exact(adapted.clauses),
    expectedReject: kind === "unrelated" };
});
const report = {
  note: "Agent-authored supervised dictionary; meanings are supplied, not inferred. Unrelated case expects rejection.",
  command: "npm run experiment:vocabulary", node: process.version,
  hardware: `${process.platform} ${process.arch}; ${cpus()[0]?.model ?? "unknown CPU"}`,
  modelSha256: createHash("sha256").update(readFileSync("src/engine/weights.json")).digest("hex"),
  calibration: calibration.map(({ query, schema, tokenIndex }) => ({ query, schema: schema.name, tokenIndex })),
  collision, surgery, matchedSupervision: { calibration: ["revisions topping 5 => gt", "revisions beneath 5 => lt"], stats: matchedStats,
    featureMatrixIdentical: true, queries: matchedQueries, rawClauses: matchedRaw, dictionaryClauses: matchedQueries.map((query) => new SieveModel().parse(adapt(query, ALL_SCHEMAS.client_work), ALL_SCHEMAS.client_work).clauses) },
  retention: retentionQueries.map(({ query, schema }, i) => ({ query, schema: schema.name, before: retentionBefore[i], after: retentionAfter[i], retained: JSON.stringify(retentionBefore[i]) === JSON.stringify(retentionAfter[i]) })), results,
};
writeFileSync("docs/vocabulary-results.json", JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ surgery, results: results.map(({ query, kind, baseRawExact, editedRawExact, editedGuardedAccepted, adapterAccepted, adapterExact }) => ({ query, kind, baseRawExact, editedRawExact, editedGuardedAccepted, adapterAccepted, adapterExact })) }, null, 2));
