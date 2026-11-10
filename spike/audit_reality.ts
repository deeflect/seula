import assert from "node:assert/strict";
import {
  ALL_SCHEMAS, CentroidMemory, SieveModel, featurizeQuery, inferSchemaFromRows,
  normalizeImportedRow, parseCsvOrJsonToRows,
} from "../src/engine";
import { DATASET_PRESETS, filterDatasetRows, type DatasetRow } from "../src/datasets";
import type { SchemaDef } from "../src/engine/schema";

const model = new SieveModel();
const client = ALL_SCHEMAS.client_work;
const rows = DATASET_PRESETS.client_work.rows;
const parse = (query: string, schema = client) => model.parse(query, schema);
const accepted = (query: string, schema = client) => {
  const result = parse(query, schema);
  assert.equal(result.accepted, true, `${query}: ${result.issues.join("; ")}`);
  return result.clauses;
};
const rejected = (query: string, schema = client) => {
  const result = parse(query, schema);
  assert.equal(result.accepted, false, query);
  assert.deepEqual(result.clauses, []);
  assert.ok(result.issues.length > 0);
};

assert.deepEqual(accepted("revisions >= 4"), [{ field: "revisions", cmp: "gte", value: "4", negated: false }]);
assert.deepEqual(accepted("revisions <= 4"), [{ field: "revisions", cmp: "lte", value: "4", negated: false }]);
assert.ok(filterDatasetRows(rows, accepted("revisions >= 4")).some((row) => row.fields.revisions === 4));
rejected("vibe delusional or panicking");
rejected("without any delusional or panicking vibe");
rejected(`${Array(24).fill("please").join(" ")} revisions over 4`);
for (const query of ["revisions topping 4", "revisions banana 4", "revisions > 4 and revisions", "revisions > 4 and revisions <", "vibe = chill and vibe =", "vibe chill panicking", "not not paid", "revisions over 4 banana", "revisions before 5", "revisions after 5", "revisions contains 5", "revisions > < 5", "vibe = != chill", "revisions > 5 <", "vibe chill and", "and vibe chill", "vibe chill and and ask ai_pivot"]) rejected(query);
assert.deepEqual(accepted("vibe isn't delusional"), [{ field: "vibe", cmp: "neq", value: "delusional", negated: false }]);
assert.deepEqual(accepted("vibe is not delusional"), [{ field: "vibe", cmp: "eq", value: "delusional", negated: true }]);
assert.deepEqual(accepted("revisions min 4"), [{ field: "revisions", cmp: "gte", value: "4", negated: false }]);
assert.deepEqual(accepted("revisions max 4"), [{ field: "revisions", cmp: "lte", value: "4", negated: false }]);
assert.deepEqual(accepted("more than 5 points", ALL_SCHEMAS.issues), [{ field: "points", cmp: "gt", value: "5", negated: false }]);
assert.deepEqual(accepted("fewer than 5 points", ALL_SCHEMAS.issues), [{ field: "points", cmp: "lt", value: "5", negated: false }]);
assert.deepEqual(accepted("more than 20 duration_min", ALL_SCHEMAS.incidents), [{ field: "duration_min", cmp: "gt", value: "20", negated: false }]);
assert.deepEqual(accepted("fewer than 300 latency_ms", ALL_SCHEMAS.deployments), [{ field: "latency_ms", cmp: "lt", value: "300", negated: false }]);
assert.deepEqual(accepted("no more than 4 revisions"), [{ field: "revisions", cmp: "gt", value: "4", negated: true }]);
assert.deepEqual(filterDatasetRows(rows, accepted("no more than 4 revisions")).map((row) => row.id), ["CL-106", "CL-107"]);
const mrrOnly = inferSchemaFromRows("mrr_only", [{ mrr: 1000 }]);
rejected("arr over 12000", mrrOnly);
assert.deepEqual(accepted("mrr over 12000", mrrOnly), [{ field: "mrr", cmp: "gt", value: "12000", negated: false }]);
assert.deepEqual(filterDatasetRows(rows, [{ field: "revisions", cmp: "contains", value: "4", negated: true }]), []);
for (const query of ["points before 5", "points after 5", "points contains 5", "points > < 5", "status = != open", "points > 5 <", "status open and", "and status open", "status open and and priority p0", "locked > true", "status > open"]) rejected(query, ALL_SCHEMAS.issues);
assert.deepEqual(accepted("status is not open", ALL_SCHEMAS.issues), [{ field: "status", cmp: "eq", value: "open", negated: true }]);
assert.deepEqual(accepted("status open and priority p0", ALL_SCHEMAS.issues), [
  { field: "status", cmp: "eq", value: "open", negated: false },
  { field: "priority", cmp: "eq", value: "p0", negated: false },
]);

const memory = new CentroidMemory(96, 0.58);
for (const shortcut of DATASET_PRESETS.client_work.defaultShortcuts) memory.teach(shortcut.id, client, shortcut.examples, shortcut.clauses);
const mixed = memory.splice("make it pop and revisions over 4 not paid", client);
assert.equal(mixed.matched, true);
assert.deepEqual([...mixed.splicedClauses, ...accepted(mixed.residualQuery)], [
  { field: "ask", cmp: "eq", value: "pop_more", negated: false },
  { field: "vibe", cmp: "eq", value: "delusional", negated: false },
  { field: "revisions", cmp: "gt", value: "4", negated: false },
  { field: "invoice_state", cmp: "eq", value: "paid", negated: true },
]);
assert.equal(memory.query("not make it pop", client).matched, false);
assert.equal(memory.splice("not make it pop", client).matched, false);
rejected(memory.splice("not make it pop", client).residualQuery);
assert.equal(memory.splice("scope creep nightmare paid", client).residualQuery, "paid");

const deploymentRows: DatasetRow[] = [
  { id: "fast", title: "fast", subtitle: "", fields: { latency_ms: 500 }, badges: [], rightMeta: "", rightSub: "" },
  { id: "slow", title: "slow", subtitle: "", fields: { latency_ms: 2500 }, badges: [], rightMeta: "", rightSub: "" },
];
assert.deepEqual(accepted("latency_ms over 2s", ALL_SCHEMAS.deployments), [{ field: "latency_ms", cmp: "gt", value: "2000", negated: false }]);
assert.deepEqual(filterDatasetRows(deploymentRows, accepted("latency_ms over 2s", ALL_SCHEMAS.deployments)).map((row) => row.id), ["slow"]);
rejected("latency_ms over 2kb", ALL_SCHEMAS.deployments);

const raw = parseCsvOrJsonToRows('Name,Company,Status,Created,Price,Budget\nAlice,"Acme (West)",open,2026-11-07,10,100\nBob,"Beta, Inc",closed,2026-11-08,20,200');
const schema = inferSchemaFromRows("custom", raw);
assert.deepEqual(schema.fields.map((field) => field.name), ["name", "company", "status", "created", "price", "budget"]);
const customRows: DatasetRow[] = raw.map((row, i) => ({ id: String(i), title: "", subtitle: "", fields: normalizeImportedRow(row, schema), badges: [], rightMeta: "", rightSub: "" }));
assert.deepEqual(filterDatasetRows(customRows, accepted("name not Alice", schema)).map((row) => row.id), ["1"]);
assert.deepEqual(filterDatasetRows(customRows, accepted('company = "Acme (West)"', schema)).map((row) => row.id), ["0"]);
assert.deepEqual(accepted('name = "not"', schema), [{ field: "name", cmp: "eq", value: "not", negated: false }]);
assert.deepEqual(accepted("budget > 150", schema), [{ field: "budget", cmp: "gt", value: "150", negated: false }]);
assert.deepEqual(filterDatasetRows(customRows, accepted("created before 2026-11-08", schema)).map((row) => row.id), ["0"]);
assert.deepEqual(filterDatasetRows(customRows, accepted("created >= 2026-11-08", schema)).map((row) => row.id), ["1"]);
assert.deepEqual(filterDatasetRows(customRows, accepted('company contains "west"', schema)).map((row) => row.id), ["0"]);
assert.deepEqual(filterDatasetRows(customRows, accepted("name not Alice", schema).map((c) => ({ ...c, field: "missing" }))), []);
assert.deepEqual(parseCsvOrJsonToRows('name,note\nAlice,"line one\nline ""two"""')[0].note, 'line one\nline "two"');
assert.throws(() => parseCsvOrJsonToRows('name,note\nAlice,"unclosed'));
assert.throws(() => parseCsvOrJsonToRows('{broken'));
assert.throws(() => inferSchemaFromRows("bad", [{ Status: "open", status: "closed" }]));

// Feature collisions remain a limitation of the frozen role model, not evidence of learned word meaning.
const words = ["topping", "falling", "beneath"];
const features = words.map((word) => featurizeQuery(`revisions ${word} 4`, client, false)[1].featureIds);
for (const ids of features.slice(1)) assert.deepEqual(ids, features[0]);
assert.throws(() => model.applyRank1HeadSurgery([]));
assert.throws(() => model.applyRank1HeadSurgery([{ query: "revisions over 4", schema: client, tokenIndex: 99 }]));
assert.throws(() => model.applyRank1HeadSurgery([{ query: "revisions over 4", schema: client, tokenIndex: 1 }], "OP_GT", NaN));
assert.throws(() => model.applyRank1HeadSurgery([{ query: "revisions over 4", schema: client, tokenIndex: 1 }], "NEGATE"));
console.log("audit regressions: passed; comparator feature collision retained");
