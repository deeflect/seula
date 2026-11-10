import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { ALL_SCHEMAS, FEATURES_PER_TOKEN, LEX_AFTER_SET, LEX_AND_SET, LEX_BEFORE_SET, LEX_BOOL_FALSE_SET, LEX_BOOL_TRUE_SET, LEX_CONTAINS_SET, LEX_EQ_SET, LEX_FILLER_SET, LEX_GT_SET, LEX_LT_SET, LEX_NEGATE_SET, LEX_NEQ_SET, ROLES, SieveModel, compileTokens, featurizeQuery, forwardTokensCPU, validateCompiledQuery } from "../src/engine";
import { filterDatasetRows, type DatasetRow } from "../src/datasets";
import type { FilterClause, SchemaDef } from "../src/engine/schema";
import { oracleRows, rejectedCases, validCases } from "./challenge_cases";

const fixtureSha256 = createHash("sha256").update(readFileSync("spike/challenge_cases.ts")).digest("hex");
const modelSha256 = createHash("sha256").update(readFileSync("src/engine/weights.json")).digest("hex");
const model = new SieveModel();
const cases = validCases.map((line) => {
  const [id, query, expectedText] = line.split("|");
  const schemaName = id[0] === "I" ? "issues" : id[0] === "C" ? "client_work" : "deployments";
  const expected: FilterClause[] = expectedText.split(" & ").map((part) => {
    const [field, cmp, ...value] = part.replace(/^!/, "").split(":");
    return { field, cmp: cmp as FilterClause["cmp"], value: value.join(":"), negated: part.startsWith("!") };
  });
  return { id, schemaName, query, expected, valid: true, reason: undefined as string | undefined };
});
const invalid = rejectedCases.map((line) => {
  const [id, schemaName, query, reason] = line.split("|");
  return { id, schemaName, query, reason, expected: [], valid: false };
});
assert.equal(cases.length, 90); assert.equal(invalid.length, 10);

const canonical = (clauses: readonly FilterClause[], schema: SchemaDef): string => JSON.stringify(clauses.map((c) => {
  const field = schema.fields.find((f) => f.name === c.field);
  let { cmp, negated } = c;
  let value = c.value.toLocaleLowerCase();
  if (cmp === "neq") { cmp = "eq"; negated = !negated; }
  if (field?.kind === "BOOL" && value === "false") { value = "true"; negated = !negated; }
  if (negated && (field?.kind === "NUMBER" || field?.kind === "DATE")) {
    const opposite: Partial<Record<FilterClause["cmp"], FilterClause["cmp"]>> = {
      gt: "lte", gte: "lt", lt: "gte", lte: "gt", before: "gte", after: "lte",
    };
    const normalized = opposite[cmp];
    if (normalized) { cmp = normalized; negated = false; }
  }
  return { field: c.field, cmp, value, negated };
}).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))));

// Independent row oracle: evaluate the expected AST, never the production evaluator.
const oracleRowsFor = (schemaName: string, clauses: FilterClause[]): string[] => oracleRows[schemaName].filter((row) => clauses.every((c) => {
  const raw = row.fields[c.field];
  if (raw === undefined || raw === null || raw === "") return false;
  const value = c.value.normalize("NFC").toLocaleLowerCase();
  let hit: boolean;
  if (typeof raw === "number") {
    const n = Number(value);
    hit = c.cmp === "gt" ? raw > n : c.cmp === "gte" ? raw >= n : c.cmp === "lt" ? raw < n : c.cmp === "lte" ? raw <= n : c.cmp === "neq" ? raw !== n : raw === n;
  } else if (typeof raw === "boolean") {
    hit = c.cmp === "neq" ? raw !== (value === "true") : raw === (value === "true");
  } else {
    const s = raw.normalize("NFC").toLocaleLowerCase();
    hit = c.cmp === "contains" ? s.includes(value) : c.cmp === "before" || c.cmp === "lt" ? s < value :
      c.cmp === "after" || c.cmp === "gt" ? s > value : c.cmp === "gte" ? s >= value : c.cmp === "lte" ? s <= value :
      c.cmp === "neq" ? s !== value : s === value;
  }
  return c.negated ? !hit : hit;
})).map((r) => r.id);

const raw = (query: string, schema: SchemaDef, rules = false) => {
  const metas = featurizeQuery(query, schema, false);
  if (!metas.length) return { clauses: [], accepted: false, issues: ["No filter could be applied"] };
  let roles: number[];
  let boundaries: boolean[];
  if (rules) {
    // Same schema matches, token features, compiler and validator; only role labels come from rules.
    roles = metas.map((m, i) => {
      const w = m.token.norm;
      const prev = metas.slice(0, i).reverse().find((p) => p.matchedField);
      const kind = schema.fields.find((f) => f.name === prev?.matchedField)?.kind;
      const role = m.matchedField ? "FIELD" : m.matchedEnumField ? "VALUE" :
        LEX_AND_SET.has(w) ? "AND" : LEX_NEGATE_SET.has(w) ? "NEGATE" :
        LEX_NEQ_SET.has(w) ? "OP_NEQ" : LEX_GT_SET.has(w) ? "OP_GT" : LEX_LT_SET.has(w) ? "OP_LT" :
        LEX_BEFORE_SET.has(w) ? "OP_BEFORE" : LEX_AFTER_SET.has(w) ? "OP_AFTER" :
        LEX_CONTAINS_SET.has(w) ? "OP_CONTAINS" : LEX_EQ_SET.has(w) ? "OP_EQ" :
        LEX_BOOL_TRUE_SET.has(w) || LEX_BOOL_FALSE_SET.has(w) || /^-?\d+(?:\.\d+)?(?:ms|s)?$/.test(w) || /^\d{4}-\d{2}-\d{2}$/.test(w) || /^['"]/.test(m.token.text) ? "VALUE" :
        kind === "TEXT" && !LEX_FILLER_SET.has(w) ? "VALUE" : "NONE";
      return ROLES.indexOf(role as (typeof ROLES)[number]);
    });
    boundaries = metas.map(() => false);
  } else {
    const flat = new Int32Array(metas.flatMap((m) => m.featureIds));
    assert.equal(flat.length, metas.length * FEATURES_PER_TOKEN);
    const fwd = forwardTokensCPU(flat, metas.length, model.weights);
    roles = Array.from(fwd.roleIds);
    boundaries = Array.from(fwd.boundaryProbs, (p) => p >= 0.5);
  }
  const clauses = compileTokens(metas, roles, boundaries, schema);
  const issues = validateCompiledQuery(query, schema, metas, clauses);
  return { clauses, accepted: clauses.length > 0 && (!rules || issues.length === 0), issues };
};

const modes = ["raw", "guarded", "rules"] as const;
const results = [...cases, ...invalid].map((item) => {
  const schema = ALL_SCHEMAS[item.schemaName];
  const expectedRows = item.valid ? oracleRowsFor(item.schemaName, item.expected) : [];
  const byMode = Object.fromEntries(modes.map((mode) => {
    const result = mode === "guarded" ? model.parse(item.query, schema) : raw(item.query, schema, mode === "rules");
    const accepted = result.accepted && result.clauses.length > 0;
    const exact = accepted && canonical(result.clauses, schema) === canonical(item.expected, schema);
    const datasetRows: DatasetRow[] = oracleRows[item.schemaName].map((r) => ({ ...r, title: "", subtitle: "", badges: [], rightMeta: "", rightSub: "" }));
    const actualRows = accepted ? filterDatasetRows(datasetRows, result.clauses).map((r) => r.id) : [];
    return [mode, { accepted, clauses: result.clauses, issues: result.issues, exact, actualRows, correctRows: accepted && JSON.stringify(actualRows) === JSON.stringify(expectedRows) }];
  }));
  return { id: item.id, schema: item.schemaName, query: item.query, expected: item.expected, valid: item.valid, expectedRows, ...(!item.valid && { reason: item.reason }), modes: byMode };
});
const summary = Object.fromEntries(modes.map((mode) => {
  const valid = results.filter((r) => r.valid).map((r) => r.modes[mode] as any);
  const invalid = results.filter((r) => !r.valid).map((r) => r.modes[mode] as any);
  return [mode, { acceptedValid: valid.filter((r) => r.accepted).length, acceptedWrongClauses: valid.filter((r) => r.accepted && !r.exact).length,
    exactAcceptedAst: valid.filter((r) => r.exact).length, correctRowSets: valid.filter((r) => r.correctRows).length,
    rejectedValid: valid.filter((r) => !r.accepted).length, rejectedInvalid: invalid.filter((r) => !r.accepted).length,
    acceptedInvalid: invalid.filter((r) => r.accepted).length }];
}));
const report = { fixtureSha256, modelSha256, command: "npx tsx spike/evaluate_challenge.ts", node: process.version,
  hardware: `${process.platform} ${process.arch}; ${cpus()[0]?.model ?? "unknown CPU"}`, note: "Development recheck after a compiler fix informed by the initial frozen score. Agent-authored cases, not human-collected or a population accuracy estimate.",
  counts: { valid: cases.length, invalid: invalid.length, oracleRows: Object.values(oracleRows).flat().length }, summary, results };
writeFileSync("docs/experiment-results.json", JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ fixtureSha256, modelSha256, summary }, null, 2));
