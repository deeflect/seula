import assert from "node:assert/strict";
import { ALL_SCHEMAS, SieveModel, parseWithOperatorAliases } from "../src/engine";
import type { SchemaDef } from "../src/engine";

const model = new SieveModel();
const schema = ALL_SCHEMAS.client_work;
const aliases = {
  topping: "gt", beneath: "lt", equaling: "eq", "at least": "gte", "no fewer than": "gte",
  "at most": "lte", "no more than": "lte", "greater than": "gt", "less than": "lt",
} as const;
const parse = (query: string) => parseWithOperatorAliases(model, query, schema, aliases);
const clause = (cmp: string, value = "4") => [{ field: "revisions", cmp, value, negated: false }];
for (const [query, cmp] of [
  ["revisions topping 4", "gt"], ["revisions beneath 4", "lt"], ["revisions equaling 4", "eq"],
  ["revisions at least 4", "gte"], ["revisions no fewer than 4", "gte"],
  ["revisions at most 4", "lte"], ["revisions no more than 4", "lte"],
  ["revisions greater than 4", "gt"], ["revisions less than 4", "lt"],
  ["4 topping revisions", "gt"], ["4 beneath revisions", "lt"],
  ["TOPPING 4 REVISIONS", "gt"], ["No MoRe ThAn 4 revisions", "lte"],
] as const) {
  const adapted = parse(query);
  assert.equal(adapted.result.accepted, true, `${query}: ${adapted.result.issues.join("; ")}`);
  assert.deepEqual(adapted.result.clauses, clause(cmp), query);
  assert.equal(adapted.originalQuery, query);
  assert.equal(adapted.result.query, adapted.normalizedQuery);
  assert.equal(adapted.rewrites.length, 1);
  const { start, end, phrase, replacement } = adapted.rewrites[0];
  assert.equal(query.slice(start, end), phrase);
  assert.equal(adapted.normalizedQuery, query.slice(0, start) + replacement + query.slice(end));
}

const mixed = parse("Revisions   NO   FEWER THAN  4 and revisions beneath 7");
assert.equal(mixed.normalizedQuery, "Revisions   >=  4 and revisions < 7");
assert.deepEqual(mixed.result.clauses, [...clause("gte"), ...clause("lt", "7")]);
assert.equal(mixed.rewrites.length, 2);
assert.equal(parse("revisions topping 4 and invoice_state paid").result.accepted, true);

for (const query of [
  'founder = "topping"', "revisions banana 4", "revisions topping x", "revisions topping 4 or revisions beneath 2",
  "revisions topping (4)", "revisions topping 4kb", "revisions topping 4 and revisions banana 2",
  'topping "4" revisions', "topping 4 invoice_state", "topping (4) revisions", "topping, 4 revisions",
]) {
  const adapted = parse(query);
  if (query === 'founder = "topping"') {
    assert.equal(adapted.normalizedQuery, query);
    assert.equal(adapted.result.accepted, true);
  } else assert.equal(adapted.result.accepted, false, query);
}
const long = `${Array(21).fill("please").join(" ")} revisions no fewer than 4`;
assert.equal(parse(long).normalizedQuery, long);
assert.equal(parse(long).result.accepted, false);
assert.equal(parse("revisions topping 4").rewrites[0].phrase, "topping");

const invoices: SchemaDef = {
  name: "invoices", description: "Invoice finder", split: "transfer", fields: [
    { name: "status", kind: "ENUM", aliases: [], enumValues: ["overdue", "due", "paid"] },
    { name: "balance_usd", kind: "NUMBER", aliases: ["balance"], enumValues: [] },
  ],
};
const invoice = parseWithOperatorAliases(model, "overdue and balance at least $1k", invoices, aliases);
assert.equal(invoice.normalizedQuery, "overdue and balance >= $1k");
assert.equal(invoice.result.accepted, true, invoice.result.issues.join("; "));
assert.deepEqual(invoice.result.clauses, [
  { field: "status", cmp: "eq", value: "overdue", negated: false },
  { field: "balance_usd", cmp: "gte", value: "1000", negated: false },
]);

for (const bad of [
  { over: "lt" }, { "at least": "lt" }, { "not topping": "gt" }, { "or topping": "gt" },
  { "topping!": "gt" }, { "many many many many many": "gt" }, { topping: "contains" },
  { "over banana": "lt" }, { before: "gt" },
  { Topping: "lt", topping: "gt" },
]) assert.throws(() => parseWithOperatorAliases(model, "revisions topping 4", schema, bad as never));

const input = "revisions topping 4";
const before = JSON.stringify({ schema, aliases });
parse(input);
assert.equal(input, "revisions topping 4");
assert.equal(JSON.stringify({ schema, aliases }), before);
console.log("vocabulary adapter: passed");
