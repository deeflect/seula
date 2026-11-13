import assert from "node:assert/strict";
import { DatabaseSync } from "node:sqlite";
import { columnFilteringFeature, constructTable, createFilteredRowModel, tableFeatures } from "@tanstack/table-core";
import { storeReactivityBindings } from "@tanstack/table-core/store-reactivity-bindings";
import { load } from "./node.js";
import { toSql } from "./sql.js";
import { seulaFilterFn, toTanStack } from "./tanstack.js";

const s = await load();
const rows = [
  { id: "a", client: "Acme", status: "overdue", balance: 2400, daysLate: 18 },
  { id: "b", client: "Bolt", status: "paid", balance: 0, daysLate: 0 },
  { id: "c", client: "Cord", status: "pending", balance: 450, daysLate: 0 },
  { id: "d", client: "Dune", status: "overdue", balance: 800, daysLate: 45 },
];
const concepts = { "at risk": [{ column: "daysLate", type: "number" as const, op: ">" as const, value: 10 }] };
const t = s.table(rows, { concepts });
const ids = (q: string) => t.apply(t.parse(q)).map((r) => r.id);
assert.deepEqual(ids("balance over 1000000"), []);
assert.deepEqual(t.parse("balance under 0").conditions, [{ column: "balance", type: "number", op: "<", value: 0 }]);
assert.ok(t.parse("balance over 1000000").confidence > 0);
assert.deepEqual(ids("status cancelled"), []);
assert.deepEqual(ids("overdue or balance under 500"), ["a", "b", "c", "d"]);
assert.deepEqual(ids("overdue and (balance over 1000 or days late over 30)"), ["a", "d"]);
assert.deepEqual(ids("overdue but not Acme"), ["d"]);
assert.deepEqual(ids("at risk except Acme"), ["d"]);
assert.deepEqual(ids("not at risk"), ["b", "c"]);
for (const q of ["at risk", "not at risk", "at risk or paid", "at risk except Acme"]) assert.ok(Number.isFinite(t.parse(q).confidence) && t.parse(q).confidence >= 0 && t.parse(q).confidence <= 1, q);
assert.deepEqual(ids("at risk or paid"), ["a", "b", "d"]);
assert.deepEqual(ids("paid or at risk"), ["a", "b", "d"]);
assert.deepEqual(ids("at risk and paid"), []);
assert.equal(t.parse("overdue and nonsense").status, "ambiguous");
assert.deepEqual(ids("overdue and paid"), []);
assert.deepEqual(s.schema(Object.fromEntries(Object.keys(rows[0]).map((k) => [k, [...new Set(rows.map((r) => r[k as keyof typeof r]))]])), { concepts }).parse("at risk").status, "ok");
assert.throws(() => s.table(rows, { concepts: { bad: [{ column: "missing", type: "text", op: "=", value: "x" }] } }));
for (const q of ["overdue and balance over", "balance over", "overdue or", "(overdue", "overdue and (balance over)"]) {
  const r = t.parse(q); assert.equal(r.status, "incomplete", q); assert.throws(() => t.apply(r), q);
}
for (const q of ["balance >", "balance >=", "overdue and balance >"]) assert.equal(t.parse(q).status, "incomplete", q);
for (const q of ["average balance", "group by status", "join customers"]) { const r = t.parse(q); assert.equal(r.status, "unsupported"); assert.throws(() => toSql(r)); }
for (const q of ["high risk clients", "clients who owe money"]) { const r = t.parse(q); assert.equal(r.status, "ambiguous", q); assert.throws(() => t.apply(r)); }
assert.deepEqual(s.table(rows.filter((r) => r.status !== "pending")).apply(s.table(rows.filter((r) => r.status !== "pending")).parse("overdue or pending")).map((r) => r.id), ["a", "d"]);
assert.equal(t.aggregate(t.parse("how many overdue invoices")), 2);
assert.equal(t.aggregate(t.parse("count overdue")), 2);
assert.equal(t.aggregate(t.parse("total overdue balance")), 3200);
assert.equal(t.aggregate(t.parse("sum balance for overdue")), 3200);
assert.equal(t.aggregate(t.parse("total balance over 1000000")), 0);
assert.equal(t.parse("sum status for overdue").status, "unsupported");
assert.deepEqual(t.parse("total balance over 1000").conditions, [{ column: "balance", type: "number", op: ">", value: 1000 }]);
assert.equal(t.aggregate(t.parse("total balance over 1000")), 2400);
assert.throws(() => t.aggregate({ ...t.parse("total overdue balance"), aggregate: { op: "sum", column: "status" } }));
assert.equal(t.parse("count overdue top 2 by balance").status, "unsupported");
assert.equal(t.parse("(".repeat(40) + "overdue" + ")".repeat(40)).status, "unsupported");
assert.equal(t.parse("x".repeat(5000)).status, "unsupported");
assert.throws(() => t.apply({ conditions: [{ column: "missing", type: "text", op: "=", value: "x" }] }));
assert.throws(() => toSql({ conditions: [{ column: "balance", type: "number", op: ">", value: Infinity }] }));
assert.throws(() => s.table(rows, { concepts: { bad: [{ column: "client", type: "text", op: ">", value: "Bob" }] } }));

const special = s.table([{ id: "x", name: "Research and Development" }, { id: "y", name: "Sales" }, { id: "z", name: "Average Joe" }]);
assert.deepEqual(special.apply(special.parse("Research and Development")).map((r) => r.id), ["x"]);
assert.deepEqual(special.apply(special.parse('name "Research and Development"')).map((r) => r.id), ["x"]);
assert.deepEqual(special.apply(special.parse("Average Joe")).map((r) => r.id), ["z"]);
const apostrophe = s.table([{ id: "a", name: "O'Brien", status: "open" }, { id: "b", name: "Bolt", status: "paid" }]);
assert.deepEqual(apostrophe.apply(apostrophe.parse("name O'Brien or paid")).map((r) => r.id), ["a", "b"]);
assert.equal(apostrophe.parse("name 'O'Brien").status, "incomplete");

const q = t.parse("overdue and (balance over 1000 or days late over 30)");
const sql = toSql(q), db = new DatabaseSync(":memory:");
db.exec('CREATE TABLE t (id TEXT, client TEXT, status TEXT, balance REAL, daysLate REAL)');
const insert = db.prepare("INSERT INTO t VALUES (?, ?, ?, ?, ?)");
for (const r of rows) insert.run(r.id, r.client, r.status, r.balance, r.daysLate);
assert.deepEqual(db.prepare(`SELECT id FROM t WHERE ${sql.where}`).all(...sql.params).map((r) => r.id), ["a", "d"]);
const sum = t.parse("total overdue balance"), sumSql = toSql(sum);
assert.equal(db.prepare(`SELECT ${sumSql.select} AS n FROM t WHERE ${sumSql.where}`).get(...sumSql.params)?.n, 3200);
db.close();

const state = toTanStack(q);
const table = constructTable({
  features: tableFeatures({ coreReactivityFeature: storeReactivityBindings(), columnFilteringFeature, filteredRowModel: createFilteredRowModel() }),
  data: rows, initialState: { columnFilters: state.columnFilters },
  columns: Object.keys(rows[0]).map((id) => ({ id, accessorFn: (row: (typeof rows)[number]) => row[id as keyof typeof row], filterFn: seulaFilterFn })),
});
assert.deepEqual(table.getRowModel().rows.map((r) => r.original.id), ["a", "d"]);
assert.deepEqual(toTanStack(sum).aggregate, { op: "sum", column: "balance" });

const nullable = [{ id: "a", status: "paid" }, { id: "b", status: null }, { id: "c", status: "open" }];
const nt = s.table(nullable), nr = nt.parse("not (paid or open)");
assert.deepEqual(nt.apply(nr).map((r) => r.id), ["b"]);
const ns = toSql(nr), nd = new DatabaseSync(":memory:");
nd.exec("CREATE TABLE t (id TEXT, status TEXT)");
const ni = nd.prepare("INSERT INTO t VALUES (?, ?)");
for (const row of nullable) ni.run(row.id, row.status);
assert.deepEqual(nd.prepare(`SELECT id FROM t WHERE ${ns.where}`).all(...ns.params).map((r) => r.id), ["b"]);
nd.close();

const games = s.table([{ Title: "A", Developer: "Ubisoft Montreal", Publisher: "Ubisoft" }, { Title: "B", Developer: "Other", Publisher: "Ubisoft" }]);
assert.deepEqual(games.apply(games.parse("publisher is Ubisoft and the developer name has ubisoft in it")).map((r) => r.Title), ["A"]);
const collisions = s.table([{ id: "a", count: 8, total: 1, name: "Count Dracula" }, { id: "b", count: 3, total: 7, name: "Total Recall" }]);
assert.equal(collisions.parse("count > 5").aggregate, undefined);
assert.deepEqual(collisions.apply(collisions.parse("count > 5")).map((r) => r.id), ["a"]);
assert.deepEqual(collisions.apply(collisions.parse("Count Dracula")).map((r) => r.id), ["a"]);
assert.deepEqual(collisions.apply(collisions.parse("Total Recall")).map((r) => r.id), ["b"]);

const semanticCases: { rows: Record<string, string>[]; query: string; ids: number[] }[] = [
  { rows: [{ incumbent: "John Shimkus" }, { incumbent: "Jane Doe" }], query: "incumbent isnt john shimkus", ids: [1] },
  { rows: [{ Competition: "World Cup" }, { Competition: "Regional" }], query: "competition world", ids: [0] },
  { rows: [{ Decision: "LaBarbera" }, { Decision: "Smith" }], query: "decision labarbera pls", ids: [0] },
  { rows: [{ Player: "Bótin" }, { Player: "Dean Caliguire" }, { Player: "Other" }], query: "player botin", ids: [0] },
  { rows: [{ Player: "Bótin" }, { Player: "Dean Caliguire" }, { Player: "Other" }], query: "player isnt dean caliguire", ids: [0, 2] },
  { rows: [{ Venue: "Stadion na Koniku", Score: "0:1" }, { Venue: "Other", Score: "0:1" }, { Venue: "Other", Score: "2:0" }], query: "not stadion na koniku score 0:1", ids: [1] },
  { rows: [{ Writer: "Balcer", Title: "Denial", Director: "Jace" }, { Writer: "Balcer", Title: "Other", Director: "Jace" }, { Writer: "Balcer", Title: "Other", Director: "Jill" }], query: `show me episodes written by balcer that aren't titled "Denial" and directed by jace`, ids: [1] },
];
for (const { rows: sample, query, ids: expected } of semanticCases) {
  const table = s.table(sample), parsed = table.parse(query);
  assert.equal(parsed.status, "ok", query);
  assert.deepEqual(table.apply(parsed).map((row) => sample.indexOf(row)), expected, query);
}
console.log("seula upgrade: checks passed");
