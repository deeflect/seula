// Run: npx tsx src/seula/api.test.ts
import assert from "node:assert/strict";
import { DatabaseSync } from "node:sqlite";
import { columnFilteringFeature, constructTable, createFilteredRowModel, createSortedRowModel, rowSortingFeature, sortFns, tableFeatures } from "@tanstack/table-core";
import { storeReactivityBindings } from "@tanstack/table-core/store-reactivity-bindings";
import { benchTables, invoiceTable, withTypes } from "../live-tables.js";
import { filter, load, rowsOf, type Result } from "./node.js";
import { seulaFilterFn, toTanStack } from "./tanstack.js";
import { toSql } from "./sql.js";
import { parseNumber } from "./model.js";

let n = 0;
const eq = (a: unknown, b: unknown, msg = "") => (assert.deepEqual(a, b, msg), n++);
const today = new Date(2026, 10, 9);
const seula = await load();
const { model } = seula;
const features = tableFeatures({
  coreReactivityFeature: storeReactivityBindings(),
  columnFilteringFeature, filteredRowModel: createFilteredRowModel(),
  rowSortingFeature, sortedRowModel: createSortedRowModel(), sortFns,
});

const QUERIES: Record<string, string[]> = {
  invoices: ["overdue and balance at least 1k", "overdue or paid, top 3 by balance", "days late between 10 and 40", "overdue clients except fieldwork", "paid", "balance under 500", "sort by client"],
  eu: ["euro and ep seats more than 20", "not euro", "population over 50 million", "gini below 26"],
  race: ["points over 25 and laps 97", "team newman/haas racing", "grid less than 5"],
  episodes: ["directed by marcos siega", "written by julie plec", "aired after january 2010"],
  nfl: ["attendance over 50000", "weeks after 10", "games at memorial stadium", "games in october 1975"],
};

for (const t of [invoiceTable, ...benchTables]) {
  const ref = withTypes(t); // the object API infers types, so the reference does too
  const objs = t.rows.map((r) => Object.fromEntries(t.header.map((h, j) => [h, r[j]])));
  const api = seula.table(objs);
  for (const q of QUERIES[t.key]) {
    const r = api.parse(q, { today }), got = api.apply({ ...r, confirmAmbiguous: true }); // legacy parity exercises candidates, including uncertain ones
    const idx = (rows: typeof objs) => rows.map((o) => objs.indexOf(o));
    // 1. same rows, same order as the low-level API
    eq(idx(got), rowsOf(model, ref, { ...filter(model, q, ref, today), confirmAmbiguous: true }), `apply: ${q}`);
    const all = idx(api.apply({ ...r, limit: undefined, confirmAmbiguous: true }));

    // 2. TanStack table-core with seulaFilterFn on every column
    const s = toTanStack(r, { confirmAmbiguous: true });
    const tt = constructTable({
      features, data: objs, initialState: { columnFilters: s.columnFilters, sorting: s.sorting },
      columns: t.header.map((h) => ({ id: h, accessorFn: (o: (typeof objs)[number]) => o[h], filterFn: seulaFilterFn })),
    });
    eq(tt.getRowModel().rows.map((x) => x.index), all, `tanstack: ${q}`);

    // 3. SQL in node:sqlite, for queries without contains / date conditions (LIKE is substring, dates are text in SQL)
    const conds = r.conditions ?? [];
    if (!conds.some((c) => c.type === "date" || c.op.startsWith("contains"))) {
      const db = new DatabaseSync(":memory:"), qn = (h: string) => `"${h.replaceAll('"', '""')}"`;
      db.exec(`CREATE TABLE t (${t.header.map((h, j) => `${qn(h)} ${ref.types[j] === "real" ? "REAL" : "TEXT"}`).join(", ")})`);
      const ins = db.prepare(`INSERT INTO t VALUES (${t.header.map(() => "?").join(", ")})`);
      for (const row of api.apply({ conditions: null }).map((o) => t.header.map((h) => o[h]))) {
        ins.run(...row.map((c, j) => (ref.types[j] === "real" ? parseNumber(String(c)) : String(c))));
      }
      const sql = toSql(r, { confirmAmbiguous: true });
      let ids = db.prepare(`SELECT rowid FROM t WHERE ${sql.where} ${sql.orderBy ? "ORDER BY " + sql.orderBy : ""}`).all(...sql.params).map((x) => Number(x.rowid) - 1);
      if (sql.limit !== undefined) ids = ids.slice(0, sql.limit);
      eq(r.sort ? ids : [...ids].sort((a, b) => a - b), r.sort ? idx(got) : [...idx(got)].sort((a, b) => a - b), `sql: ${q} ${sql.where}`);
      db.close();
    }

    // 5. dropping one condition (not an alternative that shares its column) only widens the result
    const i = conds.findIndex((c, k) => !conds.some((d, m) => m !== k && d.column === c.column && /^(=|in|contains)/.test(c.op) && /^(=|in|contains)/.test(d.op)));
    if (i >= 0) {
      const wider = idx(api.apply({ ...r, conditions: conds.filter((_, k) => k !== i), limit: undefined, confirmAmbiguous: true }));
      assert(all.every((x) => wider.includes(x)), `superset: ${q}`), n++;
    }
  }
}

// 4. schema() from distinct values parses like table()
const inv = invoiceTable.rows.map((r) => Object.fromEntries(invoiceTable.header.map((h, j) => [h, r[j]])));
const cols = Object.fromEntries(invoiceTable.header.map((h, j) => [h, [...new Set(invoiceTable.rows.map((r) => r[j]))]]));
const q4 = "overdue and balance at least 1k";
const fromSchema: Result = seula.schema(cols).parse(q4, { today });
eq(fromSchema.conditions, seula.table(inv).parse(q4, { today }).conditions, "schema vs table");
assert(fromSchema.conditions?.length, "schema parsed something");

// 5. snake_case and camelCase keys parse like plain ones, and results name the caller's own keys
const hero = "overdue and balance over 1k, top 5 by balance";
for (const [balance, late] of [["balance_usd", "days_late"], ["balanceUsd", "daysLate"]]) {
  const rows = [{ client: "Acme", [balance]: 2400, [late]: 18, status: "overdue" }, { client: "Bolt", [balance]: 450, [late]: 3, status: "overdue" }, { client: "Cord", [balance]: 3200, [late]: 0, status: "due" }];
  const t = seula.table(rows), r = t.parse(hero);
  eq(r.conditions, [{ column: "status", type: "text", op: "=", value: "overdue" }, { column: balance, type: "number", op: ">", value: 1000 }], balance);
  eq([r.sort, r.limit], [{ column: balance, dir: "desc" }, 5], balance);
  eq(t.apply(r), [rows[0]], balance);
  eq(t.parse(`top 2 by days late`).sort, { column: late, dir: "desc" }, late);
}

console.log(`api: ${n} checks passed`);
