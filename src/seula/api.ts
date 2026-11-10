// Object API over filter(): load() -> { table(rows).parse(q) / .apply(result), schema(columns).parse(q) }.
// Conditions name columns and hold real values, so they can be shown, edited, and run anywhere (passes, tanstack.ts, sql.ts).

import { isIso, matchesDate } from "./extras.js";
import { filter, sortBy } from "./index.js";
import { colNumeric, parseNumber, tokenize, type Cond, type Seula, type Table } from "./model.js";

export type Condition = {
  column: string;
  type: "text" | "number" | "date";
  op: "=" | "!=" | ">" | ">=" | "<" | "<=" | "contains" | "in" | "not in" | "contains any";
  value: string | number | (string | number)[]; // array for in / not in / contains any
};
export type Result = { conditions: Condition[] | null; sort?: { column: string; dir: "asc" | "desc" }; limit?: number; confidence: number };
type Row = Record<string, unknown>;

const cellOf = (v: unknown): string | number =>
  typeof v === "number" ? v
  : v instanceof Date ? (isNaN(+v) ? "" : new Date(+v - v.getTimezoneOffset() * 6e4).toISOString().slice(0, 10)) // local calendar day
  : v == null ? "" : String(v);
const norm = (v: unknown) => tokenize(String(v)).join(" ");
const MULTI: Record<string, Condition["op"]> = { "=": "in", "!=": "not in", contains: "contains any" };
const ALT = new Set<string>(["=", "in", "contains", "contains any"]);

// On one column, =, in, contains and contains any are alternatives (any may match); every other condition must hold.
export function groups(conditions: Condition[]): { all: Condition[]; any: Condition[][] } {
  const all: Condition[] = [], any = new Map<string, Condition[]>();
  for (const c of conditions) ALT.has(c.op) ? (any.get(c.column) ?? any.set(c.column, []).get(c.column)!).push(c) : all.push(c);
  return { all, any: [...any.values()] };
}

// One condition against one cell, mirroring matches() in model.ts.
function test(cell: unknown, c: Condition): boolean {
  const v = cellOf(cell), vals = Array.isArray(c.value) ? c.value : [c.value];
  if (c.type === "date") return matchesDate(v, c.op, String(c.value));
  if (c.type === "number") {
    const a = parseNumber(String(v)), b = Number(vals[0]);
    if (a === null) return false;
    if (c.op === "in" || c.op === "not in") return vals.map(Number).includes(a) === (c.op === "in");
    return ({ "=": a === b, "!=": a !== b, ">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b } as Record<string, boolean>)[c.op] ?? false;
  }
  const x = norm(v), ys = vals.map(norm);
  if (c.op === "=" || c.op === "in") return ys.includes(x);
  if (c.op === "!=" || c.op === "not in") return !ys.includes(x);
  return ys.some((y) => !!y && ` ${x} `.includes(` ${y} `));
}

export function passes(get: (column: string) => unknown, conditions: Condition[]): boolean {
  const { all, any } = groups(conditions);
  return all.every((c) => test(get(c.column), c)) && any.every((g) => g.some((c) => test(get(g[0].column), c)));
}

function conditionsOf(model: Seula, table: Table, conds: Cond[]): Condition[] {
  const numeric = model.index(table).rowNumeric, out: Condition[] = [], merged = new Map<string, [Condition, Set<string | number>]>();
  for (const [j, op, v] of conds) {
    const column = table.header[j], n = parseNumber(v), o = op as Condition["op"];
    if (isIso(v) && op !== "contains") { out.push({ column, type: "date", op: o, value: v }); continue; }
    if ([">", ">=", "<", "<="].includes(op)) { out.push({ column, type: "number", op: o, value: n ?? v }); continue; }
    const type = (op === "=" || op === "!=") && numeric[j] && n !== null ? "number" : "text";
    // text = / != : back to the cells as written ("Overdue", "Newman/Haas Racing"), so they work outside Seula too
    const orig = type === "text" && op !== "contains" ? [...new Set(table.rows.map((r) => String(r[j])))].filter((c) => norm(c) === norm(v)) : [];
    const key = `${j}\0${op}\0${type}`;
    if (!merged.has(key)) { const c: Condition = { column, type, op: o, value: [] }; out.push(c); merged.set(key, [c, new Set()]); }
    const vals = merged.get(key)![1];
    for (const x of type === "number" ? [n!] : orig.length ? orig : [v]) vals.add(x);
  }
  for (const [c, vals] of merged.values()) {
    const a = [...vals];
    if (a.length > 1) [c.op, c.value] = [MULTI[c.op], a];
    else c.value = a[0];
  }
  return out;
}

function parser(model: Seula, table: Table) {
  return (query: string, opts?: { today?: Date }): Result => {
    const f = filter(model, query, table, opts?.today);
    return {
      conditions: f.conds && conditionsOf(model, table, f.conds),
      sort: f.sort && { column: table.header[f.sort[0]], dir: f.sort[1] },
      limit: f.limit,
      confidence: f.conf,
    };
  };
}

const typed = (header: string[], rows: Table["rows"], valuesOnly?: boolean): Table => ({
  header, rows, valuesOnly, types: header.map((_, j) => (colNumeric(rows, j) ? "real" : "text")),
});

export function wrap(model: Seula) {
  return {
    model,
    // Rows as objects; the header is the keys of the first row.
    table<T extends Row>(rows: T[]) {
      const header = Object.keys(rows[0] ?? {});
      const t = typed(header, rows.map((r) => header.map((h) => cellOf(r[h]))));
      return {
        parse: parser(model, t),
        apply(result: Pick<Result, "conditions" | "sort" | "limit">): T[] {
          const hits = rows.filter((r) => !result.conditions || passes((c) => r[c], result.conditions));
          if (result.sort) sortBy(hits, (r) => cellOf(r[result.sort!.column]), result.sort.dir);
          return result.limit === undefined ? hits : hits.slice(0, result.limit);
        },
      };
    },
    // Server-paged tables: sample distinct values per column (numbers: include min and max), no real rows.
    schema(columns: Record<string, unknown[]>) {
      const header = Object.keys(columns), n = Math.max(0, ...header.map((h) => columns[h].length));
      const rows = Array.from({ length: n }, (_, i) => header.map((h) => (columns[h].length ? cellOf(columns[h][i % columns[h].length]) : "")));
      return { parse: parser(model, typed(header, rows, true)) };
    },
  };
}
