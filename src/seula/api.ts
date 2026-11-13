// Object API over filter(): load() -> { table(rows).parse(q) / .apply(result), schema(columns).parse(q) }.
// Conditions name columns and hold real values, so they can be shown, edited, and run anywhere (passes, tanstack.ts, sql.ts).

import { isIso, matchesDate } from "./extras.js";
import { assertExecutable, filter, hasExactCell, sortBy, type Expression, type Status } from "./index.js";
import { colNumeric, parseNumber, tokenize, type Cond, type Seula, type Table } from "./model.js";

export type Condition = {
  column: string;
  type: "text" | "number" | "date";
  op: "=" | "!=" | ">" | ">=" | "<" | "<=" | "contains" | "in" | "not in" | "contains any";
  value: string | number | (string | number)[]; // array for in / not in / contains any
};
export type Result = { conditions: Condition[] | null; expression?: Expression<Condition>; sort?: { column: string; dir: "asc" | "desc" }; limit?: number; confidence: number; status: Status; message?: string; aggregate?: { op: "count" | "sum"; column?: string } };
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

export function passesExpression(get: (column: string) => unknown, expression: Expression<Condition>): boolean {
  if ("and" in expression) return expression.and.every((x) => passesExpression(get, x));
  if ("or" in expression) return expression.or.some((x) => passesExpression(get, x));
  if ("not" in expression) return !passesExpression(get, expression.not);
  return test(get(expression.column), expression);
}

export function validateExpression(expression: Expression<Condition>, names?: string[], depth = 0): void {
  if (depth > 16) throw new Error("seula: expression is too deep");
  if ("and" in expression || "or" in expression) {
    const children = "and" in expression ? expression.and : expression.or;
    if (!Array.isArray(children) || !children.length || children.length > 64) throw new Error("seula: invalid expression group");
    children.forEach((x) => validateExpression(x, names, depth + 1)); return;
  }
  if ("not" in expression) { validateExpression(expression.not, names, depth + 1); return; }
  if (typeof expression.column !== "string" || !expression.column || (names && !names.includes(expression.column)) || !["text", "number", "date"].includes(expression.type) || !["=", "!=", ">", ">=", "<", "<=", "contains", "in", "not in", "contains any"].includes(expression.op)) throw new Error("seula: invalid condition");
  const values = Array.isArray(expression.value) ? expression.value : [expression.value];
  if (!values.length || values.length > 64 || values.some((x) => (typeof x !== "string" && typeof x !== "number") || (typeof x === "number" && !Number.isFinite(x)))) throw new Error("seula: invalid condition value");
  if (expression.type === "text" && !["=", "!=", "in", "not in", "contains", "contains any"].includes(expression.op)) throw new Error("seula: invalid text operator");
  if (expression.type === "number" && !["=", "!=", "in", "not in", ">", ">=", "<", "<="].includes(expression.op)) throw new Error("seula: invalid numeric operator");
  if (expression.type === "date" && !["=", "!=", ">", ">=", "<", "<="].includes(expression.op)) throw new Error("seula: invalid date operator");
  if (expression.type === "number" && values.some((x) => !Number.isFinite(Number(x)))) throw new Error("seula: invalid numeric value");
  if (expression.type === "date" && values.some((x) => typeof x !== "string" || !isIso(x) || !Number.isFinite(Date.parse(`${x}T00:00:00Z`)) || new Date(`${x}T00:00:00Z`).toISOString().slice(0, 10) !== x)) throw new Error("seula: invalid date value");
}

function conditionsOf(model: Seula, table: Table, names: string[], conds: Cond[]): Condition[] {
  const numeric = model.index(table).rowNumeric, out: Condition[] = [], merged = new Map<string, [Condition, Set<string | number>]>();
  for (const [j, op, v] of conds) {
    const column = names[j], n = parseNumber(v), o = op as Condition["op"];
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

function expressionOf(model: Seula, table: Table, names: string[], e: Expression<Cond>): Expression<Condition> {
  if (Array.isArray(e)) return conditionsOf(model, table, names, [e as Cond])[0];
  if ("and" in e) return { and: e.and.map((x) => expressionOf(model, table, names, x)) };
  if ("or" in e) return { or: e.or.map((x) => expressionOf(model, table, names, x)) };
  return { not: expressionOf(model, table, names, e.not) };
}

function flatAnd(e: Expression<Condition>): Condition[] | null {
  if ("and" in e) {
    const parts = e.and.map(flatAnd);
    if (parts.some((x) => x === null)) return null;
    const flat = parts.flat() as Condition[];
    const equals = flat.filter((c) => c.op === "=" || c.op === "in" || c.op === "contains" || c.op === "contains any");
    return new Set(equals.map((c) => c.column)).size === equals.length ? flat : null;
  }
  return "or" in e || "not" in e ? null : [e];
}

// names: the caller's own column keys, returned in results; table.header is what the model reads.
function parser(model: Seula, table: Table, names: string[], concepts: Record<string, Condition[] | Expression<Condition>> = {}) {
  const toCond = (c: Condition): Expression<Cond> => {
    const j = names.indexOf(c.column), values = Array.isArray(c.value) ? c.value : [c.value];
    const op = c.op === "in" || c.op === "not in" ? "=" : c.op === "contains any" ? "contains" : c.op;
    const leaves = values.map((v) => [j, op, String(v)] as Cond);
    const e: Expression<Cond> = leaves.length === 1 ? leaves[0] : { or: leaves };
    return c.op === "not in" ? { not: e } : e;
  };
  const conceptOf = (e: Condition[] | Expression<Condition>): Expression<Cond> => {
    if (Array.isArray(e)) return { and: e.map(conceptOf) };
    if ("and" in e) return { and: e.and.map(conceptOf) };
    if ("or" in e) return { or: e.or.map(conceptOf) };
    if ("not" in e) return { not: conceptOf(e.not) };
    return toCond(e);
  };
  const resolved = Object.fromEntries(Object.entries(concepts).map(([key, e]) => [key, conceptOf(e)]));
  return (query: string, opts?: { today?: Date }): Result => {
    let aggregate: Result["aggregate"];
    let work = query.trim();
    const knownCell = hasExactCell(table, work);
    if (!knownCell && /^(?:how many|count)\s+/i.test(work) && !names.some((n) => n.toLowerCase() === work.split(/\s+/)[0].toLowerCase())) { aggregate = { op: "count" }; work = work.replace(/^(?:how many|count)\s+/i, ""); }
    else if (!knownCell && /^(?:total|sum)\s+/i.test(work) && !names.some((n) => n.toLowerCase() === work.split(/\s+/)[0].toLowerCase())) {
      const after = work.replace(/^(?:total|sum)\s+/i, "");
      const col = names.find((n) => new RegExp(`\\b${n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(after));
      if (!col || table.types[names.indexOf(col)] !== "real") return { conditions: null, confidence: 0, status: "unsupported", message: "sum needs a numeric column" };
      aggregate = { op: "sum", column: col };
      if (after.toLowerCase().startsWith(col.toLowerCase())) {
        const tail = after.slice(col.length).trim();
        work = /^(?:over|under|above|below|greater|less|between|at least|at most|>|<|=)/i.test(tail) ? after : tail.replace(/^for\s+/i, "");
      } else work = after.replace(new RegExp(`\\b${col}\\b`, "i"), "").trim();
    }
    const f = filter(model, work, table, opts?.today, (phrase) => resolved[phrase]);
    let expression = f.expression && expressionOf(model, table, names, f.expression);
    const flattened = expression ? flatAnd(expression) : null;
    if (flattened) expression = undefined;
    let status = f.status, message = f.message;
    if (aggregate && !work) { status = "ok"; message = undefined; }
    if (aggregate && (f.sort || f.limit)) { status = "unsupported"; message = "aggregation with sort or limit is unsupported"; }
    return {
      conditions: expression ? null : flattened ?? (f.conds && conditionsOf(model, table, names, f.conds)),
      ...(expression && { expression }),
      sort: f.sort && { column: names[f.sort[0]], dir: f.sort[1] },
      limit: f.limit,
      confidence: f.conf,
      status, ...(message && { message }), ...(aggregate && { aggregate }),
    };
  };
}

// Keys reach the model as words: "balanceUsd" reads as "balance Usd" (the tokenizer already splits on _ and -).
const typed = (keys: string[], rows: Table["rows"], valuesOnly?: boolean): Table => ({
  header: keys.map((k) => k.replace(/([\p{Ll}\p{N}])(\p{Lu})/gu, "$1 $2")), rows, valuesOnly, types: keys.map((_, j) => (colNumeric(rows, j) ? "real" : "text")),
});

export function wrap(model: Seula) {
  return {
    model,
    // Rows as objects; the header is the keys of the first row.
    table<T extends Row>(rows: T[], opts: { concepts?: Record<string, Condition[] | Expression<Condition>> } = {}) {
      const header = Object.keys(rows[0] ?? {});
      const t = typed(header, rows.map((r) => header.map((h) => cellOf(r[h]))));
      const concepts = Object.fromEntries(Object.entries(opts.concepts ?? {}).map(([key, e]) => [key.toLowerCase().replace(/\s+/g, " ").trim(), e]));
      if (Object.keys(concepts).length !== Object.keys(opts.concepts ?? {}).length) throw new Error("seula: duplicate concept name");
      for (const [key, e] of Object.entries(concepts)) { if (!key) throw new Error("seula: invalid concept name"); validateExpression(Array.isArray(e) ? { and: e } : e, header); }
      const matched = (result: Pick<Result, "conditions" | "expression">) => rows.filter((r) => result.expression ? passesExpression((c) => r[c], result.expression) : !result.conditions || passes((c) => r[c], result.conditions));
      return {
        parse: parser(model, t, header, concepts),
        apply(result: Pick<Result, "conditions" | "sort" | "limit"> & Partial<Result> & { confirmAmbiguous?: boolean }): T[] {
          assertExecutable(result, result.confirmAmbiguous);
          if (result.expression) validateExpression(result.expression, header);
          for (const c of result.conditions ?? []) validateExpression(c, header);
          if (result.sort && (!header.includes(result.sort.column) || !["asc", "desc"].includes(result.sort.dir))) throw new Error("seula: invalid sort");
          if (result.limit !== undefined && (!Number.isSafeInteger(result.limit) || result.limit < 0)) throw new Error("seula: invalid limit");
          const hits = matched(result);
          if (result.sort) sortBy(hits, (r) => cellOf(r[result.sort!.column]), result.sort.dir);
          return result.limit === undefined ? hits : hits.slice(0, result.limit);
        },
        aggregate(result: Result & { confirmAmbiguous?: boolean }): number {
          assertExecutable(result, result.confirmAmbiguous);
          if (!result.aggregate) throw new Error("seula: no aggregate");
          if (result.sort || result.limit) throw new Error("seula: aggregate cannot use sort or limit");
          if (result.expression) validateExpression(result.expression, header);
          for (const c of result.conditions ?? []) validateExpression(c, header);
          const hits = matched(result);
          if (result.aggregate.op === "count") return hits.length;
          if (result.aggregate.op !== "sum" || !result.aggregate.column || !header.includes(result.aggregate.column) || t.types[header.indexOf(result.aggregate.column)] !== "real") throw new Error("seula: invalid aggregate");
          return hits.reduce((sum, row) => sum + (parseNumber(String(row[result.aggregate!.column!])) ?? 0), 0);
        },
      };
    },
    // Server-paged tables: sample distinct values per column (numbers: include min and max), no real rows.
    schema(columns: Record<string, unknown[]>, opts: { concepts?: Record<string, Condition[] | Expression<Condition>> } = {}) {
      const header = Object.keys(columns), n = Math.max(0, ...header.map((h) => columns[h].length));
      const rows = Array.from({ length: n }, (_, i) => header.map((h) => (columns[h].length ? cellOf(columns[h][i % columns[h].length]) : "")));
      const concepts = Object.fromEntries(Object.entries(opts.concepts ?? {}).map(([key, e]) => [key.toLowerCase().replace(/\s+/g, " ").trim(), e]));
      if (Object.keys(concepts).length !== Object.keys(opts.concepts ?? {}).length) throw new Error("seula: duplicate concept name");
      for (const [key, e] of Object.entries(concepts)) { if (!key) throw new Error("seula: invalid concept name"); validateExpression(Array.isArray(e) ? { and: e } : e, header); }
      return { parse: parser(model, typed(header, rows, true), header, concepts) };
    },
  };
}
