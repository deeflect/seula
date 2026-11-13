// Public entry: the rule pre-pass (dates, yes/no columns, sort, top-N) plus the model on what is left.
// Usage: const f = filter(model, "overdue last 30 days top 5 by balance", table); rowsOf(model, table, f)

import { dateOf, extras, type Extras } from "./extras.js";
import { parseNumber, runConds, Seula, tokenize, type Cond, type Table } from "./model.js";
import { wrap } from "./api.js";

export { Seula, type Cond, type Table } from "./model.js";
export * from "./api.js";
export type Status = "ok" | "incomplete" | "unsupported" | "ambiguous";
export type Expression<T> = T | { and: Expression<T>[] } | { or: Expression<T>[] } | { not: Expression<T> };
export type Filter = { conds: Cond[] | null; expression?: Expression<Cond>; conf: number; status: Status; message?: string; sort?: Extras["sort"]; limit?: number; aggregate?: { op: "count" | "sum"; column?: number } };

export function assertExecutable(result: { status?: Status; message?: string }, confirmAmbiguous = false) {
  if (result.status && result.status !== "ok" && !(result.status === "ambiguous" && confirmAmbiguous))
    throw new Error(`seula: ${result.status}: ${result.message ?? "query cannot be executed"}`);
}

const vocabCache = new WeakMap<Table, { exact: Map<string, Set<number>>; byColumn: Set<string>[]; protected: string[]; operators: string[] }>();
function vocab(table: Table) {
  let v = vocabCache.get(table);
  if (v) return v;
  v = { exact: new Map(), byColumn: table.header.map(() => new Set<string>()), protected: [], operators: [] };
  for (const row of table.rows) row.forEach((cell, j) => {
    const raw = String(cell), key = tokenize(raw).join(" ");
    if (key) { (v!.exact.get(key) ?? v!.exact.set(key, new Set()).get(key)!).add(j); v!.byColumn[j].add(key); }
    if (/\b(?:and|or)\b|[()]/i.test(raw) && raw.length > 4) v!.protected.push(raw);
    if (/\b(?:avg|average|join|group by)\b/i.test(raw)) v!.operators.push(raw);
  });
  vocabCache.set(table, v);
  return v;
}
export const hasExactCell = (table: Table, query: string) => vocab(table).exact.has(tokenize(query).join(" "));

function parts(query: string, table: Table): string[] {
  const out: string[] = [];
  let start = 0, quote = "", depth = 0;
  const protectedRanges = vocab(table).protected.flatMap((v) => {
    const at = query.toLowerCase().indexOf(v.toLowerCase());
    return at >= 0 && (at === 0 || !/\w/.test(query[at - 1])) && (at + v.length === query.length || !/\w/.test(query[at + v.length])) ? [[at, at + v.length]] : [];
  });
  for (let i = 0; i < query.length; i++) {
    const c = query[i];
    if (quote) { if (c === quote && !(c === "'" && i > 0 && /[\p{L}\p{N}]/u.test(query[i - 1]) && /[\p{L}\p{N}]/u.test(query[i + 1] ?? ""))) quote = ""; continue; }
    if ((c === "'" || c === '"') && (c !== "'" || i === 0 || !/[\p{L}\p{N}]/u.test(query[i - 1]))) { quote = c; continue; }
    if (protectedRanges.some(([a, z]) => i >= a && i < z)) continue;
    if (c === "(" || c === ")") { out.push(query.slice(start, i)); out.push(c); start = i + 1; depth += c === "(" ? 1 : -1; continue; }
    const m = /^(and|or|but|except)\b/i.exec(query.slice(i));
    if (m && (i === 0 || /\s/.test(query[i - 1])) && /\s/.test(query[i + m[0].length] ?? "")
      && !(m[0].toLowerCase() === "and" && /\bbetween\s+\S+\s*$/i.test(query.slice(start, i)))) {
      out.push(query.slice(start, i)); out.push(m[0].toLowerCase()); i += m[0].length - 1; start = i + 1;
    }
  }
  out.push(query.slice(start));
  if (quote) throw new Error("unclosed quote");
  return out.map((x) => x.trim()).filter(Boolean);
}

function expressionOf(model: Seula, query: string, table: Table, today: Date, resolve?: (phrase: string) => Expression<Cond> | undefined): Filter {
  let tokens: string[];
  try { tokens = parts(query, table); } catch { return { conds: null, conf: 0, status: "incomplete", message: "unclosed quote" }; }
  let i = 0, leaves: Filter[] = [];
  const leaf = (): Expression<Cond> => {
    const word = tokens[i++];
    if (word === "(") { const e = or(); if (tokens[i++] !== ")") throw new Error("missing closing parenthesis"); return e; }
    if (!word || word === ")" || ["and", "or", "but", "except"].includes(word)) throw new Error("missing clause");
    if (/^not\s+/i.test(word) && !hasExactCell(table, word)) {
      const phrase = word.replace(/^not\s+/i, "");
      const concept = resolve?.(phrase.toLowerCase().trim());
      if (concept) { leaves.push({ conds: null, conf: 1, status: "ok" }); return { not: concept }; }
      return parseLeaf(word);
    }
    return parseLeaf(word);
  };
  const parseLeaf = (word: string): Expression<Cond> => {
    const concept = resolve?.(word.toLowerCase().trim());
    if (concept) { leaves.push({ conds: null, conf: 1, status: "ok" }); return concept; }
    const f = filterSimple(model, word, table, today); leaves.push(f);
    if ((!f.conds?.length) && leaves.length > 1 && tokens[i - 2] === "or" && /^[\p{L}][\p{L}\p{N}_-]*$/u.test(word)) {
      const prev = leaves.at(-2)?.conds;
      if (prev?.length === 1 && prev[0][1] === "=" && !model.index(table).rowNumeric[prev[0][0]]) { leaves[leaves.length - 1] = { conds: [[prev[0][0], "=", word]], conf: 0.85, status: "ok" }; return [prev[0][0], "=", word]; }
    }
    if (!f.conds?.length || f.status === "incomplete" || f.status === "unsupported") throw new Error(`${f.status}: ${f.message ?? "missing filter"}`);
    return f.conds.length === 1 ? f.conds[0] : { and: f.conds };
  };
  const not = (): Expression<Cond> => tokens[i] === "not" ? (i++, { not: not() }) : leaf();
  const and = (): Expression<Cond> => { let x = not(); while (["and", "but", "except"].includes(tokens[i])) { const op = tokens[i++]; const y = not(); x = { and: [x, op === "except" ? { not: y } : y] }; } return x; };
  const or = (): Expression<Cond> => { let x = and(); while (tokens[i] === "or") { i++; x = { or: [x, and()] }; } return x; };
  try {
    let expression = or(); if (i !== tokens.length) throw new Error("unexpected clause");
    const view = leaves.find((x) => x.sort || x.limit !== undefined);
    if (tokens.includes("and") && !tokens.some((x) => ["or", "but", "except", "(", ")"].includes(x)) && !view && leaves.every((x) => x.conds?.length)) {
      const whole = model.parse(query, table), count = leaves.reduce((n, x) => n + x.conds!.length, 0);
      if (whole.status === "ok" && whole.conf >= 0.8 && whole.conds?.length === count) expression = { and: whole.conds };
    }
    const conf = Math.min(...leaves.map((x) => x.conf)), ambiguous = leaves.some((x) => x.status === "ambiguous") || conf < 0.8;
    return { expression, conds: null, conf, status: ambiguous ? "ambiguous" : "ok", ...(ambiguous && { message: "review the proposed filter" }), ...(view?.sort && { sort: view.sort }), ...(view?.limit !== undefined && { limit: view.limit }) };
  } catch (e) {
    const message = String((e as Error).message);
    return { conds: null, conf: 0, status: /incomplete|missing|unexpected/.test(message) ? "incomplete" : /unsupported/.test(message) ? "unsupported" : "ambiguous", message };
  }
}

function filterSimple(model: Seula, query: string, table: Table, today: Date): Filter {
  if (/(?:\b(?:over|under|above|below|greater than|less than|at least|at most|between)|[<>!=]=?)\s*$/i.test(query))
    return { conds: null, conf: 0, status: "incomplete", message: "comparison needs a value" };
  const { rest, conds, ...view } = extras(query, table, today);
  const exact = [...(vocab(table).exact.get(tokenize(rest).join(" ")) ?? [])];
  if (exact.length === 1 && rest && !/\b(?:and|or|but|except|not)\b/i.test(rest)) return { conds: [[exact[0], "=", rest], ...conds], conf: 1, status: "ok", ...view };
  const r = /[\p{L}\p{N}]/u.test(rest) ? model.parse(rest, table) : { conds: [] as Cond[], conf: 1, status: "ok" as Status };
  const namedAtStart = table.header.map((h, j) => new RegExp(`^(?:the\\s+)?${h.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(rest) ? j : -1).filter((j) => j >= 0);
  if (namedAtStart.length === 1 && r.conds?.length === 1) r.conds = [[namedAtStart[0], /\b(?:has|contains)\b/i.test(rest) && !model.index(table).rowNumeric[namedAtStart[0]] ? "contains" : r.conds[0][1], r.conds[0][2]]];
  if (!r.conds?.length && namedAtStart.length === 1) {
    const h = table.header[namedAtStart[0]];
    const value = rest.replace(new RegExp(`^(?:the\\s+)?${h.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*(?:is|=|:)?\\s+`, "i"), "").replace(/^["']|["']$/g, "").trim();
    if (value && !/\b(?:over|under|above|below|between|after|before|not|isnt|isn't|greater|less|has|contains|at least|at most)\b|[!<>=]/i.test(value) && !/\d/.test(value))
      return { conds: [[namedAtStart[0], "=", value], ...conds], conf: 1, status: "ok", ...view };
  }
  const dateOnly = conds.length && !r.conds?.length && r.status === "ambiguous" && /^(?:games|aired)$/i.test(rest);
  const unknown = r.conds?.some(([j, op, value]) => (op === "=" || op === "contains") && parseNumber(value) === null && ![...vocab(table).byColumn[j]].some((cell) => cell.includes(tokenize(value).join(" "))));
  const unsure = !dateOnly && (unknown || r.status === "ambiguous" || r.conf < 0.8);
  return { conds: r.conds ? [...r.conds, ...conds] : conds.length ? conds : null, conf: dateOnly ? 1 : r.conf, status: dateOnly ? "ok" : unsure ? "ambiguous" : r.status, ...(unsure ? { message: unknown ? "unknown value" : "review the proposed filter" } : !dateOnly && "message" in r && { message: r.message }), ...view };
}

export function filter(model: Seula, query: string, table: Table, today = new Date(), resolve?: (phrase: string) => Expression<Cond> | undefined): Filter {
  if (query.length > 4096) return { conds: null, conf: 0, status: "unsupported", message: "query is too long" };
  try { parts(query, table); } catch { return { conds: null, conf: 0, status: "incomplete", message: "unclosed quote" }; }
  if ((query.match(/[()]/g)?.length ?? 0) > 32) return { conds: null, conf: 0, status: "unsupported", message: "expression is too deep" };
  if (/\b(?:and|or|but|except|not)\s*$/i.test(query)) return { conds: null, conf: 0, status: "incomplete", message: "missing clause" };
  const knownCell = hasExactCell(table, query);
  if (/^(?:total|sum|count|how many)\s+/i.test(query) && !knownCell && !table.header.some((h) => h.toLowerCase() === query.split(/\s+/)[0].toLowerCase())) return { conds: null, conf: 0, status: "unsupported", message: "use the object API for aggregation" };
  const operators = vocab(table).operators.reduce((q, v) => q.replace(v, ""), query.replace(/(["']).*?\1/g, ""));
  if (/\b(?:avg|average|join|group by)\b/i.test(operators)) return { conds: null, conf: 0, status: "unsupported", message: "operation is not supported" };
  const simple = filterSimple(model, query, table, today);
  if ((!/\b(?:and|or|but|except|not)\b|[()]/i.test(query) || /^not\s+/i.test(query) && !/[()]/.test(query) && !/\b(?:and|or|but|except)\b/i.test(query)) && !resolve?.(query.toLowerCase().trim()) && !resolve?.(query.replace(/^not\s+/i, "").toLowerCase().trim())) return simple;
  return expressionOf(model, query, table, today, resolve);
}

// Row indices the filter keeps, in order: sorted by date or number where the column has them (others last), else by text.
export function rowsOf(model: Seula, table: Table, f: Pick<Filter, "conds" | "sort" | "limit"> & Partial<Filter> & { confirmAmbiguous?: boolean }): number[] {
  assertExecutable(f, f.confirmAmbiguous);
  const numeric = model.index(table).rowNumeric;
  const evalExpr = (e: Expression<Cond>, row: Table["rows"][number]): boolean =>
    Array.isArray(e) ? runConds([row], [e as Cond], numeric).length > 0
    : "and" in e ? e.and.every((x) => evalExpr(x, row)) : "or" in e ? e.or.some((x) => evalExpr(x, row)) : !Array.isArray(e) && "not" in e ? !evalExpr(e.not, row) : false;
  const hits = f.expression ? table.rows.flatMap((r, i) => evalExpr(f.expression!, r) ? [i] : []) : runConds(table.rows, f.conds ?? [], numeric);
  if (f.sort) sortBy(hits, (i) => table.rows[i][f.sort![0]], f.sort[1]);
  return f.limit === undefined ? hits : hits.slice(0, f.limit);
}

// The ordering rowsOf and apply share. Sorts items in place and returns them.
export function sortBy<T>(items: T[], cell: (x: T) => unknown, dir: "asc" | "desc"): T[] {
  const s = dir === "asc" ? 1 : -1;
  const key = new Map(items.map((x) => [x, dateOf(cell(x)) ?? parseNumber(String(cell(x)))]));
  return items.sort((a, b) => {
    const x = key.get(a)!, y = key.get(b)!;
    if (x === null || y === null) return x === y ? String(cell(a)).localeCompare(String(cell(b))) * s : x === null ? 1 : -1;
    return (x - y) * s;
  });
}

// Bytes are used as is; a string or URL is fetched; nothing fetches the weights.bin next to this file.
export async function load(weights?: ArrayBuffer | Uint8Array | string | URL) {
  if (weights === undefined || typeof weights === "string" || weights instanceof URL) {
    const url = weights ?? new URL("./weights.bin", import.meta.url);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`seula: could not load weights from ${url} (HTTP ${res.status})`);
    weights = await res.arrayBuffer();
  }
  return wrap(new Seula(weights));
}
