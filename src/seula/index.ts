// Public entry: the rule pre-pass (dates, yes/no columns, sort, top-N) plus the model on what is left.
// Usage: const f = filter(model, "overdue last 30 days top 5 by balance", table); rowsOf(model, table, f)

import { dateOf, extras, type Extras } from "./extras.js";
import { parseNumber, runConds, Seula, type Cond, type Table } from "./model.js";
import { wrap } from "./api.js";

export { Seula, type Cond, type Table } from "./model.js";
export * from "./api.js";
export type Filter = { conds: Cond[] | null; conf: number; sort?: Extras["sort"]; limit?: number };

export function filter(model: Seula, query: string, table: Table, today = new Date()): Filter {
  const { rest, conds, ...view } = extras(query, table, today);
  const r = /[\p{L}\p{N}]/u.test(rest) ? model.parse(rest, table) : { conds: [], conf: 1 }; // nothing left for the model
  return { conds: r.conds ? [...r.conds, ...conds] : conds.length ? conds : null, conf: r.conf, ...view };
}

// Row indices the filter keeps, in order: sorted by date or number where the column has them (others last), else by text.
export function rowsOf(model: Seula, table: Table, f: Pick<Filter, "conds" | "sort" | "limit">): number[] {
  const hits = runConds(table.rows, f.conds ?? [], model.index(table).rowNumeric);
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
