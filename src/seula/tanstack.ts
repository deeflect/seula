// TanStack Table adapter: give every column filterFn: seulaFilterFn, then set state from toTanStack(result).
import { passes, passesExpression, validateExpression, type Condition, type Result } from "./api.js";
import { assertExecutable, type Expression } from "./index.js";

export function toTanStack(result: Pick<Result, "conditions" | "sort" | "limit"> & Partial<Result>, opts: { confirmAmbiguous?: boolean } = {}) {
  assertExecutable(result, opts.confirmAmbiguous);
  if (result.expression) validateExpression(result.expression);
  for (const c of result.conditions ?? []) validateExpression(c);
  if (result.sort && (typeof result.sort.column !== "string" || !result.sort.column || !["asc", "desc"].includes(result.sort.dir))) throw new Error("seula: invalid sort");
  if (result.limit !== undefined && (!Number.isSafeInteger(result.limit) || result.limit < 0)) throw new Error("seula: invalid limit");
  const by = new Map<string, Condition[]>();
  for (const c of result.conditions ?? []) (by.get(c.column) ?? by.set(c.column, []).get(c.column)!).push(c);
  const anchor = (e: Expression<Condition>): string => "and" in e ? anchor(e.and[0]) : "or" in e ? anchor(e.or[0]) : "not" in e ? anchor(e.not) : e.column;
  return {
    columnFilters: result.expression ? [{ id: anchor(result.expression), value: result.expression }] : [...by].map(([id, value]) => ({ id, value })),
    sorting: result.sort ? [{ id: result.sort.column, desc: result.sort.dir === "desc" }] : [],
    limit: result.limit,
    aggregate: result.aggregate,
  };
}

export const seulaFilterFn = (row: { getValue(id: string): unknown }, columnId: string, value: Condition[] | Expression<Condition>) =>
  Array.isArray(value) ? passes(() => row.getValue(columnId), value) : passesExpression((c) => row.getValue(c), value);
