// TanStack Table adapter: give every column filterFn: seulaFilterFn, then set state from toTanStack(result).
import { passes, type Condition, type Result } from "./api.js";

export function toTanStack(result: Pick<Result, "conditions" | "sort" | "limit">) {
  const by = new Map<string, Condition[]>();
  for (const c of result.conditions ?? []) (by.get(c.column) ?? by.set(c.column, []).get(c.column)!).push(c);
  return {
    columnFilters: [...by].map(([id, value]) => ({ id, value })),
    sorting: result.sort ? [{ id: result.sort.column, desc: result.sort.dir === "desc" }] : [],
    limit: result.limit,
  };
}

export const seulaFilterFn = (row: { getValue(id: string): unknown }, columnId: string, value: Condition[]) => passes(() => row.getValue(columnId), value);
