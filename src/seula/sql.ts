// SQL adapter: WHERE / ORDER BY text with every value as a parameter. Column names come from the parse result (the table header) and are quoted.
import { groups, validateExpression, type Condition, type Result } from "./api.js";
import { assertExecutable, type Expression } from "./index.js";

const CMP = new Set([">", ">=", "<", "<="]);
// "!" as the LIKE escape: a backslash there is read differently by MySQL than by SQLite and Postgres
const like = (v: string | number) => `%${String(v).toLowerCase().replace(/[!%_]/g, "!$&")}%`;

export function toSql(
  result: Pick<Result, "conditions" | "sort" | "limit"> & Partial<Result>,
  opts: { quote?: (name: string) => string; placeholder?: (index: number) => string; confirmAmbiguous?: boolean } = {},
) {
  assertExecutable(result, opts.confirmAmbiguous);
  if (result.expression) validateExpression(result.expression);
  for (const c of result.conditions ?? []) validateExpression(c);
  if (result.sort && (typeof result.sort.column !== "string" || !result.sort.column || !["asc", "desc"].includes(result.sort.dir))) throw new Error("seula: invalid sort");
  if (result.limit !== undefined && (!Number.isSafeInteger(result.limit) || result.limit < 0)) throw new Error("seula: invalid limit");
  if (result.aggregate && !(result.aggregate.op === "count" || result.aggregate.op === "sum" && typeof result.aggregate.column === "string" && !!result.aggregate.column)) throw new Error("seula: invalid aggregate");
  const quote = opts.quote ?? ((n: string) => `"${n.replaceAll('"', '""')}"`), ph = opts.placeholder ?? (() => "?");
  const params: (string | number)[] = [];
  const p = (v: string | number) => (params.push(v), ph(params.length));
  const clause = (c: Condition): string => {
    const col = c.type === "text" ? `COALESCE(${quote(c.column)}, '')` : quote(c.column), vals = Array.isArray(c.value) ? c.value : [c.value];
    let sql: string;
    if (c.op === "=" || c.op === "!=") sql = `${col} ${c.op === "=" ? "=" : "<>"} ${p(vals[0])}`;
    else if (c.op === "in" || c.op === "not in") sql = `${col} ${c.op === "in" ? "IN" : "NOT IN"} (${vals.map(p).join(", ")})`;
    else if (c.op === "contains" || c.op === "contains any") sql = vals.map((v) => `LOWER(${col}) LIKE ${p(like(v))} ESCAPE '!'`).join(" OR ");
    else if (CMP.has(c.op)) sql = `${col} ${c.op} ${p(vals[0])}`;
    else throw new Error(`seula: unknown op ${JSON.stringify(c.op)}`);
    return c.type === "text" ? `(${sql})` : `COALESCE((${sql}), FALSE)`;
  };
  const expr = (e: Expression<Condition>): string => {
    if ("and" in e) return `(${e.and.map(expr).join(" AND ")})`;
    if ("or" in e) return `(${e.or.map(expr).join(" OR ")})`;
    if ("not" in e) return `(NOT ${expr(e.not)})`;
    return `(${clause(e)})`;
  };
  const { all, any } = groups(result.conditions ?? []);
  const where = result.expression ? expr(result.expression) : [...all.map(clause), ...any.map((g) => `(${g.map(clause).join(" OR ")})`)].join(" AND ") || "1=1";
  const orderBy = result.sort ? `${quote(result.sort.column)} ${result.sort.dir === "asc" ? "ASC" : "DESC"}` : "";
  if (result.aggregate && (orderBy || result.limit !== undefined)) throw new Error("seula: aggregate cannot use sort or limit");
  const select = result.aggregate?.op === "count" ? "COUNT(*)" : result.aggregate?.op === "sum" && result.aggregate.column ? `COALESCE(SUM(${quote(result.aggregate.column)}), 0)` : "*";
  return { select, where, orderBy, limit: result.limit, params };
}
