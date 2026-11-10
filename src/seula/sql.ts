// SQL adapter: WHERE / ORDER BY text with every value as a parameter. Column names come from the parse result (the table header) and are quoted.
import { groups, type Condition, type Result } from "./api.js";

const CMP = new Set([">", ">=", "<", "<="]);
// "!" as the LIKE escape: a backslash there is read differently by MySQL than by SQLite and Postgres
const like = (v: string | number) => `%${String(v).toLowerCase().replace(/[!%_]/g, "!$&")}%`;

export function toSql(
  result: Pick<Result, "conditions" | "sort" | "limit">,
  opts: { quote?: (name: string) => string; placeholder?: (index: number) => string } = {},
) {
  const quote = opts.quote ?? ((n: string) => `"${n.replaceAll('"', '""')}"`), ph = opts.placeholder ?? (() => "?");
  const params: (string | number)[] = [];
  const p = (v: string | number) => (params.push(v), ph(params.length));
  const clause = (c: Condition): string => {
    const col = quote(c.column), vals = Array.isArray(c.value) ? c.value : [c.value];
    if (c.op === "=" || c.op === "!=") return `${col} ${c.op === "=" ? "=" : "<>"} ${p(vals[0])}`;
    if (c.op === "in" || c.op === "not in") return `${col} ${c.op === "in" ? "IN" : "NOT IN"} (${vals.map(p).join(", ")})`;
    if (c.op === "contains" || c.op === "contains any") return vals.map((v) => `LOWER(${col}) LIKE ${p(like(v))} ESCAPE '!'`).join(" OR ");
    if (CMP.has(c.op)) return `${col} ${c.op} ${p(vals[0])}`;
    throw new Error(`seula: unknown op ${JSON.stringify(c.op)}`); // op is written into the SQL, so only known ones
  };
  const { all, any } = groups(result.conditions ?? []); // clauses are built in output order so ? placeholders line up
  const where = [...all.map(clause), ...any.map((g) => `(${g.map(clause).join(" OR ")})`)].join(" AND ") || "1=1";
  const orderBy = result.sort ? `${quote(result.sort.column)} ${result.sort.dir === "asc" ? "ASC" : "DESC"}` : "";
  return { where, orderBy, limit: result.limit, params };
}
