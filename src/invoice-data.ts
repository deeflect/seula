import { filterDatasetRows, type DatasetRow } from "./datasets";
import { MAX_TOKENS, parseCsvOrJsonToRows, parseWithOperatorAliases, SieveModel, tokenizeQuery, type FilterClause, type SchemaDef } from "./engine";

export interface Invoice {
  invoice: string;
  client: string;
  balance_usd: number;
  days_late: number;
  status: "overdue" | "due" | "paid";
}

export const sampleInvoices: Invoice[] = [
  { invoice: "INV-1048", client: "Northstar Studio", balance_usd: 2400, days_late: 18, status: "overdue" },
  { invoice: "INV-1051", client: "Fieldwork Co", balance_usd: 1750, days_late: 12, status: "overdue" },
  { invoice: "INV-1054", client: "Daybreak Coffee", balance_usd: 1000, days_late: 6, status: "overdue" },
  { invoice: "INV-1056", client: "Common Ground", balance_usd: 450, days_late: 3, status: "overdue" },
  { invoice: "INV-1058", client: "Atlas Labs", balance_usd: 3200, days_late: 0, status: "due" },
  { invoice: "INV-1060", client: "Good Form", balance_usd: 1250, days_late: 0, status: "due" },
  { invoice: "INV-1041", client: "Studio Juniper", balance_usd: 0, days_late: 0, status: "paid" },
  { invoice: "INV-1043", client: "Linden Goods", balance_usd: 0, days_late: 0, status: "paid" },
];

const schema: SchemaDef = {
  name: "invoices", description: "Invoice finder", split: "transfer", fields: [
    { name: "balance_usd", kind: "NUMBER", aliases: ["balance"], enumValues: [] },
    { name: "status", kind: "ENUM", aliases: [], enumValues: ["overdue", "due", "paid"] },
    { name: "days_late", kind: "NUMBER", aliases: ["days late"], enumValues: [] },
    { name: "client", kind: "TEXT", aliases: [], enumValues: [] },
    { name: "invoice", kind: "TEXT", aliases: [], enumValues: [] },
  ],
};

const model = new SieveModel();
const aliases = { "at least": "gte", beneath: "lt" } as const;

function invoiceQuery(query: string, invoices: Invoice[]): string {
  const tokens = tokenizeQuery(query, MAX_TOKENS + 1);
  if (tokens.length > MAX_TOKENS) return query;
  const parts: string[] = [];
  let start = 0;
  for (const token of tokens) {
    if (token.norm === "and" && !/^["']/.test(token.text)) {
      parts.push(query.slice(start, token.start).trim());
      start = token.end;
    }
  }
  parts.push(query.slice(start).trim());
  return parts.map((part) => {
    const amount = /^(at\s+least|over|under)\s*(\$?\d+(?:\.\d+)?k?)$/i.exec(part);
    if (amount) return `balance ${amount[1].toLowerCase().replace(/\s+/g, " ") === "at least" ? ">=" : amount[1].toLowerCase() === "over" ? ">" : "<"} ${amount[2]}`;
    const client = /^from\s+(?:"([^"]+)"|'([^']+)'|([\p{L}\p{N}][\p{L}\p{N} .'-]*))$/iu.exec(part);
    if (client) {
      const name = client[1] ?? client[2] ?? client[3];
      if (client[3] && (/\b(?:overdue|due|paid|balance|days|late|client|invoice|over|under|at|least|above|below|more|less|not|or|either)\b/i.test(name) ||
        /\s+\$?\d+(?:\.\d+)?k?$/i.test(name) && !invoices.some((invoice) => invoice.client.toLowerCase().includes(name.toLowerCase())))) return part;
      const quote = name.includes('"') ? "'" : '"';
      if (name.includes(quote)) return part;
      return `client contains ${quote}${name}${quote}`;
    }
    return part;
  }).join(" and ");
}

export function findInvoices(query: string, invoices: Invoice[]): { matches: Invoice[]; clauses: FilterClause[]; error: string } {
  const parsed = parseWithOperatorAliases(model, invoiceQuery(query, invoices), schema, aliases).result;
  if (!parsed.accepted || parsed.clauses.length === 0) return { matches: [], clauses: [], error: !query.trim() ? "Enter a filter to search." :
    tokenizeQuery(query, MAX_TOKENS + 1).some((token) => !/^["']/.test(token.text) && (token.norm === "or" || token.norm === "either")) ?
      "Use AND to combine filters; OR is not supported." : "Use a supported filter, such as “at least $1k” or “from Fieldwork”." };
  const rows: DatasetRow[] = invoices.map((invoice, index) => ({ id: String(index), title: invoice.client, subtitle: "", fields: { ...invoice }, badges: [], rightMeta: "", rightSub: "" }));
  const ids = new Set(filterDatasetRows(rows, parsed.clauses).map((row) => row.id));
  return { matches: invoices.filter((_, index) => ids.has(String(index))), clauses: parsed.clauses, error: "" };
}

const required = ["invoice", "client", "balance_usd", "days_late", "status"] as const;

export function importInvoices(raw: string): Invoice[] {
  const rows = parseCsvOrJsonToRows(raw);
  if (!rows.length) throw new Error("Add a header and at least one invoice row.");
  if (required.some((key) => !Object.hasOwn(rows[0], key))) throw new Error(`Required columns: ${required.join(", ")}. Example: INV-1001,Acme,1000,5,overdue`);
  const seen = new Set<string>();
  return rows.map((row, index) => {
    if (required.some((key) => !Object.hasOwn(row, key))) throw new Error(`Row ${index + 1}: missing a required column.`);
    const invoice = typeof row.invoice === "string" ? row.invoice.trim() : "";
    const client = typeof row.client === "string" ? row.client.trim() : "";
    const balance_usd = Number(row.balance_usd);
    const days_late = Number(row.days_late);
    const status = typeof row.status === "string" ? row.status.trim().toLowerCase() : "";
    if (!invoice || !client) throw new Error(`Row ${index + 1}: invoice and client are required.`);
    if (seen.has(invoice)) throw new Error(`Row ${index + 1}: duplicate invoice ID ${invoice}.`);
    seen.add(invoice);
    if ((typeof row.balance_usd !== "string" && typeof row.balance_usd !== "number") || !String(row.balance_usd).trim() || !Number.isFinite(balance_usd) || balance_usd < 0 ||
      (typeof row.days_late !== "string" && typeof row.days_late !== "number") || !String(row.days_late).trim() || !Number.isFinite(days_late) || days_late < 0) throw new Error(`Row ${index + 1}: balance_usd and days_late must be nonnegative numbers.`);
    const cents = balance_usd * 100;
    const roundedCents = Math.round(cents);
    if ((typeof row.balance_usd === "string" && !/^\+?(?:\d+(?:\.\d{0,2})?|\.\d{1,2})$/.test(row.balance_usd.trim())) ||
      !Number.isSafeInteger(roundedCents) || Math.abs(cents - roundedCents) > Number.EPSILON * Math.max(1, Math.abs(cents))) {
      throw new Error(`Row ${index + 1}: balance_usd supports at most two decimal places.`);
    }
    if (status !== "overdue" && status !== "due" && status !== "paid") throw new Error(`Row ${index + 1}: status must be overdue, due, or paid.`);
    return { invoice, client, balance_usd, days_late, status };
  });
}

export function exportCsv(invoices: Invoice[]): string {
  const cell = (value: string | number): string => {
    let text = String(value);
    if (/^[\s]*[=+\-@]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  };
  return [required.join(","), ...invoices.map((row) => required.map((key) => cell(row[key])).join(","))].join("\r\n");
}
