// The 8 sample invoices, on their own so the landing can import them without the old engine.
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
