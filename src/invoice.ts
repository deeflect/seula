import "./invoice.css";
import { exportCsv, findInvoices, importInvoices, sampleInvoices, type Invoice } from "./invoice-data";

const money = (value: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value);
const escape = (value: unknown) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!);
const examples = [
  ["All overdue", "overdue"],
  ["At least $1k", "overdue and balance at least $1k"],
  ["Small balances", "overdue and balance beneath $1k"],
];
const names: Record<string, string> = { balance_usd: "Balance", status: "Status", days_late: "Days late", client: "Client", invoice: "Invoice" };
const symbols: Record<string, string> = { eq: "=", neq: "≠", gt: ">", gte: "≥", lt: "<", lte: "≤", contains: "contains" };
let invoices: Invoice[] = sampleInvoices;
let query = "overdue and balance at least $1k";
let rawImport = "";
let importError = "";
let importOpen = false;
let imported = false;

function render(): void {
  const app = document.getElementById("app")!;
  const { matches, clauses, error } = findInvoices(query, invoices);
  const count = matches.length;
  const total = matches.reduce((sum, row) => sum + row.balance_usd, 0);
  app.innerHTML = `
    <header class="invoice-header">
      <div class="invoice-brand"><span class="brand-dot" aria-hidden="true"></span><strong>Sieve</strong><span class="brand-sub">Invoice finder</span></div>
      <nav aria-label="Page links"><a href="#how-it-works">How it works</a><button type="button" id="open-import">Try your CSV</button></nav>
    </header>
    <main class="invoice-page">
      <section class="intro" aria-labelledby="page-title">
        <div class="intro-copy"><p class="eyebrow">Your follow-up list, without the spreadsheet hunt.</p><h1 id="page-title">Find the invoices worth chasing.</h1><p>Describe a filter. Check the matches. Export your list.</p></div>
        <div class="summary" aria-live="polite"><span>Matching balance</span><strong>${error ? "—" : money(total)}</strong><small>${error ? "Search not applied" : `${count} of ${invoices.length} ${imported ? "imported" : "sample"} invoices`}</small></div>
      </section>
      <form id="search-form" class="search-card"><label for="invoice-query">Which invoices do you need?</label><div class="search-line"><input id="invoice-query" name="query" value="${escape(query)}" autocomplete="off" /><button type="submit" class="primary">Find invoices <span aria-hidden="true">→</span></button></div><div class="examples"><span>Try:</span>${examples.map(([label, value]) => `<button type="button" data-example="${escape(value)}">${escape(label)}</button>`).join("<span aria-hidden=\"true\">·</span>")}</div></form>
      ${error ? `<div class="state-message error" role="alert"><strong>We couldn’t apply that filter.</strong><span>${escape(error)}</span></div>` : `
      <section class="filter-bar" aria-label="Search results"><div class="filters"><span>Applied filters</span>${clauses.map((clause) => `<span class="chip">${clause.negated ? "Not " : ""}${escape(names[clause.field] || clause.field)} ${escape(symbols[clause.cmp] || clause.cmp)} ${escape(clause.field === "balance_usd" ? money(Number(clause.value)) : clause.value)}</span>`).join("")}</div><button type="button" id="export" ${count ? "" : "disabled"}>Export ${count} invoice${count === 1 ? "" : "s"} <span aria-hidden="true">↓</span></button></section>
      ${count ? `<div class="invoice-table" role="table" aria-label="Matching invoices"><div class="table-head" role="row"><span role="columnheader">Invoice</span><span role="columnheader">Client</span><span role="columnheader">Balance</span><span role="columnheader">Days late</span><span role="columnheader">Status</span></div>${matches.map((row) => `<div class="invoice-row" role="row"><span role="cell" class="row-id" data-label="Invoice">${escape(row.invoice)}</span><span role="cell" class="row-client" data-label="Client">${escape(row.client)}</span><span role="cell" class="row-balance" data-label="Balance">${money(row.balance_usd)}</span><span role="cell" class="row-days" data-label="Days late"><span class="mobile-status">${escape(row.status)} · </span>${row.days_late} days</span><span role="cell" class="row-status" data-label="Status">${escape(row.status)}</span></div>`).join("")}</div>` : `<div class="state-message"><strong>No invoices match those filters.</strong><span>Try another filter or reset to sample data.</span></div>`}`}
      <footer><span>${imported ? "Imported data" : "Sample data"} · USD · Your imported data stays in this browser.</span><span>Small model. Explicit filters. No API call to search. <a href="?lab=1">Parser lab</a></span></footer>
      <details id="how-it-works"><summary>How it works</summary><p>A small trained model reads supported invoice fields. Explicit aliases handle “at least” and “beneath”; validation rejects unsupported filters. Matching and CSV export run in your browser. This is a focused demo, not a general AI assistant.</p></details>
      <section class="import-panel" id="import-panel" ${importOpen ? "" : "hidden"} aria-labelledby="import-heading"><div class="import-heading"><h2 id="import-heading">Try your CSV</h2><button type="button" id="close-import" aria-label="Close import">×</button></div><p>USD only. Required columns: invoice,client,balance_usd,days_late,status. Status: overdue, due, or paid.</p><p>Example: <code>INV-1001,Acme,1000,5,overdue</code></p><label for="import-text">Paste CSV</label><textarea id="import-text" rows="6" placeholder="invoice,client,balance_usd,days_late,status&#10;INV-1001,Acme,1000,5,overdue">${escape(rawImport)}</textarea><label for="import-file">Or select a CSV file</label><input type="file" id="import-file" accept=".csv,text/csv" />${importError ? `<p class="import-error" role="alert">${escape(importError)}</p>` : ""}<div class="import-actions"><button type="button" id="import-submit" class="primary">Use these invoices</button><button type="button" id="sample-reset">Reset sample data</button></div></section>
    </main>`;
  app.querySelector<HTMLFormElement>("#search-form")!.addEventListener("submit", (event) => { event.preventDefault(); query = app.querySelector<HTMLInputElement>("#invoice-query")!.value; render(); app.querySelector<HTMLInputElement>("#invoice-query")!.focus(); });
  app.querySelectorAll<HTMLButtonElement>("[data-example]").forEach((button) => button.addEventListener("click", () => { query = button.dataset.example!; render(); }));
  app.querySelector<HTMLButtonElement>("#open-import")!.addEventListener("click", () => { importOpen = true; render(); app.querySelector<HTMLTextAreaElement>("#import-text")!.focus(); });
  app.querySelector<HTMLButtonElement>("#close-import")!.addEventListener("click", () => { importOpen = false; render(); app.querySelector<HTMLButtonElement>("#open-import")!.focus(); });
  app.querySelector<HTMLTextAreaElement>("#import-text")!.addEventListener("input", (event) => { rawImport = (event.target as HTMLTextAreaElement).value; });
  app.querySelector<HTMLInputElement>("#import-file")!.addEventListener("change", async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) { try { rawImport = await file.text(); importError = ""; } catch { importError = "Could not read this file."; } render(); }
  });
  app.querySelector<HTMLAnchorElement>('a[href="#how-it-works"]')!.addEventListener("click", () => { app.querySelector<HTMLDetailsElement>("#how-it-works")!.open = true; });
  app.querySelector<HTMLButtonElement>("#import-submit")!.addEventListener("click", () => {
    try { const next = importInvoices(rawImport); invoices = next; imported = true; importError = ""; importOpen = false; render(); }
    catch (reason) { importError = reason instanceof Error ? reason.message : "Could not read this CSV."; render(); app.querySelector<HTMLTextAreaElement>("#import-text")!.focus(); }
  });
  app.querySelector<HTMLButtonElement>("#sample-reset")!.addEventListener("click", () => { invoices = sampleInvoices; imported = false; importOpen = false; importError = ""; render(); });
  app.querySelector<HTMLButtonElement>("#export")?.addEventListener("click", () => {
    const blob = new Blob([exportCsv(matches)], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = "sieve-invoices.csv"; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
  });
}

render();
