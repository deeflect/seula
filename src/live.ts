/// <reference types="vite/client" />
import "./invoice.css";
import "./live.css";
import { Seula, filter, rowsOf, type Filter } from "./seula/index";
import { parseCsvOrJsonToRows } from "./engine/autoSchema";
import { benchTables, invoiceTable, withTypes, type LiveTable } from "./live-tables";

const THRESHOLD = 0.5;
// Both are off unless set at build time: where a yes/no verdict on a parse is posted, and where "get in touch" points.
const CAPTURE_URL: string | undefined = import.meta.env.VITE_CAPTURE_URL, CONTACT_URL: string | undefined = import.meta.env.VITE_CONTACT_URL;
const examples: Record<string, string[]> = {
  invoices: ["overdue and balance at least 1k", "overdue or paid, top 3 by balance", "days late between 10 and 40", "overdue clients except fieldwork", "clients who owe a lot"],
  eu: ["euro and ep seats more than 20", "gdp per cap over 30k", "not euro", "big countries with high gini"],
  race: ["points over 25 and laps 97", "grid under 5", "team newman/haas racing", "fast drivers from good teams"],
  episodes: ["directed by marcos siega", "more than 3.5 million viewers", "episodes before 5", "written by julie plec"],
  nfl: ["attendance over 50000", "home games at memorial stadium", "weeks after 10", "games we won big"],
};
const tables: LiveTable[] = [invoiceTable, ...benchTables].map((t) => ({ ...t, examples: examples[t.key] ?? [] }));
const sym: Record<string, string> = { "=": "=", "!=": "≠", contains: "contains", ">": ">", ">=": "≥", "<": "<", "<=": "≤" };
const esc = (v: unknown) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

let model: Seula | null = null;
let table = tables[0];
// Per query: chips the reader removed, whether an unsure guess was applied anyway, and the verdict already sent.
let seen = "", removed = new Set<string>(), forced = false, verdict = "", last: Filter | null = null;

const app = document.getElementById("app")!;
document.title = "Seula: plain-English table filters";
app.innerHTML = `
  <header class="invoice-header">
    <div class="invoice-brand"><span class="brand-dot" aria-hidden="true"></span><strong>Seula</strong><span class="brand-sub">Live filter model</span></div>
    <nav aria-label="Page links"><a href="./?invoice">Invoice demo</a></nav>
  </header>
  <main class="invoice-page live-page">
    <section class="intro-copy"><h1>Type a filter. It parses as you type.</h1><p>A small model in this tab turns your words into table filters on every keystroke, using the values that are really in the table.</p></section>
    <section class="live-picker" aria-label="Table"><div class="live-tabs" id="tabs"></div><label class="live-csv">Use a CSV<input type="file" id="csv" accept=".csv,.json,text/csv" /></label></section>
    <div class="search-card">
      <label for="q" id="q-label"></label>
      <input id="q" class="live-input" autocomplete="off" spellcheck="false" />
      <div class="examples" id="examples"></div>
    </div>
    <section class="live-parse" aria-live="polite" id="parse"></section>
    <section id="results"></section>
    <footer><span>184k-parameter model · 182 KB · runs locally, no API calls</span>${CONTACT_URL ? `<a href="${esc(CONTACT_URL)}">Want this in your product? Get in touch</a>` : ""}</footer>
  </main>`;
const input = app.querySelector<HTMLInputElement>("#q")!;

function selectTable(t: LiveTable): void {
  table = t;
  app.querySelector("#tabs")!.innerHTML = tables.map((x) => `<button type="button" data-key="${esc(x.key)}" aria-pressed="${x === t}">${esc(x.name)}</button>`).join("");
  app.querySelector("#q-label")!.textContent = `Filter ${t.name} · ${t.rows.length} rows`;
  app.querySelector("#examples")!.innerHTML = t.examples.length ? `<span>Try:</span>${t.examples.map((q) => `<button type="button" data-q="${esc(q)}">${esc(q)}</button>`).join('<span aria-hidden="true">·</span>')}` : "";
  input.value = t.examples[0] ?? "";
  update();
}

function update(): void {
  const parseEl = app.querySelector("#parse")!;
  const q = input.value.trim();
  let r: Filter | null = null, ms = 0;
  if (model && q) { const t0 = performance.now(); r = filter(model, q, table); ms = performance.now() - t0; }
  if (seen !== `${table.key}\n${q}`) { seen = `${table.key}\n${q}`; removed = new Set(); forced = false; verdict = ""; }
  last = r;
  const unsure = !!r && (!r.conds || r.conf < THRESHOLD), guess = unsure && !forced;
  const status = !model ? `<span class="live-status">Loading model…</span>`
    : !q ? `<span class="live-status">Type a filter to start.</span>`
    : !r!.conds ? `<span class="live-status unsure">No filter found</span>`
    : guess ? `<span class="live-status unsure">Unsure. The guess is shown, not applied. <button type="button" data-force>Apply anyway</button></span>`
    : `<span class="live-status ok">${unsure ? "Guess applied" : "Confident. Filters applied"}</span>`;
  // One chip per column and operator, so "overdue or pending" reads as one filter. Removing a chip drops that part.
  const by = new Map<string, string[]>();
  for (const [j, op, v] of r?.conds ?? []) by.set(`${j} ${op}`, [...(by.get(`${j} ${op}`) ?? []), v]);
  const labels = [...by].map(([k, vs]) => [k, `${table.header[+k.split(" ")[0]]} ${sym[k.split(" ")[1]] ?? k.split(" ")[1]} ${vs.join(/ (=|contains)$/.test(k) ? " or " : ", ")}`]);
  if (r?.sort) labels.push(["sort", `Sort: ${table.header[r.sort[0]]} ${r.sort[1] === "asc" ? "↑" : "↓"}`]);
  if (r?.limit !== undefined) labels.push(["limit", `First ${r.limit}`]);
  const chips = labels.filter(([k]) => !removed.has(k)).map(([k, l]) => `<span class="chip${guess ? " guess" : ""}">${esc(l)}<button type="button" data-rm="${esc(k)}" aria-label="Remove ${esc(l)}">×</button></span>`).join("");
  const feedback = !CAPTURE_URL || !r || !q ? "" : verdict ? `<p class="live-feedback">Sent, thank you.</p>`
    : `<p class="live-feedback"><span>Did it read this right?</span><button type="button" data-verdict="right">Yes</button><button type="button" data-verdict="wrong">No</button><small>Answering sends your query, the column names and the parsed filter. The rows never leave this tab.</small></p>`;
  parseEl.innerHTML = `
    <div class="live-chips">${chips || `<span class="live-none">${q && model ? "No filters" : ""}</span>`}</div>
    <div class="live-metrics">
      <div class="live-metric"><span>Confidence</span><strong>${r ? r.conf.toFixed(2) : "—"}</strong><span class="live-meter" aria-hidden="true"><i style="width:${r ? r.conf * 100 : 0}%" class="${unsure ? "low" : ""}"></i><b></b></span></div>
      <div class="live-metric"><span>Parse time</span><strong>${r ? `${ms.toFixed(2)} ms` : "—"}</strong></div>
    </div>
    ${status}${feedback}`;
  const kept = r?.conds && !guess ? { conds: r.conds.filter(([j, op]) => !removed.has(`${j} ${op}`)), sort: removed.has("sort") ? undefined : r.sort, limit: removed.has("limit") ? undefined : r.limit } : { conds: [] };
  renderResults(kept, guess && !!r?.conds);
}

function renderResults(f: Pick<Filter, "conds" | "sort" | "limit">, unsure: boolean): void {
  const hits = model ? rowsOf(model, table, f) : table.rows.map((_, i) => i);
  const hitCols = new Set((f.conds ?? []).map(([j]) => j));
  app.querySelector("#results")!.innerHTML = `
    <p class="live-count"><strong>${hits.length}</strong> of ${table.rows.length} rows${unsure ? " · guess not applied" : ""}</p>
    <div class="live-scroll"><table class="live-table">
      <thead><tr>${table.header.map((h, j) => `<th class="${hitCols.has(j) ? "hit" : ""}">${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${hits.map((i) => `<tr>${table.rows[i].map((c, j) => `<td class="${hitCols.has(j) ? "hit" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    ${hits.length ? "" : `<div class="state-message"><strong>No rows match.</strong><span>The filter parsed, but no row passes it.</span></div>`}`;
}

input.addEventListener("input", update);
app.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-key], button[data-q], button[data-rm], button[data-force], button[data-verdict]");
  if (!b) return;
  if (b.dataset.rm) { removed.add(b.dataset.rm); update(); }
  else if ("force" in b.dataset) { forced = true; update(); }
  else if (b.dataset.verdict && CAPTURE_URL && last) {
    verdict = b.dataset.verdict;
    // text/plain keeps this a simple request (no CORS preflight); a failed send is not the reader's problem
    fetch(CAPTURE_URL, { method: "POST", headers: { "content-type": "text/plain" }, keepalive: true, body: JSON.stringify({ verdict, query: input.value.trim(), header: table.header, conds: last.conds, conf: last.conf, sort: last.sort, limit: last.limit, removed: [...removed], forced }) }).catch(() => {});
    update();
  }
  else if (b.dataset.key) selectTable(tables.find((t) => t.key === b.dataset.key)!);
  else { input.value = b.dataset.q!; update(); input.focus(); }
});
app.querySelector<HTMLInputElement>("#csv")!.addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const objs = parseCsvOrJsonToRows(await file.text());
    if (!objs.length) throw new Error("No rows found.");
    const header = Object.keys(objs[0]);
    const t = withTypes({ key: "csv", name: file.name, examples: [], header, rows: objs.map((o) => header.map((h) => String(o[h] ?? ""))) });
    const i = tables.findIndex((x) => x.key === "csv");
    i < 0 ? tables.push(t) : (tables[i] = t);
    selectTable(t);
  } catch (err) { alert(err instanceof Error ? err.message : "Could not read this file."); }
});

selectTable(table);
input.focus();
fetch(new URL("./seula/weights.bin", import.meta.url))
  .then((res) => res.arrayBuffer())
  .then((buf) => { model = new Seula(buf); model.parse("warm up", table); update(); })
  .catch(() => { app.querySelector("#parse")!.innerHTML = `<div class="state-message error" role="alert"><strong>Could not load the model.</strong></div>`; });
