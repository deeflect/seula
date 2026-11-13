/// <reference types="vite/client" />
import "./live.css";
import { Seula, wrap, type Condition, type Result } from "./seula/index";
import { parseCsvOrJsonToRows } from "./engine/autoSchema";
import { benchTables, invoiceTable, withTypes, type LiveTable } from "./live-tables";
import { chipsHtml, esc, expressionLabel, fadeEdges, icon, kind, linkColumns, morph, sym, type Part } from "./ui";

// Both are off unless set at build time: where a yes/no verdict on a parse is posted, and where "get in touch" points.
const CAPTURE_URL: string | undefined = import.meta.env.VITE_CAPTURE_URL, CONTACT_URL: string | undefined = import.meta.env.VITE_CONTACT_URL;
const examples: Record<string, string[]> = {
  invoices: ["overdue and balance at least 1k", "overdue and (balance over 1k or days late over 30)", "total overdue balance", "at risk except Fieldwork", "overdue or paid, top 3 by balance"],
  eu: ["euro and ep seats more than 20", "gdp per cap over 30k", "not euro", "big countries with high gini"],
  race: ["points over 25 and laps 97", "grid under 5", "team newman/haas racing", "fast drivers from good teams"],
  episodes: ["directed by marcos siega", "more than 3.5 million viewers", "episodes before 5", "written by julie plec"],
  nfl: ["attendance over 50000", "home games at memorial stadium", "weeks after 10", "games we won big"],
};
const tables: LiveTable[] = [invoiceTable, ...benchTables].map((t) => ({ ...t, examples: examples[t.key] ?? [] }));

let model: Seula | null = null;
let table = tables[0];
let objects: Record<string, unknown>[] = [], indexes = new Map<Record<string, unknown>, number>();
let api: ReturnType<ReturnType<typeof wrap>["table"]> | null = null;
// Per query: chips the reader removed, whether an unsure guess was applied anyway, and the verdict already sent.
let seen = "", removed = new Set<string>(), forced = false, verdict = "", last: Result | null = null;

const app = document.getElementById("app")!;
app.innerHTML = `
  <header class="bar">
    <div class="bar-in">
      <a class="brand" href="/"><span class="logo" aria-hidden="true"></span>Seula</a>
      <a class="pill" href="https://github.com/deeflect/seula" target="_blank" rel="noopener">GitHub<svg class="i ext" aria-hidden="true"><use href="/icons.svg#arrow-up-right" /></svg></a>
    </div>
  </header>
  <main class="col">
    <section class="sec">
    <div class="head">
      <h1>Live demo</h1>
      <label class="btn btn-secondary upload">${icon("upload-simple")}CSV<input type="file" id="csv" class="visually-hidden" accept=".csv,.json,text/csv" /></label>
    </div>
    <div class="tabs" id="tabs" role="group" aria-label="Table"></div>
    <div class="card">
      <div class="qrow">
        <span class="slot" aria-hidden="true">${icon("funnel-simple")}</span>
        <label for="q" id="q-label" class="visually-hidden"></label>
        <input id="q" class="qin" autocomplete="off" spellcheck="false" />
      </div>
      <div class="examples" id="examples"></div>
      <div id="alert" role="alert"></div>
      <div id="parse" aria-live="polite"></div>
      <div id="results"></div>
      <div class="facts" id="facts"></div>
      <div id="verdict"></div>
    </div>
    </section>
    <footer class="foot">
      <div>
        ${CONTACT_URL ? `<p>Want this in your product? Get in touch at <a class="text-link" href="${esc(CONTACT_URL)}">${esc(CONTACT_URL.replace(/^https?:\/\//, "").replace(/\/$/, ""))}</a>.</p>` : ""}
        <p>Seula is Finnish for sieve. Made by <a class="text-link" href="https://deeflect.com">Dee</a>.</p>
      </div>
      <div class="foot-row">
        <div class="foot-rings">
          <a class="ring" href="https://github.com/deeflect/seula" target="_blank" rel="noopener" aria-label="GitHub">${icon("github-logo")}</a>
          <a class="ring" href="https://www.npmjs.com/package/seula" target="_blank" rel="noopener" aria-label="npm">${icon("package")}</a>
        </div>
        <small>Apache-2.0 · © 2026</small>
      </div>
    </footer>
  </main>`;
const input = app.querySelector<HTMLInputElement>("#q")!;
const alertEl = app.querySelector("#alert")!;

function selectTable(t: LiveTable): void {
  table = t;
  objects = t.rows.map((row) => Object.fromEntries(t.header.map((h, j) => [h, row[j]])));
  indexes = new Map(objects.map((row, i) => [row, i]));
  const late = t.header.find((h) => h.toLowerCase() === "days late"), balance = t.header.find((h) => h.toLowerCase() === "balance");
  api = model ? wrap(model).table(objects, t.key === "invoices" && late && balance ? { concepts: { "at risk": [{ column: late, type: "number", op: ">", value: 10 }, { column: balance, type: "number", op: ">", value: 1000 }] } } : {}) : null;
  app.querySelector("#tabs")!.innerHTML = tables.map((x) => `<button type="button" class="tab" data-key="${esc(x.key)}" aria-pressed="${x === t}">${esc(x.name)}</button>`).join("");
  app.querySelector("#q-label")!.textContent = `Filter ${t.name}`;
  app.querySelector("#examples")!.innerHTML = t.examples.map((q) => `<button type="button" class="ex" data-q="${esc(q)}">${esc(q)}</button>`).join("");
  input.value = t.examples[0] ?? "";
  update();
}

function update(): void {
  const parseEl = app.querySelector("#parse")!;
  const q = input.value.trim();
  let r: Result | null = null, ms = 0;
  if (api && q) { const t0 = performance.now(); r = api.parse(q); ms = performance.now() - t0; }
  if (seen !== `${table.key}\n${q}`) { seen = `${table.key}\n${q}`; removed = new Set(); forced = false; verdict = ""; }
  last = r;
  const guess = r?.status === "ambiguous" && !forced;
  // One chip per column and operator, so "overdue or pending" reads as one filter. Removing a chip drops that part.
  const by = new Map<string, string[]>();
  for (const c of r?.conditions ?? []) by.set(`${c.column} ${c.op}`, [...(by.get(`${c.column} ${c.op}`) ?? []), ...(Array.isArray(c.value) ? c.value : [c.value]).map(String)]);
  const parts: Part[] = [...by].map(([k, vs]) => {
    const c = r!.conditions!.find((x) => `${x.column} ${x.op}` === k)!, j = table.header.indexOf(c.column);
    return { key: k, col: j, kind: kind[c.op] ?? "value", label: `${c.column} ${sym[c.op] ?? c.op} ${vs.join(/ (=|contains)$/.test(k) ? " or " : ", ")}` };
  });
  if (r?.expression) parts.push({ key: "expression", kind: "value", label: expressionLabel(r.expression, (c: Condition) => `${c.column} ${sym[c.op] ?? c.op} ${Array.isArray(c.value) ? c.value.join(", ") : c.value}`) });
  if (r?.aggregate) parts.push({ key: "aggregate", kind: "order", label: r.aggregate.op === "count" ? "Count matching rows" : `Sum ${r.aggregate.column}` });
  if (r?.sort) parts.push({ key: "sort", col: table.header.indexOf(r.sort.column), kind: "order", label: `Sort: ${r.sort.column} ${r.sort.dir === "asc" ? "↑" : "↓"}` });
  if (r?.limit !== undefined) parts.push({ key: "limit", kind: "order", label: `First ${r.limit}` });
  const chips = chipsHtml(parts.filter((p) => !removed.has(p.key)), guess);
  const note = !api ? "Loading model…" : r?.status !== "ok" ? r?.message ?? r?.status : "";
  parseEl.innerHTML = chips || note ? `<div class="chips">${chips}${chips && guess && (r?.conditions?.length || r?.expression) ? `<button type="button" class="btn btn-secondary" data-force>Confirm guess</button>` : ""}${note ? `<span class="note">${esc(note)}</span>` : ""}</div>` : "";
  app.querySelector("#verdict")!.innerHTML = !CAPTURE_URL || !r || !q ? "" : `<div class="verdict">${verdict ? `<span class="note">Sent, thank you.</span>`
    : `<button type="button" class="ring" data-verdict="right" aria-label="Read it right">${icon("thumbs-up")}</button><button type="button" class="ring" data-verdict="wrong" aria-label="Read it wrong">${icon("thumbs-down")}</button>`}<small>Answering sends your query, the column names and the parsed filter. The rows never leave this tab.</small></div>`;
  const canApply = !!r && (r.status === "ok" || r.status === "ambiguous" && forced);
  const kept: Result | null = canApply && r ? { ...r, conditions: r.conditions?.filter((c) => !removed.has(`${c.column} ${c.op}`)) ?? null, expression: removed.has("expression") ? undefined : r.expression, aggregate: removed.has("aggregate") ? undefined : r.aggregate, sort: removed.has("sort") ? undefined : r.sort, limit: removed.has("limit") ? undefined : r.limit } : null;
  const hits = renderResults(kept, !!r);
  const total = kept?.aggregate && api ? api.aggregate({ ...kept, confirmAmbiguous: forced }) : undefined;
  app.querySelector("#facts")!.innerHTML = `${kept || !r ? `<span class="fact">${icon("rows")}${hits} of ${table.rows.length} rows</span>` : `<span class="fact">${icon("rows")}No rows previewed</span>`}${total !== undefined ? `<span class="fact">${icon("gauge")}${esc(kept!.aggregate!.op === "count" ? `Count: ${total}` : `Sum: ${total}`)}</span>` : ""}${r ? `<span class="fact">${icon("lightning")}${ms.toFixed(2)} ms</span><span class="fact">${icon("gauge")}${r.confidence.toFixed(2)}</span>` : ""}`;
}

function renderResults(f: Result | null, query: boolean): number {
  const hits = f && api ? api.apply({ ...f, confirmAmbiguous: forced }).map((row) => indexes.get(row)!) : query ? [] : table.rows.map((_, i) => i);
  const hitCols = new Set((f?.conditions ?? []).map((c) => table.header.indexOf(c.column)));
  if (f?.sort) hitCols.add(table.header.indexOf(f.sort.column));
  const cls = (j: number) => `class="${[table.types[j] === "real" ? "num" : "", hitCols.has(j) ? "hit" : ""].join(" ").trim()}" data-c="${j}"`;
  const el = app.querySelector<HTMLElement>("#results")!;
  const left = el.querySelector(".scroll")?.scrollLeft ?? 0;
  el.innerHTML = `
    <div class="scroll"><table class="rows">
      <thead><tr>${table.header.map((h, j) => `<th ${cls(j)}>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${hits.map((i) => `<tr data-r="${esc(table.key)}-${i}">${table.rows[i].map((c, j) => `<td ${cls(j)}>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    ${hits.length ? "" : `<p class="empty">${f ? "No rows match." : query ? "Confirm the interpretation to preview rows." : "No rows."}</p>`}`;
  const sc = el.querySelector<HTMLElement>(".scroll")!;
  sc.scrollLeft = left;
  fadeEdges(sc);
  return hits.length;
}

input.addEventListener("input", () => update());
linkColumns(app.querySelector<HTMLElement>("#parse")!, () => app.querySelector<HTMLTableElement>("#results .rows"));
// Clicked changes morph the rows; typing swaps them at once.
const tbody = () => app.querySelector<HTMLTableSectionElement>("#results tbody");
app.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-key], button[data-q], button[data-rm], button[data-force], button[data-verdict]");
  if (!b) return;
  if (b.dataset.rm) { removed.add(b.dataset.rm); morph(tbody, update); input.focus(); }
  else if ("force" in b.dataset) { forced = true; morph(tbody, update); input.focus(); }
  else if (b.dataset.verdict && CAPTURE_URL && last) {
    verdict = b.dataset.verdict;
    // text/plain keeps this a simple request (no CORS preflight); a failed send is not the reader's problem
    fetch(CAPTURE_URL, { method: "POST", headers: { "content-type": "text/plain" }, keepalive: true, body: JSON.stringify({ verdict, query: input.value.trim(), header: table.header, conditions: last.conditions, expression: last.expression, aggregate: last.aggregate, status: last.status, message: last.message, confidence: last.confidence, sort: last.sort, limit: last.limit, removed: [...removed], forced }) }).catch(() => {});
    update();
  }
  else if (b.dataset.key) { alertEl.innerHTML = ""; selectTable(tables.find((t) => t.key === b.dataset.key)!); }
  else { input.value = b.dataset.q!; morph(tbody, update); input.focus(); }
});
app.querySelector<HTMLInputElement>("#csv")!.addEventListener("change", async (e) => {
  const el = e.target as HTMLInputElement, file = el.files?.[0];
  if (!file) return;
  alertEl.innerHTML = "";
  try {
    const objs = parseCsvOrJsonToRows(await file.text());
    if (!objs.length) throw new Error("No rows found.");
    const header = Object.keys(objs[0]);
    const t = withTypes({ key: "csv", name: file.name, examples: [], header, rows: objs.map((o) => header.map((h) => String(o[h] ?? ""))) });
    const i = tables.findIndex((x) => x.key === "csv");
    i < 0 ? tables.push(t) : (tables[i] = t);
    selectTable(t);
  } catch (err) { alertEl.innerHTML = `<p class="error">${esc(err instanceof Error ? err.message : "Could not read this file.")}</p>`; }
  el.value = "";
});

selectTable(table);
input.focus();
fetch(new URL("./seula/weights.bin", import.meta.url))
  .then((res) => res.arrayBuffer())
  .then((buf) => { model = new Seula(buf); model.parse("warm up", table); const query = input.value; selectTable(table); input.value = query; update(); })
  .catch(() => { app.querySelector("#parse")!.innerHTML = `<p class="error" role="alert">Could not load the model.</p>`; });
