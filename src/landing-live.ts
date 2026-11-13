// The landing's live parts, loaded after first paint: the hero card on the real model, the unsure card,
// the code card's adapter tabs, and the weights map drawn from the same weights.bin.
import { Seula, wrap, type Result, type Condition } from "./seula/index";
import { toSql } from "./seula/sql";
import { toTanStack } from "./seula/tanstack";
import { sampleInvoices } from "./sample-invoices";
import { esc, expressionLabel, icon, sym, kind, chipsHtml, linkColumns, morph, fadeEdges, reduced, type Part } from "./ui";

const KEYS = ["invoice", "client", "balance", "days_late", "status"] as const;
const HEAD = ["Invoice", "Client", "Balance", "Days late", "Status"];
const rows = sampleInvoices.map((i) => ({ invoice: i.invoice, client: i.client, balance: i.balance_usd, days_late: i.days_late, status: i.status }));
const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
const mobile = matchMedia("(max-width: 719px)");

// Authored glosses for the four examples. Labels are never authored: they come from the live parse, and a
// gloss shows only when the parse has exactly these parts; otherwise the card shows chips.
type Seg = string | { w: string; keys: string[]; left?: boolean; tall?: boolean };
const EXAMPLES: { q: string; segs: Seg[] }[] = [
  { q: "overdue and balance over 1k, top 5 by balance", segs: [{ w: "overdue", keys: ["status ="], left: true }, "and", { w: "balance over 1k", keys: ["balance >"] }, ",", { w: "top 5", keys: ["limit"] }, { w: "by balance", keys: ["sort"], tall: true }] },
  { q: "overdue or paid, top 3 by balance", segs: [{ w: "overdue or paid", keys: ["status in"], left: true, tall: true }, ",", { w: "top 3", keys: ["limit"] }, { w: "by balance", keys: ["sort"], tall: true }] },
  { q: "clients except fieldwork", segs: [{ w: "clients except fieldwork", keys: ["client !="], left: true }] },
  { q: "days late over 5 and balance under 2k", segs: [{ w: "days late over 5", keys: ["days_late >"], left: true }, "and", { w: "balance under 2k", keys: ["balance <"] }] },
];

// No top-level await: the build targets browsers without it.
void (async () => {
const bytes = new Uint8Array(await (await fetch(new URL("./seula/weights.bin", import.meta.url))).arrayBuffer());
const seula = wrap(new Seula(bytes.slice().buffer));
const invoices = seula.table(rows, { concepts: { "at risk": [{ column: "days_late", type: "number", op: ">", value: 10 }, { column: "balance", type: "number", op: ">", value: 1000 }] } });

// One part per column and operator, labelled the way the demo labels chips.
function partsOf(r: Result): Part[] {
  const parts: Part[] = (r.conditions ?? []).map((c: Condition) => {
    const j = KEYS.indexOf(c.column as (typeof KEYS)[number]), vs = Array.isArray(c.value) ? c.value : [c.value];
    return { key: `${c.column} ${c.op}`, col: j, kind: kind[c.op] ?? "value", label: `${HEAD[j]} ${sym[c.op] ?? c.op} ${vs.join(c.op === "not in" ? ", " : " or ")}` };
  });
  if (r.expression) parts.push({ key: "expression", kind: "value", label: expressionLabel(r.expression, (c: Condition) => `${c.column} ${sym[c.op] ?? c.op} ${Array.isArray(c.value) ? c.value.join(", ") : c.value}`) });
  if (r.aggregate) parts.push({ key: "aggregate", kind: "order", label: r.aggregate.op === "count" ? "Count matching rows" : `Sum ${r.aggregate.column}` });
  if (r.sort) parts.push({ key: "sort", col: KEYS.indexOf(r.sort.column as (typeof KEYS)[number]), kind: "order", label: `Sort: ${HEAD[KEYS.indexOf(r.sort.column as (typeof KEYS)[number])]} ${r.sort.dir === "asc" ? "↑" : "↓"}` });
  if (r.limit !== undefined) parts.push({ key: "limit", kind: "order", label: `First ${r.limit}` });
  return parts;
}
const kept = (r: Result, removed: Set<string>): Result => ({
  ...r,
  conditions: (r.conditions ?? []).filter((c) => !removed.has(`${c.column} ${c.op}`)),
  expression: removed.has("expression") ? undefined : r.expression,
  aggregate: removed.has("aggregate") ? undefined : r.aggregate,
  sort: removed.has("sort") ? undefined : r.sort,
  limit: removed.has("limit") ? undefined : r.limit,
});

/* ---------- Hero ---------- */
const hero = $("#hero"), qbox = hero.querySelector<HTMLElement>(".qbox")!, gloss = qbox.querySelector<HTMLElement>(".q")!;
const input = $<HTMLInputElement>("#hq"), chipsEl = $("#hchips"), table = $<HTMLTableElement>("#htable"), facts = $("#hfacts"), exEl = $("#hex");
const tbody = () => table.tBodies[0];
let seen = "", removed = new Set<string>(), forced = false, shown = 0, current: Result | null = null;
invoices.parse(EXAMPLES[0].q); // warm up, so the first time shown is the steady one

function glossHtml(ex: (typeof EXAMPLES)[number], parts: Map<string, Part>): string {
  let i = 0;
  const g = ex.segs.map((s, n) => {
    if (typeof s === "string") return `<span class="j${s === "," ? " comma" : ""}">${esc(s)}</span>`;
    const ps = s.keys.map((k) => parts.get(k)!), p = ps[0], last = n === ex.segs.length - 1;
    return `<span class="g${s.left ? " left" : ""}"${p.col === undefined ? "" : ` data-col="${p.col}"`} style="--c: var(--gloss-${p.kind}); --i: ${i++}"><span class="w${last ? " caret" : ""}">${esc(s.w)}</span><span class="br" aria-hidden="true"></span><span class="st${s.tall ? " tall" : ""}" aria-hidden="true"></span><span class="lb"><span>${esc(ps.map((x) => x.label).join(", "))}</span></span></span>`;
  });
  return `<span class="slot" aria-hidden="true">${icon("funnel-simple")}</span>${g.join("")}`;
}

function render(): void {
  const q = input.value.trim();
  if (seen !== q) { seen = q; removed = new Set(); forced = false; }
  const t0 = performance.now();
  const r = q ? invoices.parse(q) : null;
  const ms = performance.now() - t0;
  current = r;
  const guess = r?.status === "ambiguous" && !forced;
  const parts = r ? partsOf(r) : [];
  const exIdx = EXAMPLES.findIndex((e) => e.q === q);
  const ex = EXAMPLES[exIdx];
  const map = new Map(parts.map((p) => [p.key, p]));
  const fits = !!ex && !guess && removed.size === 0 && parts.length === new Set(ex.segs.flatMap((s) => (typeof s === "string" ? [] : s.keys))).size && ex.segs.every((s) => typeof s === "string" || s.keys.every((k) => map.has(k)));

  // Gloss when the parse is exactly the authored one and the input is not being edited; chips otherwise.
  qbox.classList.toggle("edit", qbox.classList.contains("edit") || !fits);
  if (fits && shown !== exIdx) { gloss.classList.remove("draw"); gloss.innerHTML = glossHtml(ex, map); shown = exIdx; }
  const note = q && r && r.status !== "ok" ? `<span class="note">${esc(r.message ?? r.status)}</span>` : "";
  const chips = chipsHtml(parts.filter((p) => !removed.has(p.key)), guess);
  chipsEl.innerHTML = `${chips}${chips && guess && (r?.conditions?.length || r?.expression) ? `<button type="button" class="btn btn-secondary" data-force>Confirm guess</button>` : ""}${note}`;
  chipsEl.hidden = !chips && !note;
  for (const b of exEl.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.textContent === q));

  const usable = !!r && (r.status === "ok" || r.status === "ambiguous" && forced);
  const res = usable && r ? kept(r, removed) : null;
  const hits = res ? invoices.apply({ ...res, confirmAmbiguous: forced }) : r ? [] : rows;
  const aggregate = res?.aggregate ? invoices.aggregate({ ...res, confirmAmbiguous: forced }) : undefined;
  const sortCol = res?.sort ? KEYS.indexOf(res.sort.column as (typeof KEYS)[number]) : -1;
  const th = ["w88 d-col", "", "num w80", "num w72 d-col", "w72"], td = ["id d-col", "", "num", "num d-col", ""];
  table.tHead!.rows[0].innerHTML = HEAD.map((h, j) => `<th class="${th[j]}${j === sortCol ? " on" : ""}" data-c="${j}">${j === sortCol ? (res?.sort?.dir === "asc" ? "↑ " : "↓ ") : ""}${h}</th>`).join("");
  tbody().innerHTML = hits.map((row) => `<tr data-r="${esc(row.invoice)}">${KEYS.map((k, j) => `<td class="${td[j]}" data-c="${j}">${esc(row[k])}</td>`).join("")}</tr>`).join("") || `<tr><td colspan="5" class="empty">${res ? "No rows match." : "No rows previewed."}</td></tr>`;
  facts.innerHTML = `<span class="fact">${icon("rows")}${res || !r ? `${hits.length} of ${rows.length} rows` : "No rows previewed"}</span>${aggregate !== undefined ? `<span class="fact">${icon("gauge")}${esc(res!.aggregate!.op === "count" ? `Count: ${aggregate}` : `Sum: ${aggregate}`)}</span>` : ""}${r ? `<span class="fact">${icon("lightning")}${ms.toFixed(2)} ms</span><span class="fact">${icon("gauge")}${r.confidence.toFixed(2)}</span>` : ""}`;
  fadeEdges(table.parentElement!);
  renderUse();
}

linkColumns(hero, () => table);
qbox.querySelector(".qrow")!.removeAttribute("hidden");
exEl.hidden = false;
gloss.addEventListener("click", () => input.focus());
input.addEventListener("focus", () => qbox.classList.add("edit"));
input.addEventListener("input", render);
hero.addEventListener("focusout", (e) => { if (!hero.contains(e.relatedTarget as Node)) { qbox.classList.remove("edit"); render(); } });
hero.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-rm], button[data-force]");
  if (!b) return;
  if (b.dataset.rm) removed.add(b.dataset.rm); else forced = true;
  morph(tbody, render);
  input.focus();
});
exEl.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest("button");
  if (!b) return;
  input.value = b.textContent!;
  qbox.classList.remove("edit");
  morph(tbody, render);
});

/* ---------- When it is unsure ---------- */
{
  const r = invoices.parse("northwind"), apply = $<HTMLButtonElement>("#uapply"), label = $("#ulabel"), uf = $("#ufacts");
  const text = partsOf(r).map((p) => p.label).join(", ");
  const candidate = !!(r.conditions?.length || r.expression) && r.status === "ambiguous";
  const show = () => {
    const on = apply.getAttribute("aria-pressed") === "true";
    const n = on && candidate ? invoices.apply({ ...r, confirmAmbiguous: true }).length : rows.length;
    label.textContent = candidate ? on ? text : `${text}?` : r.message ?? r.status;
    uf.innerHTML = `<span class="fact">${icon("rows")}${n} of ${rows.length} rows</span><span class="fact">${icon("gauge")}${r.confidence.toFixed(2)}</span>`;
  };
  apply.disabled = !candidate;
  apply.addEventListener("click", () => { apply.setAttribute("aria-pressed", String(apply.getAttribute("aria-pressed") !== "true")); show(); });
  show();
  const calib = $("#calib");
  for (const t of calib.querySelectorAll<HTMLElement>(".tile[data-m]")) {
    t.addEventListener("pointerenter", () => (calib.dataset.mark = t.dataset.m));
    t.addEventListener("pointerleave", () => calib.removeAttribute("data-mark"));
  }
}

/* ---------- Use it: adapter tabs on the hero's live parse ---------- */
// JS literal printer: a value stays on one line when it fits `width` columns, else one entry per line.
function lit(v: unknown, ind: number, col: number, width: number): string {
  const flat = inline(v);
  if (col + flat.length + 1 <= width || v === null || typeof v !== "object") return flat;
  const pad = " ".repeat(ind + 2);
  if (Array.isArray(v)) return `[\n${v.map((x) => pad + lit(x, ind + 2, ind + 2, width)).join(",\n")}\n${" ".repeat(ind)}]`;
  return `{\n${Object.entries(v).map(([k, x]) => `${pad}${k}: ${lit(x, ind + 2, ind + 4 + k.length, width)}`).join(",\n")}\n${" ".repeat(ind)}}`;
}
function inline(v: unknown): string {
  if (typeof v === "string") return v.includes('"') ? `'${v}'` : `"${v}"`;
  if (Array.isArray(v)) return `[${v.map(inline).join(", ")}]`;
  if (v && typeof v === "object") { const e = Object.entries(v).filter(([, x]) => x !== undefined); return e.length ? `{ ${e.map(([k, x]) => `${k}: ${inline(x)}`).join(", ")} }` : "{}"; }
  return String(v);
}
const k = (s: string) => `<span class="k">${s}</span>`, c = (s: string) => `<span class="c">${esc(s)}</span>`;
const str = (q: string, m: boolean) => {
  if (!m || q.length < 34) return [`"${esc(q)}"`];
  const out: string[] = []; let line = "";
  for (const w of q.split(" ")) { if (line && line.length + w.length > 30) { out.push(line); line = ""; } line += `${w} `; }
  out.push(line.trimEnd());
  return out.map((l, i) => `"${esc(l)}"${i < out.length - 1 ? " +" : ""}`);
};
const server = seula.schema({ status: ["open", "shipped", "refunded"], country: ["Germany", "France", "Japan"], total: [0, 25000], placed: ["2024-01-03", "2026-11-09"] });
const serverQ = "shipped to germany, total over 500", serverR = server.parse(serverQ);
const round = (r: Result) => ({ ...r, confidence: +r.confidence.toFixed(2) });

const panel = $("#use-panel"), codeEl = $("#use-code"), outEl = $("#use-out"), copy = panel.querySelector<HTMLButtonElement>(".code-copy")!;
const tabs = [...document.querySelectorAll<HTMLButtonElement>(".use-tabs [role=tab]")];
let tab = "api";

function renderUse(): void {
  const m = mobile.matches, room = m ? 40 : 76, q = input.value.trim();
  const r = current ? kept(current, removed) : null;
  const usable = !!r && (r.status === "ok" || r.status === "ambiguous" && forced);
  const n = usable && r ? invoices.apply({ ...r, confirmAmbiguous: forced }).length : r ? 0 : rows.length;
  let code: string, out: unknown;
  if (tab === "api") {
    const passes = usable ? n === 1 ? "the 1 row that passes" : `the ${n} rows that pass` : "review the status before applying";
    code = `${k("import")} { load } ${k("from")} "seula";\n\n${k("const")} seula = ${k("await")} load();\n${k("const")} invoices = seula.table(rows);\n\n${k("const")} result = invoices.parse(\n  ${str(q, m).join("\n  ")}\n);\n${k("if")} (result.status === "ok") invoices.apply(result); ${c(`// ${passes}`)}`;
    out = r ? round(r) : null;
  } else if (tab === "tanstack") {
    code = `${k("import")} { toTanStack, seulaFilterFn } ${k("from")} "seula/tanstack";\n\n${c("// in your column definitions: filterFn: seulaFilterFn")}\n${k("if")} (result.status === "ok") {\n  ${k("const")} { columnFilters, sorting, aggregate } = toTanStack(result);\n  table.setColumnFilters(columnFilters);\n}`;
    try { out = usable && r ? toTanStack(r, { confirmAmbiguous: forced }) : r ? { status: r.status, message: r.message } : null; } catch (e) { out = { status: "error", message: e instanceof Error ? e.message : String(e) }; }
  } else if (tab === "sql") {
    code = `${k("import")} { toSql } ${k("from")} "seula/sql";\n\n${k("if")} (result.status === "ok") {\n  ${k("const")} { select, where, params } = toSql(result);\n  db.prepare(\`SELECT \${select} FROM invoices WHERE \${where}\`).all(...params);\n}`;
    try { out = usable && r ? toSql(r, { confirmAmbiguous: forced }) : r ? { status: r.status, message: r.message } : null; } catch (e) { out = { status: "error", message: e instanceof Error ? e.message : String(e) }; }
  } else {
    code = `${k("const")} orders = seula.schema({\n  status: ["open", "shipped", "refunded"],\n  country: ["Germany", "France", "Japan"],\n  total: [0, 25000],\n  placed: ["2024-01-03", "2026-11-09"],\n});\n\n${k("const")} result = orders.parse(\n  ${str(serverQ, m).join("\n  ")}\n);`;
    out = round(serverR);
  }
  codeEl.innerHTML = `<pre class="code">${code}</pre>`;
  outEl.innerHTML = `<pre>${esc(lit(out, 0, 0, room))}</pre>`;
  copy.dataset.copy = codeEl.querySelector(m ? "pre:not(.d-only)" : "pre:not(.m-only)")!.textContent!;
}

const select = (t: HTMLButtonElement, focus: boolean) => {
  for (const x of tabs) { const on = x === t; x.setAttribute("aria-selected", String(on)); x.tabIndex = on ? 0 : -1; }
  panel.setAttribute("aria-labelledby", t.id);
  tab = t.dataset.t!;
  if (focus) t.focus();
  renderUse();
};
for (const t of tabs) {
  t.addEventListener("click", () => select(t, false));
  t.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(t), d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length], true); }
    else if (e.key === "Home" || e.key === "End") { e.preventDefault(); select(tabs[e.key === "Home" ? 0 : tabs.length - 1], true); }
  });
}
mobile.addEventListener("change", renderUse);

render();

/* ---------- The whole model: weights.bin drawn byte by byte ---------- */
{
  const W = 432, N = bytes.length, fmt = (n: number) => n.toLocaleString("en-US");
  const hlen = new DataView(bytes.buffer).getUint32(0, true), start = 4 + hlen;
  const header: { name: string; shape: number[]; offset: number }[] = JSON.parse(new TextDecoder().decode(bytes.subarray(4, start)));
  type Span = { name: string; a: number; b: number; shape?: number[]; group: number };
  // Groups follow model.ts: token encoder = bag + dense -> relu(inp) -> BiGRU; then the tagger, span heads, operator head, column match.
  const GROUPS: [string, RegExp][] = [["File header", /^$/], ["Word embeddings", /^emb\./], ["Token encoder", /^(inp|gru)\./], ["Token tagger", /^tag\./], ["Span head", /^span\./], ["Operator head", /^op\./], ["Column match", /^(col|linkmlp)\./]];
  const tensors: Span[] = [{ name: "header", a: 0, b: start, shape: [hlen + 4], group: 0 }, ...header.map((t) => {
    const a = start + t.offset, b = a + t.shape.reduce((x, y) => x * y, 1);
    return { name: t.name, a, b, shape: t.shape, group: GROUPS.findIndex(([, re], g) => g > 0 && re.test(t.name)) };
  })];
  const groups: Span[] = GROUPS.map(([name], g) => {
    const ts = tensors.filter((t) => t.group === g);
    return { name, a: Math.min(...ts.map((t) => t.a)), b: Math.max(...ts.map((t) => t.b)), group: g };
  });
  console.assert(groups.reduce((s, g) => s + g.b - g.a, 0) === N && tensors.every((t) => t.group >= 0), "weights map: groups must cover the file");

  const box = $("#wmap .wbox"), base = document.createElement("canvas"), mask = document.createElement("canvas");
  base.width = base.height = W;
  mask.className = "mask"; mask.setAttribute("aria-hidden", "true");
  base.setAttribute("role", "img"); base.setAttribute("aria-label", "The model's 186,586 bytes drawn as pixels");
  // Same mapping as seula-weights.png: one byte per pixel, row by row, darker for a larger |int8|.
  const img = new ImageData(W, W), s8 = new Int8Array(bytes.buffer);
  for (let i = 0; i < W * W; i++) {
    const g = i < N ? Math.round(255 - 235 * Math.pow(Math.abs(s8[i]) / 127, 0.6)) : 255;
    img.data.set([g, g, g, 255], i * 4);
  }
  base.getContext("2d")!.putImageData(img, 0, 0);
  const mx = mask.getContext("2d")!;
  box.replaceChildren(base, mask);

  const strip = $("#wmap .strip"), name = $("#wname"), wfacts = $("#wfacts");
  strip.innerHTML = groups.map((g, i) => `<button type="button" aria-pressed="false" aria-label="${esc(g.name)}, ${fmt(g.b - g.a)} B" style="flex: ${g.b - g.a} 1 0" tabindex="${i ? -1 : 0}" data-g="${i}"></button>`).join("");
  strip.hidden = false;
  const segs = [...strip.querySelectorAll<HTMLButtonElement>("button")];
  const fact = (ic: string, v: string) => `<span class="fact">${icon(ic)}${esc(v)}</span>`;

  type Sel = { span: Span; at?: number; kind: "t" | "g" };
  let pinned: Sel | null = null, last: Sel | null = null;
  // The mask is drawn at device pixels so the pinned part's outline is a crisp 1px line.
  const path = (t: Span, k: number) => {
    const xa = t.a % W, ya = Math.floor(t.a / W), e = ((t.b - 1) % W) + 1, yb = Math.floor((t.b - 1) / W);
    mx.beginPath();
    if (ya === yb) { mx.rect(xa * k, ya * k, (e - xa) * k, k); return; }
    [[xa, ya], [W, ya], [W, yb], [e, yb], [e, yb + 1], [0, yb + 1], [0, ya + 1], [xa, ya + 1]].forEach(([x, y], i) => (i ? mx.lineTo(x * k, y * k) : mx.moveTo(x * k, y * k)));
    mx.closePath();
  };
  const draw = (sel: Sel) => {
    const px = Math.max(1, Math.round(box.clientWidth * devicePixelRatio));
    if (mask.width !== px) mask.width = mask.height = px;
    const k = px / W;
    mx.globalCompositeOperation = "source-over"; mx.clearRect(0, 0, px, px);
    mx.fillStyle = "rgba(16, 16, 16, 0.72)"; mx.fillRect(0, 0, px, px);
    path(sel.span, k); mx.globalCompositeOperation = "destination-out"; mx.fill(); mx.globalCompositeOperation = "source-over";
    if (pinned && pinned.span === sel.span) { mx.strokeStyle = "#ededed"; mx.lineWidth = devicePixelRatio; mx.stroke(); }
  };
  const show = (sel: Sel | null) => {
    last = sel;
    mask.classList.toggle("on", !!sel);
    for (const b of segs) b.classList.toggle("hit", !!sel && +b.dataset.g! === sel.span.group);
    for (const b of segs) b.setAttribute("aria-pressed", String(!!pinned && pinned.kind === "g" && +b.dataset.g! === pinned.span.group));
    if (!sel) { name.textContent = "weights.bin"; wfacts.innerHTML = fact("file", `${fmt(N)} B`); return; }
    const { span: t, at } = sel;
    draw(sel);
    name.textContent = GROUPS[t.group][0];
    wfacts.innerHTML = (sel.kind === "t" && t.group > 0 ? fact("table", t.shape!.join("×")) : "") + fact("file", `${fmt(t.b - t.a)} B`)
      + (at === undefined ? "" : fact("arrows-horizontal", `@${fmt(at)}`) + fact("target", String(at < start ? bytes[at] : s8[at])));
  };
  new ResizeObserver(() => last && draw(last)).observe(box);
  // Offset and value show only while the pointer is over the map.
  const rest = () => show(pinned && { ...pinned, at: undefined });
  const at = (e: PointerEvent) => {
    const r = base.getBoundingClientRect(), x = Math.floor(((e.clientX - r.left) / r.width) * W), y = Math.floor(((e.clientY - r.top) / r.height) * W);
    const i = Math.min(W - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x));
    return i < N ? i : -1;
  };
  const tensorAt = (i: number) => tensors.find((t) => i >= t.a && i < t.b)!;
  base.addEventListener("pointermove", (e) => { const i = at(e); i < 0 ? rest() : show({ span: tensorAt(i), at: i, kind: "t" }); });
  base.addEventListener("pointerleave", rest);
  base.addEventListener("click", (e) => {
    const i = at(e);
    if (i < 0) { pinned = null; rest(); return; }
    const t = tensorAt(i);
    pinned = pinned?.kind === "t" && pinned.span === t ? null : { span: t, at: i, kind: "t" };
    show(pinned ?? { span: t, at: i, kind: "t" });
  });
  // One quiet sweep through the parts the first time the map is in view, unless the visitor got there first.
  let touched = false;
  for (const el of [box, strip]) for (const ev of ["pointerenter", "pointerdown", "focusin"]) el.addEventListener(ev, () => (touched = true));
  const step = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--dur-2")) || 220;
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    if (touched || reduced()) return;
    groups.forEach((g, i) => setTimeout(() => { if (!touched) show({ span: g, kind: "g" }); }, i * step));
    setTimeout(() => { if (!touched) rest(); }, groups.length * step);
  }, { threshold: 0.6 });
  io.observe(box);
  strip.addEventListener("pointerover", (e) => { const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button"); if (b) show({ span: groups[+b.dataset.g!], kind: "g" }); });
  strip.addEventListener("pointerleave", rest);
  strip.addEventListener("focusin", (e) => show({ span: groups[+(e.target as HTMLElement).dataset.g!], kind: "g" }));
  strip.addEventListener("focusout", rest);
  strip.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button");
    if (!b) return;
    const g = groups[+b.dataset.g!];
    pinned = pinned?.kind === "g" && pinned.span === g ? null : { span: g, kind: "g" };
    show(pinned ?? { span: g, kind: "g" });
  });
  strip.addEventListener("keydown", (e) => {
    const i = segs.indexOf(e.target as HTMLButtonElement), d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (i < 0 || !d) return;
    e.preventDefault();
    const n = segs[(i + d + segs.length) % segs.length];
    for (const b of segs) b.tabIndex = b === n ? 0 : -1;
    n.focus();
  });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && pinned) { pinned = null; rest(); } });
}
})();
