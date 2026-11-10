// Shared by the landing and the live demo: chips, column marking, row morphs, scroll edge fades.

export const esc = (v: unknown) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
export const icon = (id: string) => `<svg class="i" aria-hidden="true"><use href="/icons.svg#${id}" /></svg>`;
export const sym: Record<string, string> = { "=": "=", "!=": "≠", contains: "contains", ">": ">", ">=": "≥", "<": "<", "<=": "≤", in: "=", "not in": "≠", "contains any": "contains" };
// Chip colour by what the filter does: match a value, compare a number or date, or order the rows.
export const kind: Record<string, string> = { "=": "value", "!=": "value", contains: "value", in: "value", "not in": "value", "contains any": "value", ">": "number", ">=": "number", "<": "number", "<=": "number", sort: "order", limit: "order" };
export const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// A chip part: removal key, label, kind of filter, and the column index it constrains (none for a limit).
export type Part = { key: string; label: string; kind: string; col?: number };

export function chipsHtml(parts: Part[], guess: boolean): string {
  return parts.map((p) => `<span class="chip ${guess ? "guess" : `k-${p.kind}`}"${p.col === undefined ? "" : ` data-col="${p.col}"`}>${esc(p.label)}<button type="button" data-rm="${esc(p.key)}" aria-label="Remove ${esc(p.label)}">${icon("x")}</button></span>`).join("");
}

// Hovering or focusing anything with data-col inside `scope` marks that column of `table()`.
export function linkColumns(scope: HTMLElement, table: () => HTMLTableElement | null): void {
  const set = (col: string | undefined) => {
    const t = table();
    if (!t) return;
    if (col === undefined) { t.removeAttribute("data-mark"); return; }
    t.dataset.mark = col;
    for (const r of t.rows) for (const c of r.cells) c.classList.toggle("mk", c.dataset.c === col);
  };
  const at = (e: Event) => (e.target as HTMLElement).closest<HTMLElement>("[data-col]")?.dataset.col;
  scope.addEventListener("pointerover", (e) => set(at(e)));
  scope.addEventListener("pointerleave", () => set(undefined));
  scope.addEventListener("focusin", (e) => set(at(e)));
  scope.addEventListener("focusout", () => set(undefined));
}

// Re-render rows so the ones that stay slide to their place and the ones that leave fade (view transitions,
// 40 rows or fewer). Rows carry data-r (a stable row id); names are set only for the transition, so they
// never take part in a page-to-page transition.
export function morph(tbody: () => HTMLTableSectionElement | null, update: () => void, animate = true): void {
  const doc = document as Document & { startViewTransition?: (cb: () => void) => { finished: Promise<void> } };
  const name = (on: boolean) => {
    const b = tbody();
    if (!b || b.rows.length > 40) return;
    for (const r of b.rows) r.style.viewTransitionName = on && r.dataset.r ? `r-${r.dataset.r}` : "";
  };
  if (!animate || !doc.startViewTransition || reduced() || (tbody()?.rows.length ?? 0) > 40) { update(); return; }
  name(true);
  const vt = doc.startViewTransition(() => { update(); name(true); });
  vt.finished.finally(() => name(false));
}

// Soft fade on the side of a sideways-scrolling table that has more to show.
export function fadeEdges(el: HTMLElement): void {
  const f = () => {
    const l = el.scrollLeft > 1, r = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
    const v = (l ? "l" : "") + (r ? "r" : "");
    v ? (el.dataset.fade = v) : el.removeAttribute("data-fade");
  };
  if (!el.dataset.faded) { el.dataset.faded = ""; el.addEventListener("scroll", f, { passive: true }); new ResizeObserver(f).observe(el); }
  f();
}
