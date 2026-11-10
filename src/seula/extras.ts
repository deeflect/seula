// Rule pre-pass for Seula: pulls date phrases, yes/no columns and sort / top-N phrases out of a query before the model parses the rest.
// Usage: const x = extras("overdue last 30 days top 5 by balance", table); then model.parse(x.rest, table) + x.conds.
// Prefers leaving text alone: a wrong consumption is worse than handing the words to the model.

import type { Cond, Table } from "./model.js";

export type Extras = { rest: string; conds: Cond[]; sort?: [col: number, dir: "asc" | "desc"]; limit?: number };

const DAY = 864e5;
const ms = (y: number, m0: number, d = 1) => Date.UTC(y, m0, d) / DAY; // month/day overflow rolls over
const dim = (y: number, m0: number) => ms(y, m0 + 1) - ms(y, m0);
const ymd = (y: number, m0: number, d: number) => (m0 >= 0 && m0 < 12 && d >= 1 && d <= dim(y, m0) ? ms(y, m0, d) : null);
const iso = (n: number) => new Date(n * DAY).toISOString().slice(0, 10);
export const isIso = (v: string) => /^\d{4}-\d{2}-\d{2}$/.test(v);

const M = "(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])";
const mon = (s: string) => "janfebmaraprmayjunjulaugsepoctnovdec".indexOf(s.slice(0, 3).toLowerCase()) / 3;

const FORMS: [RegExp, (m: RegExpMatchArray) => number | null][] = [
  [/^(\d{4})-(\d{2})-(\d{2})(?:[t\s].*)?$/i, (m) => ymd(+m[1], +m[2] - 1, +m[3])],
  [/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/, (m) => ymd(+m[3], +m[1] - 1, +m[2])],
  [new RegExp(`^(${M})\\.?\\s*(\\d{1,2})(?:st|nd|rd|th)?(?:,\\s*|\\s+)(\\d{4})$`, "i"), (m) => ymd(+m[3], mon(m[1]), +m[2])],
  [new RegExp(`^(\\d{1,2})\\s*(${M})\\.?,?\\s*(\\d{4})$`, "i"), (m) => ymd(+m[3], mon(m[2]), +m[1])],
  [new RegExp(`^(${M})\\.?,?\\s+(\\d{4})$`, "i"), (m) => ymd(+m[2], mon(m[1]), 1)],
];

export function dateOf(v: unknown): number | null {
  if (typeof v !== "string") return null;
  const s = v.trim();
  for (const [re, f] of FORMS) {
    const m = s.match(re);
    if (m) return f(m);
  }
  return null;
}

export function dateCols(table: Table): boolean[] {
  return table.header.map((_, j) => {
    const cells = table.rows.map((r) => r[j]).filter((c) => String(c ?? "").trim() !== "");
    return cells.length > 0 && cells.filter((c) => dateOf(c) !== null).length >= 0.8 * cells.length;
  });
}

// Per-table facts, computed once: date columns, and yes/no columns as the cell text for yes (and no, when present).
type Info = { dc: boolean[]; bools: ({ yes: string; no?: string } | null)[]; days: Map<number, Set<string>>[] }; // days: day -> cell texts, per date column
const cache = new WeakMap<Table, Info>();
function info(table: Table): Info {
  let x = cache.get(table);
  if (x) return x;
  const bools = table.header.map((h, j) => {
    const vals = [...new Set(table.rows.map((r) => String(r[j] ?? "").trim()).filter(Boolean))];
    const yes = vals.find((v) => /^(yes|y|true)$/i.test(v));
    if (!yes || vals.length > 2 || !vals.every((v) => /^(yes|y|true|no|n|false)$/i.test(v))) return null;
    // A header that is also a value elsewhere ("Paid" flag next to Status "paid") is the model's call.
    const hw = wordsOf(h).join(" ");
    return table.rows.some((r) => r.some((c, k) => k !== j && String(c).trim().toLowerCase() === hw)) ? null : { yes, no: vals.find((v) => v !== yes) };
  });
  const dc = dateCols(table);
  const days = dc.map((d, j) => {
    const m = new Map<number, Set<string>>();
    if (d) for (const r of table.rows) { const n = dateOf(r[j]); if (n !== null) (m.get(n) ?? m.set(n, new Set()).get(n)!).add(String(r[j]).trim()); }
    return m;
  });
  cache.set(table, (x = { dc, bools, days }));
  return x;
}

export function matchesDate(cell: unknown, op: string, iso: string): boolean {
  const a = dateOf(cell), b = dateOf(iso);
  if (a === null || b === null) return false;
  return ({ "=": a === b, "!=": a !== b, ">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b } as Record<string, boolean>)[op] ?? false;
}

// ---- date phrases ----

type Hit = [op: string, day: number][];
const FULL = `\\d{4}-\\d{2}-\\d{2}|\\d{1,2}/\\d{1,2}/\\d{4}|${M}\\.?\\s*\\d{1,2}(?:st|nd|rd|th)?(?:,\\s*|\\s+)\\d{4}|\\d{1,2}\\s*${M}\\.?,?\\s*\\d{4}`;
const DATE = `${FULL}|${M}(?:\\.?,?\\s+\\d{4})?`;
// "not on 5 May 2020", "date != May 2020": a negated date stays with the model, which has != (a negated range is not an AND).
const NEG = /(?:\b(?:not?|neither|nor|never|(?:is|are|was|were|do|does|did)n'?t|except|exclud\w+|without|but)\b[^,;&]{0,30}|!=?|≠|<>|(?<![\w.])-(?:[\w .]{0,24}:)?)\s*["'“]?$/i;
const BARE_MONTH = new RegExp(`^(?:in\\s+)?(?:${M})\\.?$`, "i"), SPELLED = new RegExp(`(?<![a-z])(?:${M})`, "i");
const END = "(?![\\w/-])";
const U = "(day|week|month|year)s?\\b";
const YEAR = new RegExp(`\\b(since|from|after|before|until|through|in|during)\\s+((?:19|20)\\d{2})${END}`, "i");
const MONTH_ONLY = new RegExp(`^(${M})\\.?,?(?:\\s+(\\d{4}))?$`, "i");

function dateRules(t: Date): [RegExp, (m: RegExpMatchArray) => Hit | null][] {
  const y = t.getFullYear(), m0 = t.getMonth(), td = ms(y, m0, t.getDate());
  const shift = (n: number, u: string) => {
    if (u === "day" || u === "week") return td + n * (u === "week" ? 7 : 1);
    const mm = m0 + n * (u === "year" ? 12 : 1);
    return ms(y, mm, Math.min(t.getDate(), dim(y, mm)));
  };
  // Calendar range k units from the current one; weeks start Monday.
  const cal = (k: number, u: string): [number, number] => {
    if (u === "week") return [td - ((t.getDay() + 6) % 7) + 7 * k, td - ((t.getDay() + 6) % 7) + 7 * k + 6];
    const span = { month: 1, quarter: 3, year: 12 }[u]!, s = (u === "year" ? 0 : Math.floor(m0 / span) * span) + k * span;
    return [ms(y, s), ms(y, s + span) - 1];
  };
  // A month with no year is the latest one not after today's month.
  const range = (s: string): [number, number] | null => {
    const m = s.trim().match(MONTH_ONLY);
    if (m) {
      const mi = mon(m[1]), yy = m[2] ? +m[2] : mi <= m0 ? y : y - 1;
      return [ms(yy, mi), ms(yy, mi + 1) - 1];
    }
    const d = dateOf(s);
    return d === null ? null : [d, d];
  };
  const whole = (r: [number, number] | null): Hit | null => r && (r[0] === r[1] ? [["=", r[0]]] : [[">=", r[0]], ["<=", r[1]]]);
  const kw: Record<string, [string, 0 | 1]> = { since: [">=", 0], after: [">", 1], before: ["<", 0], until: ["<=", 1], through: ["<=", 1] };
  const rules: [string, (m: RegExpMatchArray) => Hit | null][] = [
    [`\\b(?:between|from)\\s+(${DATE})\\s+(?:and|to)\\s+(${DATE})${END}`, (m) => {
      const a = range(m[1]), b = range(m[2]);
      return a && b ? [[">=", a[0]], ["<=", b[1]]] : null;
    }],
    [`\\b(since|from|after|before|until|through|on)\\s+(${DATE})${END}`, (m) => {
      const r = range(m[2]), k = m[1].toLowerCase();
      return k === "on" || k === "from" ? whole(r) : r && [[kw[k][0], r[kw[k][1]]]]; // "orders from 7 March" is that day; "from X to Y" is above
    }],
    [`\\b(?:in\\s+the\\s+)?(?:last|past)\\s+(\\d+)\\s+${U}`, (m) => [[">=", shift(-m[1], m[2].toLowerCase())], ["<=", td]]],
    [`\\bnext\\s+(\\d+)\\s+${U}`, (m) => [[">=", td], ["<=", shift(+m[1], m[2].toLowerCase())]]],
    [`\\b(?:older\\s+than\\s+(\\d+)\\s+${U}|more\\s+than\\s+(\\d+)\\s+${U}\\s+ago\\b)`, (m) => [["<", shift(-(m[1] ?? m[3]), (m[2] ?? m[4]).toLowerCase())]]],
    [`\\b(this|last|previous|next)\\s+(week|month|quarter|year)\\b`, (m) => {
      const r = cal({ this: 0, last: -1, previous: -1, next: 1 }[m[1].toLowerCase()]!, m[2].toLowerCase());
      return [[">=", r[0]], ["<=", r[1]]];
    }],
    [`\\b(today|yesterday|tomorrow)\\b`, (m) => [["=", td + { today: 0, yesterday: -1, tomorrow: 1 }[m[1].toLowerCase()]!]]],
    [`(?<![\\w-])(?<!\\d/(?=\\d{1,2}/))(${FULL})(?![\\w-])(?!/\\d+/)`, (m) => whole(range(m[1]))],
    [`\\bin\\s+(${M}(?:\\.?,?\\s+\\d{4})?)${END}`, (m) => whole(range(m[1]))],
    [`\\b(${M}\\.?,?\\s+\\d{4})${END}`, (m) => whole(range(m[1]))],
    [YEAR.source, (m) => { // a bare year: only next to a named date column (see extras)
      const k = m[1].toLowerCase(), r: [number, number] = [ms(+m[2], 0), ms(+m[2], 12) - 1];
      return kw[k] ? [[kw[k][0], r[kw[k][1]]]] : [[">=", r[0]], ["<=", r[1]]];
    }],
  ];
  return rules.map(([src, f]) => [new RegExp(src, "gi"), f]);
}

const wordsOf = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
const blank = (s: string, a: number, b: number) => s.slice(0, a) + " ".repeat(b - a) + s.slice(b);

// Date column for a phrase at [s, e): a header mentioned within 4 words before / 3 after (consumed too), else the first "date" header.
function pickDateCol(work: string, s: number, e: number, table: Table, dc: boolean[]): { col: number; spans: [number, number][] } {
  const words = [...work.matchAll(/[\p{L}\p{N}]+|[:=]/gu)];
  const win = [...words.filter((w) => w.index! + w[0].length <= s).slice(-4), ...words.filter((w) => w.index! >= e).slice(0, 3)];
  const cands = dc.flatMap((d, j) => (d ? [j] : []));
  for (const j of [...cands].sort((a, b) => wordsOf(table.header[b]).length - wordsOf(table.header[a]).length)) {
    const hw = wordsOf(table.header[j]);
    const hit = hw.map((h) => win.find((w) => w[0].toLowerCase() === h));
    if (!hw.length || !hit.every(Boolean)) continue;
    const spans = hit.map((w) => [w!.index!, w!.index! + w![0].length] as [number, number]);
    const next = words[Math.max(...hit.map((w) => words.indexOf(w!))) + 1];
    if (next && next.index! < s && /^(is|:|=)$/i.test(next[0])) spans.push([next.index!, next.index! + next[0].length]);
    return { col: j, spans };
  }
  return { col: cands.length === 1 ? cands[0] : (cands.find((j) => /date/i.test(table.header[j])) ?? cands[0]), spans: [] };
}

// Drop connectors left dangling at either end or doubled.
function tidy(s: string): string {
  const conn = (p?: string) => p !== undefined && /^(and|&|,)$/i.test(p);
  const out: string[] = [];
  for (const p of s.replace(/,(?!\d)/g, " , ").split(/\s+/).filter(Boolean)) if (!(conn(p) && (!out.length || conn(out.at(-1))))) out.push(p);
  while (conn(out.at(-1))) out.pop();
  return out.join(" ").replace(/ ,(?= |$)/g, ",");
}

export function extras(query: string, table: Table, today = new Date()): Extras {
  const { dc, bools, days } = info(table), conds: Cond[] = [], hws = table.header.map(wordsOf), nd = dc.filter(Boolean).length;
  let work = query, sort: Extras["sort"], limit: number | undefined;
  const left: [number, number][] = [];
  if (dc.some(Boolean))
    for (const [re, f] of dateRules(today))
      for (const m of [...work.matchAll(re)]) {
        const s = m.index!, e = s + m[0].length;
        if (work.slice(s, e) !== m[0] || left.some(([a, b]) => s < b && e > a)) continue;
        // Negated dates go to the model, and so does the rest of a list one of them started ("not 1 may 2012, 2 june 2012").
        const hit = NEG.test(work.slice(0, s)) || left.some(([, b]) => b <= s && /^\s*(?:,|\/|\||or|nor)\s*$/i.test(work.slice(b, s))) ? null : f(m);
        if (!hit) { left.push([s, e]); continue; } // a looser rule must not take a piece of it
        let { col, spans } = pickDateCol(work, s, e, table, dc);
        const day = hit.length === 1 && hit[0][0] === "=" ? hit[0][1] : null;
        const has = day !== null && !spans.length && nd > 1 ? dc.flatMap((d, j) => (d && days[j].has(day) ? [j] : [])) : [col];
        // Left to the model: a time after the date, a non-date column named right before it ("second edition is june 2003"),
        // a day that several date columns hold (or none) with no column named, and "in april" where the column spells its months
        // (any April, "contains april"; with ISO dates it is the latest April).
        const before = " " + wordsOf(work.slice(0, s).replace(/(?:\b(?:is|was|on|in)|[:=])\s*$/i, "")).join(" ");
        if (/^\s+\d{1,2}:\d{2}/.test(work.slice(e)) || has.length !== 1 || (!spans.length && new RegExp(`^${YEAR.source}$`, "i").test(m[0])) || hws.some((h, j) => !dc[j] && h.length && before.endsWith(" " + h.join(" ")))
          || (BARE_MONTH.test(m[0]) && table.rows.some((r) => SPELLED.test(String(r[col]))))) { left.push([s, e]); continue; }
        col = has[0];
        // One exact day written one way in the table: use the table's own text, so the filter reads like the data.
        const cell = day !== null ? days[col].get(day) : undefined;
        conds.push(...hit.map(([op, n]) => [col, op, cell?.size === 1 ? [...cell][0] : iso(n)] as Cond));
        for (const [a, b] of [[s, e], ...spans]) work = blank(work, a, b);
      }

  const toks = [...work.matchAll(/[\p{L}\p{N}]+/gu)].map((m) => ({ w: m[0].toLowerCase(), s: m.index!, e: m.index! + m[0].length }));
  const numOrDate = (j: number) => table.types[j] === "real" || dc[j];
  const reals = table.types.flatMap((t, j) => (t === "real" ? [j] : []));
  const same = (a: string, b: string) => a === b || a + "s" === b || a === b + "s";
  const colAt = (k: number, ok: (j: number) => boolean) => {
    let best: { j: number; len: number } | null = null;
    hws.forEach((hw, j) => {
      if (ok(j) && hw.length > (best?.len ?? 0) && hw.every((h, x) => toks[k + x] && same(toks[k + x].w, h))) best = { j, len: hw.length };
    });
    return best as { j: number; len: number } | null;
  };
  // Trigger + next word inside a header ("Top scorer", "Most recent result") is a header mention, not a trigger.
  const guarded = (i: number) => !!toks[i + 1] && hws.some((hw) => hw.some((h, x) => h === toks[i].w && hw[x + 1] === toks[i + 1].w));
  // The same words inside a cell ("Top 10 Most Popular Artiste", version "latest") are a value, not a request.
  const inCell = (words: string) => table.rows.some((r) => r.some((c) => new RegExp(`(?<![\\p{L}\\p{N}])${words}(?![\\p{L}\\p{N}])`, "iu").test(String(c))));
  const take = (i: number, n: number) => (work = blank(work, toks[i].s, toks[i + n - 1].e));

  // A yes/no column named with no value: "remote" is Remote = yes, "not remote" is Remote = no.
  const bw = hws.map((hw) => (hw.length > 1 && /^(is|has)$/.test(hw[0]!) ? hw.slice(1) : hw));
  for (let i = 0; i < toks.length; i++) {
    const j = bools.findIndex((b, c) => b && bw[c].length && bw[c].every((h, x) => toks[i + x] && same(toks[i + x].w, h)));
    const n = j < 0 ? 0 : bw[j].length;
    if (!n || /^(yes|y|true|no|n|false|is)$/.test(toks[i + n]?.w ?? "")) continue; // "remote: yes" names its value
    const neg = i > 0 && /^(not|non|no)$/.test(toks[i - 1].w), b = bools[j]!;
    conds.push(neg ? (b.no ? [j, "=", b.no] : [j, "!=", b.yes]) : [j, "=", b.yes]);
    work = blank(work, toks[neg ? i - 1 : i].s, toks[i + n - 1].e);
    i += n - 1;
  }

  for (let i = 0; i < toks.length; i++) {
    const w = toks[i].w;
    if (!work.slice(toks[i].s, toks[i].e).trim() || guarded(i)) continue;
    if ((w === "top" || w === "bottom") && limit === undefined && /^\d+$/.test(toks[i + 1]?.w ?? "") && !/\S/.test(work.slice(toks[i].e, toks[i + 1].s)) && !inCell(`${w}\\s+${toks[i + 1].w}`)) {
      let col: number | null = null, n = 2;
      if (toks[i + 2]?.w === "by") {
        const r = colAt(i + 3, numOrDate);
        if (r) (col = r.j), (n = 3 + r.len);
      } else if (reals.length === 1) col = reals[0];
      if (col === null) continue;
      limit = +toks[i + 1].w;
      sort ??= [col, w === "top" ? "desc" : "asc"];
      take(i, n), (i += n - 1);
    } else if (sort !== undefined) continue;
    else if (/^(sort|sorted|order|ordered)$/.test(w) && toks[i + 1]?.w === "by") {
      const r = colAt(i + 2, () => true);
      if (!r) continue;
      let n = 2 + r.len;
      const d = toks[i + n]?.w ?? "";
      const dir = d.startsWith("desc") ? "desc" : "asc";
      if (/^(asc|ascending|desc|descending)$/.test(d)) n++;
      sort = [r.j, dir], take(i, n), (i += n - 1);
    } else if (/^(latest|newest|oldest|earliest)$/.test(w) || (w === "most" && toks[i + 1]?.w === "recent")) {
      if (!dc.some(Boolean) || inCell(w === "most" ? "most\\s+recent" : w)) continue;
      const k = w === "most" ? 2 : 1, { col, spans } = pickDateCol(work, toks[i].s, toks[i + k - 1].e, table, dc);
      const n = k + +(toks[i + k]?.w === "first"); // "newest first"
      sort = [col, /^(oldest|earliest)$/.test(w) ? "asc" : "desc"];
      take(i, n), spans.forEach(([a, b]) => (work = blank(work, a, b))), (i += n - 1);
    } else if (/^(highest|largest|biggest|most|max|lowest|smallest|least|fewest|min)$/.test(w)) {
      const r = colAt(i + 1, numOrDate);
      if (!r) continue;
      const n = 1 + r.len + +(toks[i + 1 + r.len]?.w === "first"); // "highest balance first"
      sort = [r.j, /^(highest|largest|biggest|most|max)$/.test(w) ? "desc" : "asc"], take(i, n), (i += n - 1);
    }
  }
  return { rest: work === query ? query.trim() : tidy(work), conds, ...(sort && { sort }), ...(limit !== undefined && { limit }) };
}
