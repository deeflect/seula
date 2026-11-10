// Seula inference, ported from bench/seula.py (+ features.py, prepare.py, align.py, filterbench.py).
// Usage: const m = new Seula(weightsBytes); m.parse("price under 5k", table)
// The per-table index is cached on the table object (WeakMap), so keep passing the same object.

import { isIso, matchesDate } from "./extras.js";

const BUCKETS = 4096, DIM = 32, HID = 48, MAX_TOKENS = 48, MAX_NGRAM = 8, NL = 8, MAX_HEADER_WORDS = 6;
const OPS = ["=", "!=", "contains", ">", ">=", "<", "<="];
const CMP = new Set([">", ">=", "<", "<="]);

// valuesOnly: the rows are sample values per column, not real records (server-paged tables), so a filter is checked one condition at a time
export type Table = { header: string[]; types: string[]; rows: (string | number)[][]; valuesOnly?: boolean };
export type Cond = [number, string, string];
export type Parse = { conds: Cond[] | null; conf: number };

// ---- text (prepare.tokenize, features.*, align.edit1, filterbench.parse_number) ----

// Python str-regex classes: \d = Nd, \w = alnum (L, Nd, Nl, No) + "_", \s = str.isspace (differs from JS \s).
const PY_S = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const TOKEN_RE = new RegExp(`\\p{Nd}+(?:[.,]\\p{Nd}+)*|[\\p{L}\\p{Nl}\\p{No}]+|\\p{Nd}+|[^\\p{L}\\p{Nd}\\p{Nl}\\p{No}_${PY_S}]`, "gu");
const NUM_RE = new RegExp(`^(-?\\p{Nd}+(?:\\.\\p{Nd}+)?)[${PY_S}]*(k|m|b|bn|thousand|million|billion)?$`, "u");
const MULT: Record<string, number> = { k: 1e3, thousand: 1e3, m: 1e6, million: 1e6, b: 1e9, bn: 1e9, billion: 1e9 };

export const tokenize = (text: string): string[] => text.toLowerCase().match(TOKEN_RE) ?? [];
const cpLen = (s: string) => (/[\ud800-\udfff]/.test(s) ? [...s].length : s.length);

// Non-ASCII decimal digits -> ASCII (Python float() accepts any Nd). Nd digits come in contiguous runs of 10.
const isNd = (cp: number) => /\p{Nd}/u.test(String.fromCodePoint(cp));
const asciiDigits = (s: string) =>
  s.replace(/[^\x00-\x7f]/gu, (ch) => {
    const cp = ch.codePointAt(0)!;
    if (!isNd(cp)) return ch;
    let z = cp;
    while (isNd(z - 1)) z--;
    return String((cp - z) % 10);
  });

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(s)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h;
}

function hashIds(word: string): number[] {
  const ids = [fnv1a("w:" + word) % BUCKETS];
  const p = [..."<" + word + ">"];
  for (let i = 0; i + 3 <= p.length && ids.length < 16; i++) ids.push(fnv1a(p[i] + p[i + 1] + p[i + 2]) % BUCKETS);
  return ids;
}

function stem(w: string): string {
  const n = cpLen(w);
  if (n > 5 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  for (const [suf, keep] of [["ing", 5], ["ed", 4], ["es", 4], ["s", 3]] as const)
    if (n > keep && w.endsWith(suf) && !w.endsWith("ss")) return w.slice(0, -suf.length);
  return w;
}

const MONTH_FULL = new Map(Object.entries({ jan: "january", feb: "february", mar: "march", apr: "april", jun: "june", jul: "july", aug: "august",
  sep: "september", sept: "september", oct: "october", nov: "november", dec: "december" })); // Map: a plain object would match "constructor"
const ALNUM = /[\p{L}\p{N}]/u;

// features.norm_word: accents stripped, month abbreviation expanded, stemmed; "" for punctuation.
function normWord(w: string): string {
  w = w.normalize("NFD").replace(/\p{M}/gu, "");
  return ALNUM.test(w) ? stem(MONTH_FULL.get(w) ?? w) : "";
}

// Python sorted() on str: by code point, not UTF-16 unit (they differ only for astral vs U+E000-U+FFFF).
const cpCmp = (a: string, b: string) => {
  for (let i = 0, j = 0; i < a.length && j < b.length; ) {
    const x = a.codePointAt(i)!, y = b.codePointAt(j)!;
    if (x !== y) return x - y;
    i += x > 0xffff ? 2 : 1; j += y > 0xffff ? 2 : 1;
  }
  return a.length - b.length;
};
// features.bag pieces of one token (split at commas), then the order-free key of several tokens.
const pieces = (w: string) => w.split(",").map(normWord).filter(Boolean);
const bagKey = (ps: string[]) => ps.sort(cpCmp).join(" ");
const bag = (toks: string[]) => bagKey(toks.flatMap(pieces));

// features.is_number on a token: float(w.replace(",", "")). Tokens are digit runs or letter runs.
function isNumber(w: string): boolean {
  const s = w.replace(/,/g, "");
  return /^\p{Nd}+(?:\.\p{Nd}+)?$/u.test(s) || s === "nan" || s === "inf" || s === "infinity";
}

function edit1(a: string, b: string): boolean {
  if (a === b) return true;
  const x = [...a], y = [...b];
  if (Math.abs(x.length - y.length) > 1) return false;
  if (x.length === y.length) {
    const d: number[] = [];
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) d.push(i);
    return d.length === 1 || (d.length === 2 && d[1] === d[0] + 1 && x[d[0]] === y[d[1]] && x[d[1]] === y[d[0]]);
  }
  const [s, l] = x.length < y.length ? [x, y] : [y, x];
  const ss = s.join("");
  for (let i = 0; i < l.length; i++) if (l.slice(0, i).join("") + l.slice(i + 1).join("") === ss) return true;
  return false;
}

export function parseNumber(s: string): number | null {
  const t = s.trim().toLowerCase().replace(/[,$€£]/g, "").replace(/%+$/, "").trim();
  const m = NUM_RE.exec(t);
  return m ? Number(asciiDigits(m[1])) * (MULT[m[2] ?? ""] ?? 1) : null;
}

// Python "%g". ponytail: JS rounds exact decimal ties half-up, Python half-even (e.g. 1234565 -> 1.23457e+06 vs 1.23456e+06).
export function fmtG(n: number): string {
  if (!Number.isFinite(n)) return Number.isNaN(n) ? "nan" : n > 0 ? "inf" : "-inf";
  if (n === 0) return Object.is(n, -0) ? "-0" : "0";
  const strip = (s: string) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  const [m, e] = n.toExponential(5).split("e");
  const x = Number(e);
  if (x < -4 || x >= 6) return `${strip(m)}e${x < 0 ? "-" : "+"}${String(Math.abs(x)).padStart(2, "0")}`;
  return strip(n.toFixed(5 - x));
}

// Python fmt_num: plain digits for whole numbers ("10000000", never "1e+07"), else %g.
export const fmtNum = (n: number) => (Number.isInteger(n) && Math.abs(n) < 1e15 ? String(n) : fmtG(n));
const LIST = new Set(["or", "nor", "/", "|", ","]);
const MAGNITUDE = new Set(Object.keys(MULT));
const MONTHS = new Set("jan january feb february mar march apr april may jun june jul july aug august sep sept september oct october nov november dec december".split(" "));

// ---- weights ----

type W = Float32Array;
type Lin = { w: W; b: W; out: number; inp: number };

function loadTensors(buf: ArrayBuffer | Uint8Array): Record<string, W> {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const n = new DataView(u8.buffer, u8.byteOffset).getUint32(0, true);
  const header: { name: string; shape: number[]; scale: number; offset: number }[] = JSON.parse(new TextDecoder().decode(u8.subarray(4, 4 + n)));
  const out: Record<string, W> = {};
  for (const h of header) {
    const len = h.shape.reduce((a, b) => a * b, 1);
    const q = new Int8Array(u8.buffer, u8.byteOffset + 4 + n + h.offset, len);
    const s = Math.fround(h.scale), w = new Float32Array(len);
    for (let i = 0; i < len; i++) w[i] = q[i] * s;
    out[h.name] = w;
  }
  return out;
}

function linear(l: Lin, x: ArrayLike<number>, y = new Float64Array(l.out)): Float64Array {
  const w = l.w, n = l.inp, n4 = n - (n % 4);
  for (let i = 0; i < l.out; i++) {
    // 4 independent sums: the CPU overlaps them instead of waiting on one long add chain (~2x)
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, j = 0;
    const o = i * n;
    for (; j < n4; j += 4) {
      s0 += w[o + j] * x[j]; s1 += w[o + j + 1] * x[j + 1];
      s2 += w[o + j + 2] * x[j + 2]; s3 += w[o + j + 3] * x[j + 3];
    }
    for (; j < n; j++) s0 += w[o + j] * x[j];
    y[i] = l.b[i] + (s0 + s1) + (s2 + s3);
  }
  return y;
}

function softmax(x: ArrayLike<number>): number[] {
  let m = -Infinity;
  for (let i = 0; i < x.length; i++) m = Math.max(m, x[i]);
  const e = Array.from(x, (v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

const argmax = (x: number[]) => x.reduce((bi, v, i) => (v > x[bi] ? i : bi), 0);
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

// ---- table index (features.TableIndex + per-table model work) ----

export type TableIndex = {
  header: string[][];
  headerStems: Set<string>[];
  numeric: boolean[];
  rowNumeric: boolean[]; // colNumeric from the rows (how matches() compares), not the declared types
  absmax: number[];
  cells: Map<string, Set<number>>;
  colCells: string[][]; // cells of each column (for snap)
  words: Set<string>[]; // norm_word of the words inside cells of each column
  bags: Map<string, Map<number, Set<string>>>; // bag key -> column -> cells sharing it
  wordsByNorm: Map<string, Set<string>>[]; // per column: norm_word -> cell words (snap_words)
  hwords: string[][]; // header words with a letter/digit, first MAX_HEADER_WORDS
  hn: number[];
  cols: Float64Array[]; // col(cat(bag(header ids), numeric)), C x 2H
};

export class Seula {
  private t: Record<string, W>;
  private L: Record<string, Lin> = {};
  private cache = new WeakMap<Table, TableIndex>();

  constructor(buf: ArrayBuffer | Uint8Array) {
    this.t = loadTensors(buf);
    const lin = (name: string, out: number, inp: number): Lin => ({ w: this.t[name + ".weight"], b: this.t[name + ".bias"], out, inp });
    this.L = {
      inp: lin("inp", HID, DIM + 3 + NL),
      tag: lin("tag", 3, 2 * HID),
      span: lin("span", 2 * HID, 4 * HID),
      op: lin("op", OPS.length, 2 * HID),
      col: lin("col", 2 * HID, DIM + 1),
      mlp1: lin("linkmlp.0", 16, NL + 6),
      mlp2: lin("linkmlp.2", 1, 16),
    };
  }

  private bag(ids: number[]): Float64Array {
    const e = this.t["emb.weight"], v = new Float64Array(DIM);
    for (const id of ids) for (let k = 0; k < DIM; k++) v[k] += e[id * DIM + k];
    for (let k = 0; k < DIM; k++) v[k] /= Math.max(ids.length, 1);
    return v;
  }

  index(table: Table): TableIndex {
    let ix = this.cache.get(table);
    if (ix) return ix;
    const header = table.header.map(tokenize);
    const C = header.length;
    const cells = new Map<string, Set<number>>();
    for (const row of table.rows)
      row.forEach((cell, j) => {
        const key = tokenize(String(cell)).join(" ");
        if (key && key.split(" ").length - 1 < MAX_NGRAM) (cells.get(key) ?? cells.set(key, new Set()).get(key)!).add(j);
      });
    const colCells: string[][] = Array.from({ length: C }, () => []);
    const words: Set<string>[] = Array.from({ length: C }, () => new Set());
    const wordsByNorm: Map<string, Set<string>>[] = Array.from({ length: C }, () => new Map());
    const bags = new Map<string, Map<number, Set<string>>>();
    for (const [k, cs] of cells) {
      const b = bag(k.split(" "));
      for (const j of cs) {
        colCells[j].push(k);
        for (const w of k.split(" ")) {
          const nw = normWord(w);
          words[j].add(nw);
          if (nw) (wordsByNorm[j].get(nw) ?? wordsByNorm[j].set(nw, new Set()).get(nw)!).add(w);
        }
        if (b) { const m = bags.get(b) ?? bags.set(b, new Map()).get(b)!; (m.get(j) ?? m.set(j, new Set()).get(j)!).add(k); }
      }
    }
    const hwords = header.map((hs) => hs.filter(normWord).slice(0, MAX_HEADER_WORDS));
    const numeric = table.types.map((t) => t === "real");
    const absmax = header.map((_, j) => table.rows.reduce((m, r) => Math.max(m, Math.abs(parseNumber(String(r[j])) ?? 0)), 0));
    const cols = header.map((h, j) => {
      const ids = h.flatMap(hashIds).slice(0, 48);
      const x = [...this.bag(ids.length ? ids : [0]), numeric[j] ? 1 : 0];
      return linear(this.L.col, x);
    });
    const rowNumeric = header.map((_, j) => colNumeric(table.rows, j));
    ix = { header, headerStems: header.map((h) => new Set(h.map(stem))), numeric, rowNumeric, absmax, cells, colCells, words, bags, wordsByNorm,
      hwords, hn: hwords.map((h) => Math.max(1, h.length)), cols };
    this.cache.set(table, ix);
    return ix;
  }

  private gru(xs: Float64Array[], dir: "" | "_reverse"): Float64Array[] {
    const wih = this.t["gru.weight_ih_l0" + dir], whh = this.t["gru.weight_hh_l0" + dir];
    const bih = this.t["gru.bias_ih_l0" + dir], bhh = this.t["gru.bias_hh_l0" + dir];
    const gi: Lin = { w: wih, b: bih, out: 3 * HID, inp: HID }, gh: Lin = { w: whh, b: bhh, out: 3 * HID, inp: HID };
    const T = xs.length, out: Float64Array[] = new Array(T);
    let h = new Float64Array(HID);
    const a = new Float64Array(3 * HID), b = new Float64Array(3 * HID);
    for (let s = 0; s < T; s++) {
      const t = dir ? T - 1 - s : s;
      linear(gi, xs[t], a); linear(gh, h, b);
      const nh = new Float64Array(HID);
      for (let k = 0; k < HID; k++) {
        const r = sigmoid(a[k] + b[k]), z = sigmoid(a[HID + k] + b[HID + k]);
        const n = Math.tanh(a[2 * HID + k] + r * b[2 * HID + k]);
        nh[k] = (1 - z) * n + z * h[k];
      }
      out[t] = h = nh;
    }
    return out;
  }

  parse(query: string, table: Table): Parse {
    const tab = this.index(table);
    const toks = tokenize(query).slice(0, MAX_TOKENS);
    const T = toks.length, C = tab.header.length;
    if (!T || !C) return { conds: null, conf: 0 }; // Python crashes on these; no filter is the sane answer

    // link matrix L[t][j][k], flat (features.link_matrix + seula.encode extras)
    const L = new Float64Array(T * C * NL);
    const at = (t: number, j: number, k: number) => (t * C + j) * NL + k;
    const mentioned = new Array(C).fill(false);
    // hw[t][j][k]: token t names word k of header j (seula.encode); span heads turn it into "how much of the header was typed"
    const HW = new Float64Array(T * C * MAX_HEADER_WORDS);
    const near1 = (w: string, h: string) => cpLen(w) >= 3 && ((cpLen(h) > 3 && edit1(w, h)) || (cpLen(w) < cpLen(h) && h.startsWith(w)));
    toks.forEach((w, t) => {
      const s = stem(w), num = isNumber(w), wl = cpLen(w), nw = normWord(w);
      for (let j = 0; j < C; j++) {
        if (tab.header[j].includes(w)) { L[at(t, j, 0)] = 1; mentioned[j] = true; }
        if (tab.headerStems[j].has(s)) L[at(t, j, 1)] = 1;
        if (num && tab.numeric[j]) L[at(t, j, 4)] = 1;
        if (tab.header[j].some((h) => near1(w, h))) L[at(t, j, 6)] = 1;
        if (wl >= 3 && tab.words[j].has(nw)) L[at(t, j, 7)] = 1;
        if (nw) tab.hwords[j].forEach((h, k) => { if (nw === normWord(h) || near1(w, h)) HW[(t * C + j) * MAX_HEADER_WORDS + k] = 1; });
      }
    });
    // features.cell_matches(fuzzy=True): exact cell, or same bag key when both ends are words
    const tp = toks.map(pieces);
    for (let i = 0; i < T; i++)
      for (let n = 1; n <= MAX_NGRAM && i + n <= T; n++) {
        const cs = new Set(tab.cells.get(toks.slice(i, i + n).join(" ")) ?? []);
        if (tp[i].length && tp[i + n - 1].length) for (const j of tab.bags.get(bagKey(tp.slice(i, i + n).flat()))?.keys() ?? []) cs.add(j);
        for (const j of cs) for (let t = i; t < i + n; t++) { L[at(t, j, 2)] = 1; if (n > 1) L[at(t, j, 3)] = 1; }
      }
    for (let t = 0; t < T; t++) for (let j = 0; j < C; j++) L[at(t, j, 5)] = mentioned[j] ? 1 : 0;

    // token encoder: bag + dense -> relu(inp) -> BiGRU
    const xs = toks.map((w, t) => {
      let cellSum = 0;
      for (let j = 0; j < C; j++) cellSum += L[at(t, j, 2)];
      const x = new Float64Array(DIM + 3 + NL);
      x.set(this.bag(hashIds(w)));
      x[DIM] = isNumber(w) ? 1 : 0; x[DIM + 1] = Math.min(1, cellSum / 3); x[DIM + 2] = t / MAX_TOKENS;
      for (let k = 0; k < NL; k++) { let m = -Infinity; for (let j = 0; j < C; j++) m = Math.max(m, L[at(t, j, k)]); x[DIM + 3 + k] = m; }
      const h = linear(this.L.inp, x);
      for (let k = 0; k < HID; k++) h[k] = Math.max(0, h[k]);
      return h;
    });
    const f = this.gru(xs, ""), b = this.gru(xs, "_reverse");
    const out = xs.map((_, t) => { const o = new Float64Array(2 * HID); o.set(f[t]); o.set(b[t], HID); return o; });

    // tags -> spans
    let conf = 1;
    const tags = out.map((o) => { const p = softmax(linear(this.L.tag, o)); conf *= Math.max(...p); return argmax(p); });
    const spans: [number, number][] = [];
    for (let t = 0; t < T; ) {
      if (tags[t] !== 0) { const a = t++; while (t < T && tags[t] === 2) t++; spans.push([a, t]); } else t++;
    }

    // hdr[t][j]: token t names column j (exact, stem, fuzzy)
    const hdr = (t: number, j: number) => Math.max(L[at(t, j, 0)], L[at(t, j, 1)], L[at(t, j, 6)]);
    const near = (from: number, to: number, j: number) => { let m = 0; for (let t = Math.max(0, from); t < Math.min(T, to); t++) m = Math.max(m, hdr(t, j)); return m; };
    // header words of j typed in tokens [from, to): sum over words of max over tokens
    const typed = (from: number, to: number, j: number) => {
      let n = 0;
      for (let k = 0; k < MAX_HEADER_WORDS; k++) {
        let m = 0;
        for (let t = Math.max(0, from); t < Math.min(T, to); t++) m = Math.max(m, HW[(t * C + j) * MAX_HEADER_WORDS + k]);
        n += m;
      }
      return n;
    };

    // span heads
    const head = ([a, z]: [number, number]) => {
      const ends = new Float64Array(4 * HID);
      ends.set(out[a]); ends.set(out[z - 1], 2 * HID);
      const s = linear(this.L.span, ends).map(Math.tanh);
      const colLogits = tab.cols.map((cv, j) => {
        let d = 0;
        for (let k = 0; k < 2 * HID; k++) d += cv[k] * s[k];
        const lk = new Float64Array(NL + 6);
        for (let t = a; t < z; t++) for (let k = 0; k < NL; k++) lk[k] += L[at(t, j, k)];
        for (let k = 0; k < NL; k++) lk[k] /= z - a;
        lk[NL] = (z - a) / 8;
        lk[NL + 1] = near(a - 3, a, j);
        lk[NL + 2] = near(z, z + 2, j);
        const nb = typed(a - 4, a, j), na = typed(z, z + 3, j);
        lk[NL + 3] = nb / tab.hn[j]; lk[NL + 4] = na / tab.hn[j]; lk[NL + 5] = Math.min(nb + na, 3) / 3;
        const hdn = linear(this.L.mlp1, lk).map((v) => Math.max(0, v));
        return d + linear(this.L.mlp2, hdn)[0];
      });
      return { po: softmax(linear(this.L.op, s)), pc: softmax(colLogits) };
    };
    const heads = spans.map(head);
    // Two values with nothing between them, same column and text operator, found together in one cell, are one value the tagger split ("julie plec").
    for (let k = 1; k < spans.length; ) {
      const p = heads[k - 1], q = heads[k], col = argmax(p.pc), op = argmax(p.po), both = ` ${toks.slice(spans[k - 1][0], spans[k][1]).join(" ")} `;
      if (spans[k - 1][1] === spans[k][0] && col === argmax(q.pc) && op === argmax(q.po) && (OPS[op] === "=" || OPS[op] === "contains") && tab.colCells[col].some((c) => ` ${c} `.includes(both))) {
        spans.splice(k - 1, 2, [spans[k - 1][0], spans[k][1]]);
        heads.splice(k - 1, 2, head(spans[k - 1]));
      } else k++;
    }

    // joint greedy columns: highest prob first, one column per cond, except a lower + upper bound may share
    const side = (o: string) => (o === ">" || o === ">=" ? 1 : o === "<" || o === "<=" ? -1 : 0);
    // A number on a numeric column is never "contains"; it only looks like it when the digits sit inside other cells ("10" in "1000").
    heads.forEach((h, k) => {
      if (!tab.numeric[argmax(h.pc)] || parseNumber(toks.slice(spans[k][0], spans[k][1]).join("")) === null) return;
      const C = OPS.indexOf("contains"), sum = h.po.reduce((t, p, o) => t + (o === C ? 0 : p), 0);
      h.po = h.po.map((p, o) => (o === C ? 0 : p / sum));
    });
    const ops = heads.map((h) => OPS[argmax(h.po)]);
    // Lists: "a, b or c" (any of them) and "not a or b" (none of them) put several values on one column.
    // Members joined by an "or" share it, so they vote jointly; a comma alone only allows sharing,
    // and a comma after the "or" starts something new ("a or b, c").
    const grp = spans.map((_, k) => k), forced = new Set<number>();
    for (let k = 1; k < spans.length; k++) {
      const gap = toks.slice(spans[k - 1][1], spans[k][0]);
      if (gap.length && gap.every((w) => LIST.has(w)) && (ops[k] === "!=" ? ops[k - 1] === "!=" : ANY.has(ops[k]) && ANY.has(ops[k - 1]))) {
        if (gap.some((w) => w !== ",")) { grp[k] = grp[k - 1]; forced.add(grp[k]); }
        else if (!forced.has(grp[k - 1])) grp[k] = grp[k - 1];
      }
    }
    for (const g of forced) {
      const mem = grp.flatMap((x, k) => (x === g ? [k] : []));
      const joint = softmax(heads[0].pc.map((_, j) => mem.reduce((s, k) => s + Math.log(Math.max(heads[k].pc[j], 1e-9)), 0)));
      for (const k of mem) heads[k].pc = joint;
    }
    // seula.sig: probabilities as float32 like Python's, ranked at 6 significant digits so near-certain ones tie the same way
    for (const h of heads) { h.pc = h.pc.map(Math.fround); h.po = h.po.map(Math.fround); }
    const sig = (p: number) => Number(p.toPrecision(6));
    const pairs = heads.flatMap((h, k) => h.pc.map((p, j) => [sig(p), k, j]));
    pairs.sort((x, y) => y[0] - x[0] || y[1] - x[1] || y[2] - x[2]); // Python sorted(reverse=True) on (p, k, j)
    const chosen = new Map<number, [number, number]>(), used = new Map<number, number[]>();
    for (const [p, k, j] of pairs) {
      if (chosen.has(k) || p <= 0) continue;
      if (used.has(j) && !used.get(j)!.every((m) => grp[k] === grp[m] || side(ops[k]) * side(ops[m]) === -1)) continue;
      chosen.set(k, [j, p]);
      (used.get(j) ?? used.set(j, []).get(j)!).push(k);
    }

    // conds, then predict_fb value normalization
    const covered = new Set(spans.flatMap(([a, z]) => Array.from({ length: z - a }, (_, i) => a + i)));
    const build = (cols: number[], ops: string[]): Cond[] => spans.map(([a, z], k) => {
      const c = cols[k];
      // "10 million": the tagger often stops at "10" (rare in training data); skip if cells are already in millions
      if (z < T && !spans.some(([s]) => s === z) && MAGNITUDE.has(toks[z]) && parseNumber(toks.slice(a, z).join("")) !== null
        && tab.absmax[c] >= MULT[toks[z]]) z++;
      // "feb 25": the tagger often keeps only the day; snap then finds "february 25"
      if (a > 0 && MONTHS.has(toks[a - 1]) && !covered.has(a - 1) && !tab.numeric[c] && parseNumber(toks.slice(a, z).join("")) !== null) a--;
      const o = ops[k], vt = toks.slice(a, z), text = vt.join(" ");
      let n = parseNumber(text.replace(/ /g, ""));
      if (n === null && CMP.has(o) && vt.length > 1) n = parseNumber(vt[0]); // "10000 gb", "18 yards": a unit the tagger kept after the number
      if (CMP.has(o) || (n !== null && tab.numeric[c])) return [c, o, n !== null ? fmtNum(n) : text];
      return [c, o, o === "=" || o === "!=" ? snap(vt, tab, c) : snapWords(vt, tab, c)];
    });
    const cols = heads.map((h, k) => chosen.get(k)?.[0] ?? argmax(h.pc));
    heads.forEach(({ po, pc }, k) => { conf *= Math.max(...po) * pc[cols[k]]; });
    const conds = build(cols, ops);
    if (!conds.length) return { conds: null, conf: 0 };

    // seula.execution_guided: a parse that returns no rows is wrong; try the next-best single change
    // (one condition's column among its top 3, or its op among the top 2), most probable first; none works -> conf 0
    const hasRows = (cs: Cond[]) => (table.valuesOnly ? cs.every((c) => runConds(table.rows, [c], tab.rowNumeric).length > 0) : runConds(table.rows, cs, tab.rowNumeric).length > 0);
    if (hasRows(conds)) return { conds, conf };
    const top = (p: number[], n: number) => p.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).slice(0, n).map(([, i]) => i);
    const alts: [number, number, number, string][] = [];
    heads.forEach(({ pc }, k) => top(pc, 3).forEach((j) => { if (j !== cols[k]) alts.push([sig(conf * pc[j] / Math.max(pc[cols[k]], 1e-9)), k, j, ops[k]]); }));
    heads.forEach(({ po }, k) => top(po, 2).forEach((o) => { if (OPS[o] !== ops[k]) alts.push([sig(conf * po[o] / Math.max(...po)), k, cols[k], OPS[o]]); }));
    alts.sort((x, y) => y[0] - x[0] || y[1] - x[1] || y[2] - x[2] || (y[3] > x[3] ? 1 : y[3] < x[3] ? -1 : 0)); // Python sorted(reverse=True)
    for (const [c, k, j, o] of alts) {
      const cs = build(cols.map((x, i) => (i === k ? j : x)), ops.map((x, i) => (i === k ? o : x)));
      if (hasRows(cs)) return { conds: cs, conf: c };
    }
    return { conds, conf: 0 };
  }
}

// Same semantics as bench/filterbench.py col_numeric / matches / run.
const num = (v: unknown) => parseNumber(String(v));
const normText = (v: unknown) => tokenize(String(v)).join(" ");
export const colNumeric = (rows: Table["rows"], j: number) => rows.filter((r) => num(r[j]) !== null).length >= 0.8 * rows.length;

export function matches(row: Table["rows"][number], [j, op, v]: Cond, numeric: boolean[]): boolean {
  if (isIso(v) && op !== "contains") return matchesDate(row[j], op, v); // dates from the pre-pass, any cell format
  if ([">", ">=", "<", "<="].includes(op) || ((op === "=" || op === "!=") && numeric[j] && num(v) !== null)) {
    const a = num(row[j]), b = num(v);
    if (a === null || b === null) return false;
    return { "=": a === b, "!=": a !== b, ">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b }[op]!;
  }
  const c = normText(row[j]), x = normText(v);
  if (op === "=") return c === x;
  if (op === "!=") return c !== x;
  return !!x && ` ${c} `.includes(` ${x} `);
}

// Rows passing every condition; several "=" / "contains" on one column mean any of them ("overdue or pending", "stark or wayne").
const ANY = new Set(["=", "contains"]);
export function runConds(rows: Table["rows"], conds: Cond[], numeric: boolean[]): number[] {
  const any = new Map<number, Cond[]>(), all: Cond[] = [];
  for (const c of conds) ANY.has(c[1]) ? (any.get(c[0]) ?? any.set(c[0], []).get(c[0])!).push(c) : all.push(c);
  const groups = [...any.values()];
  return rows.flatMap((r, i) => (all.every((c) => matches(r, c, numeric)) && groups.every((g) => g.some((c) => matches(r, c, numeric))) ? [i] : []));
}

// seula.words: tokens with a letter or digit (punctuation like , . ( ) – dropped).
const words = (toks: string[]) => toks.filter((w) => /[\p{L}\p{N}]/u.test(w));

// seula.snap: closest real cell in col for a typed text value: exact, else a unique cell matching word by word
// (typo/prefix ok), else unique with spaces ignored ("1 st" = "1st"), else the unique cell containing it.
function snap(vt: string[], tab: TableIndex, col: number): string {
  const v = vt.join(" "), cells = tab.colCells[col];
  if (cells.includes(v)) return v;
  const same = tab.bags.get(bag(vt))?.get(col); // other word order, punctuation, accents, plural, "sept"
  if (same?.size === 1) return [...same][0];
  const ok = (a: string, b: string) => a === b || (cpLen(b) > 3 && edit1(a, b)) || (cpLen(a) >= 3 && b.startsWith(a));
  const ct = words(vt), kw = cells.map((k) => words(k.split(" ")));
  const best = cells.filter((_, i) => kw[i].length === ct.length && ct.every((a, n) => ok(a, kw[i][n])));
  if (best.length) return best.length === 1 ? best[0] : v;
  const squash = (toks: string[]) => toks.join("").replace(/[^\p{L}\p{N}]/gu, "");
  const sq = squash(ct);
  for (const hit of [(s: string) => s === sq, (s: string) => cpLen(sq) >= 6 && s.includes(sq)]) {
    const found = cells.filter((_, i) => hit(squash(kw[i])));
    if (found.length) return found.length === 1 ? found[0] : v;
  }
  return v;
}

// seula.snap_words (contains): a typed word found in no cell of col becomes the one cell word it normalizes to.
function snapWords(vt: string[], tab: TableIndex, col: number): string {
  return vt.map((w) => {
    const nw = normWord(w), alt = nw ? tab.wordsByNorm[col].get(nw) : undefined; // w itself is in alt iff it is a cell word
    return alt?.size === 1 ? [...alt][0] : w;
  }).join(" ");
}
