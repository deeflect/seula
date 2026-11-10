import { featurizeQuery, simpleStem, tokenizeQuery, LEX_NEGATE_SET, MAX_TOKENS } from "./featurize";
import { FieldDef, FilterClause, SchemaDef } from "./schema";

const RULE_SPLIT_RE = /\s+(?:and|&&|&|plus|also|where|with|that\s+have)\s+|\s*[,;]\s*/i;
const RULE_INTRO_RE =
  /^(?:show\s+me|show\s+all|find\s+all\s+records\s+where|please\s+list\s+items\s+with|get\s+entries\s+that\s+have|find\s+rows\s+for|only\s+show)\s+/i;
const RULE_BIN_RE =
  /^\s*(?:(not)\s+)?([a-zA-Z0-9_]+)\s*(!==|!=|==|=|>=|<=|>|<|:|~|\bis\s+not\b|\bis\b|\bequals\b|\bcontains\b|\bbefore\b|\bafter\b|\bover\b|\bunder\b)\s*(?:(not)\s+)?([a-zA-Z0-9_./:@$-]+)\s*$/i;
const RULE_SINGLE_RE = /^\s*(?:(not|no|!)\s+)?([a-zA-Z0-9_]+)\s*$/i;

export function baselineParse(query: string, schema: SchemaDef): FilterClause[] {
  const exactFields = new Map<string, FieldDef>();
  const exactEnums = new Map<string, { field: FieldDef; value: string }>();

  for (const f of schema.fields) {
    exactFields.set(f.name.toLowerCase(), f);
    for (const a of f.aliases) {
      exactFields.set(a.toLowerCase(), f);
    }
    if (f.kind === "ENUM") {
      for (const v of f.enumValues) {
        exactEnums.set(v.toLowerCase(), { field: f, value: v });
      }
    }
  }

  const opMap: Record<string, FilterClause["cmp"]> = {
    "=": "eq",
    "==": "eq",
    ":": "eq",
    is: "eq",
    equals: "eq",
    "!=": "neq",
    "!==": "neq",
    "is not": "neq",
    ">": "gt",
    ">=": "gte",
    over: "gt",
    "<": "lt",
    "<=": "lte",
    under: "lt",
    "~": "contains",
    contains: "contains",
    before: "before",
    after: "after",
  };

  const cleanedQuery = query.trim().replace(RULE_INTRO_RE, "");
  const clauses: FilterClause[] = [];
  const segments = cleanedQuery.split(RULE_SPLIT_RE);

  for (const rawSeg of segments) {
    const seg = rawSeg.trim();
    if (!seg) continue;

    const mBin = RULE_BIN_RE.exec(seg);
    if (mBin) {
      const [, neg1, fTok, rawOp, neg2, vTok] = mBin;
      const fDef = exactFields.get(fTok.toLowerCase());
      if (!fDef) continue;

      const opNorm = rawOp.toLowerCase().trim().replace(/\s+/g, " ");
      let cmpOp: FilterClause["cmp"] = opMap[opNorm] ?? "eq";
      let negated = Boolean(neg1 || neg2);
      const vClean = vTok.trim().replace(/^["']|["']$/g, "").toLowerCase();

      if (fDef.kind === "BOOL") {
        if (cmpOp === "neq") {
          cmpOp = "eq";
          negated = !negated;
        }
        if (["false", "no", "off", "0"].includes(vClean)) {
          negated = !negated;
        }
        clauses.push({ field: fDef.name, cmp: "eq", value: "true", negated });
      } else if (fDef.kind === "ENUM") {
        const matchedV = fDef.enumValues.find((ev) => ev.toLowerCase() === vClean);
        if (matchedV !== undefined) {
          clauses.push({ field: fDef.name, cmp: cmpOp, value: matchedV, negated });
        }
      } else if (fDef.kind === "NUMBER") {
        if (/^-?\d+(?:\.\d+)?$/.test(vClean)) {
          clauses.push({ field: fDef.name, cmp: cmpOp, value: vClean, negated });
        }
      } else {
        clauses.push({ field: fDef.name, cmp: cmpOp, value: vClean, negated });
      }
      continue;
    }

    const mSingle = RULE_SINGLE_RE.exec(seg);
    if (mSingle) {
      const [, negTok, word] = mSingle;
      const negated = Boolean(negTok);
      const wLow = word.toLowerCase();
      const fDef = exactFields.get(wLow);
      if (fDef && fDef.kind === "BOOL") {
        clauses.push({ field: fDef.name, cmp: "eq", value: "true", negated });
      } else {
        const eHit = exactEnums.get(wLow);
        if (eHit) {
          clauses.push({ field: eHit.field.name, cmp: "eq", value: eHit.value, negated });
        }
      }
    }
  }

  return clauses;
}

export const CENTROID_DIM = 96;

export function fnv1a32(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function embedPersonalPhrase(
  query: string,
  schema?: SchemaDef,
  dim: number = CENTROID_DIM,
): number[] {
  const vec = new Array<number>(dim).fill(0);
  const tokens = tokenizeQuery(query);
  if (tokens.length === 0) return vec;

  const accumulate = (featureKey: string, weight: number): void => {
    const h = fnv1a32(featureKey);
    const idx = h % dim;
    const sign = ((h >>> 16) & 1) === 0 ? 1.0 : -1.0;
    vec[idx] += sign * weight;
  };

  const norms = tokens.map((t) => t.norm);
  for (let i = 0; i < norms.length; i++) {
    const norm = norms[i];
    const stem = simpleStem(norm);
    accumulate(`tok:${norm}`, 1.5);
    accumulate(`stem:${stem}`, 1.2);
    if (i + 1 < norms.length) {
      accumulate(`bi:${stem}_${simpleStem(norms[i + 1])}`, 1.4);
    }
    const padded = `^${norm}$`;
    for (let k = 0; k < padded.length - 2; k++) {
      accumulate(`tri:${padded.slice(k, k + 3)}`, 0.5);
    }
  }

  if (schema) {
    const metas = featurizeQuery(query, schema);
    for (const m of metas) {
      for (const fid of m.featureIds.slice(1)) {
        accumulate(`struct:${fid}`, 0.4);
      }
    }
  }

  let normSq = 0;
  for (const v of vec) normSq += v * v;
  if (normSq > 1e-12) {
    const inv = 1.0 / Math.sqrt(normSq);
    for (let i = 0; i < dim; i++) vec[i] *= inv;
  }
  return vec;
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  return dot;
}

export interface CentroidPrototype {
  idiomId: string;
  schemaName: string;
  examples: string[];
  clauses: FilterClause[];
  centroid: number[];
}

export interface CentroidMatch {
  matched: boolean;
  idiomId: string | null;
  similarity: number;
  clauses: FilterClause[];
}

export interface SlangSpanHit {
  idiomId: string;
  phrase: string;
  tokenStart: number;
  tokenEnd: number;
  similarity: number;
  clauses: FilterClause[];
}

export interface SlangSpliceResult {
  matched: boolean;
  mode: "whole" | "subspan" | "none";
  hits: SlangSpanHit[];
  residualQuery: string;
  splicedClauses: FilterClause[];
  bestSimilarity: number;
  bestIdiomId: string | null;
}

const SLANG_STOP_STEMS = new Set([
  "the",
  "and",
  "with",
  "for",
  "where",
  "that",
  "have",
  "has",
  "who",
  "want",
  "wants",
  "client",
  "show",
  "find",
  "all",
  "are",
  "is",
  "not",
  "over",
  "under",
  "from",
]);

export class CentroidMemory {
  readonly dim: number;
  threshold: number;
  private readonly prototypes = new Map<string, CentroidPrototype>();

  constructor(dim: number = CENTROID_DIM, threshold: number = 0.62) {
    this.dim = dim;
    this.threshold = threshold;
  }

  teach(
    idiomId: string,
    schema: SchemaDef,
    examples: readonly string[],
    clauses: readonly FilterClause[],
  ): CentroidPrototype {
    if (examples.length === 0) {
      throw new Error("Provide at least 1 example phrase");
    }
    const avg = new Array<number>(this.dim).fill(0);
    for (const ex of examples) {
      const emb = embedPersonalPhrase(ex, schema, this.dim);
      for (let i = 0; i < this.dim; i++) avg[i] += emb[i];
    }
    let normSq = 0;
    for (const v of avg) normSq += v * v;
    if (normSq > 1e-12) {
      const inv = 1.0 / Math.sqrt(normSq);
      for (let i = 0; i < this.dim; i++) avg[i] *= inv;
    }

    const proto: CentroidPrototype = {
      idiomId,
      schemaName: schema.name,
      examples: [...examples],
      clauses: clauses.map((c) => ({ ...c })),
      centroid: avg,
    };
    this.prototypes.set(`${schema.name}:${idiomId}`, proto);
    return proto;
  }

  query(text: string, schema: SchemaDef): CentroidMatch {
    const tokens = tokenizeQuery(text, MAX_TOKENS + 1);
    if (tokens.length > MAX_TOKENS || tokens.some((t) => LEX_NEGATE_SET.has(t.norm) || t.norm === "or") || /[()]/.test(text)) {
      return { matched: false, idiomId: null, similarity: 0, clauses: [] };
    }
    const qEmb = embedPersonalPhrase(text, schema, this.dim);
    let bestSim = -1;
    let bestProto: CentroidPrototype | null = null;

    for (const proto of this.prototypes.values()) {
      if (proto.schemaName !== schema.name) continue;
      const sim = cosineSimilarity(qEmb, proto.centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestProto = proto;
      }
    }

    if (bestProto && bestSim >= this.threshold) {
      return {
        matched: true,
        idiomId: bestProto.idiomId,
        similarity: Number(bestSim.toFixed(4)),
        clauses: bestProto.clauses.map((c) => ({ ...c })),
      };
    }

    return {
      matched: false,
      idiomId: bestProto?.idiomId ?? null,
      similarity: Number(Math.max(bestSim, 0).toFixed(4)),
      clauses: [],
    };
  }

  /**
   * Compositional Sub-Span Slang Splicer:
   * Finds taught slang idioms embedded INSIDE larger natural-language queries
   * (e.g., "unpaid retainer over $4k with make it pop"), extracts the taught clauses,
   * and returns the `residualQuery` ("unpaid retainer over $4k") for the 21.7 KiB neural tagger.
   */
  splice(text: string, schema: SchemaDef): SlangSpliceResult {
    const protos = this.listPrototypes(schema.name);
    if (protos.length === 0) {
      return {
        matched: false,
        mode: "none",
        hits: [],
        residualQuery: text,
        splicedClauses: [],
        bestSimilarity: 0,
        bestIdiomId: null,
      };
    }

    const tokens = tokenizeQuery(text, MAX_TOKENS + 1);
    if (tokens.length > MAX_TOKENS || tokens.some((t) => t.norm === "or") || /[()]/.test(text)) {
      return { matched: false, mode: "none", hits: [], residualQuery: text, splicedClauses: [], bestSimilarity: 0, bestIdiomId: null };
    }
    if (tokens.length === 0) {
      return {
        matched: false,
        mode: "none",
        hits: [],
        residualQuery: "",
        splicedClauses: [],
        bestSimilarity: 0,
        bestIdiomId: null,
      };
    }

    // 1. Check if the whole query is a direct slang call (<= 4 tokens and high similarity)
    const direct = this.query(text, schema);
    const exactTaughtPhrase = protos.some((p) => p.idiomId === direct.idiomId && p.examples.some((ex) =>
      tokenizeQuery(ex).map((t) => t.norm).join(" ") === tokens.map((t) => t.norm).join(" ")));
    if (direct.matched && exactTaughtPhrase) {
      return {
        matched: true,
        mode: "whole",
        hits: [
          {
            idiomId: direct.idiomId!,
            phrase: text.trim(),
            tokenStart: 0,
            tokenEnd: tokens.length,
            similarity: direct.similarity,
            clauses: direct.clauses,
          },
        ],
        residualQuery: "",
        splicedClauses: direct.clauses,
        bestSimilarity: direct.similarity,
        bestIdiomId: direct.idiomId,
      };
    }

    // Precompute content stems for each prototype
    const protoStems = new Map<string, Set<string>>();
    for (const p of protos) {
      const stems = new Set<string>();
      for (const ex of p.examples) {
        for (const t of tokenizeQuery(ex)) {
          const st = simpleStem(t.norm);
          if (st.length >= 2 && !SLANG_STOP_STEMS.has(st)) {
            stems.add(st);
          }
        }
      }
      protoStems.set(p.idiomId, stems);
    }

    // 2. Scan sliding windows of length 1..5 tokens
    const candidates: SlangSpanHit[] = [];
    let globalBestSim = direct.similarity;
    let globalBestIdiom = direct.idiomId;

    const maxW = Math.min(5, tokens.length);
    for (let w = maxW; w >= 1; w--) {
      for (let start = 0; start + w <= tokens.length; start++) {
        const end = start + w;
        const winTokens = tokens.slice(start, end);
        if (winTokens.some((t) => LEX_NEGATE_SET.has(t.norm)) || (start > 0 && LEX_NEGATE_SET.has(tokens[start - 1].norm))) continue;
        const winText = winTokens.map((t) => t.text).join(" ");
        const winStems = winTokens
          .map((t) => simpleStem(t.norm))
          .filter((st) => st.length >= 2 && !SLANG_STOP_STEMS.has(st));

        if (winStems.length === 0) continue;

        const winEmb = embedPersonalPhrase(winText, schema, this.dim);
        for (const proto of protos) {
          if (proto.examples.some((ex) => tokenizeQuery(ex).map((t) => t.norm).join(" ") === winTokens.map((t) => t.norm).join(" "))) {
            candidates.push({ idiomId: proto.idiomId, phrase: winText, tokenStart: start, tokenEnd: end, similarity: 1, clauses: proto.clauses.map((c) => ({ ...c })) });
            continue;
          }
          if (winTokens.some((t) => schema.fields.some((f) => f.name === t.norm || f.aliases.includes(t.norm) || f.enumValues.includes(t.norm)))) continue;
          const pStems = protoStems.get(proto.idiomId)!;
          const overlapCount = winStems.filter((st) => pStems.has(st)).length;
          if (overlapCount === 0) continue;

          const sim = cosineSimilarity(winEmb, proto.centroid);
          if (sim > globalBestSim) {
            globalBestSim = Number(sim.toFixed(4));
            globalBestIdiom = proto.idiomId;
          }

          const minRequiredSim = w === 1 ? Math.max(this.threshold, 0.68) : Math.max(this.threshold, 0.58);
          if (sim >= minRequiredSim) {
            candidates.push({
              idiomId: proto.idiomId,
              phrase: winText,
              tokenStart: start,
              tokenEnd: end,
              similarity: Number(sim.toFixed(4)),
              clauses: proto.clauses.map((c) => ({ ...c })),
            });
          }
        }
      }
    }

    if (candidates.length === 0) {
      return {
        matched: false,
        mode: "none",
        hits: [],
        residualQuery: text,
        splicedClauses: [],
        bestSimilarity: globalBestSim,
        bestIdiomId: globalBestIdiom,
      };
    }

    // Greedily select highest-similarity non-overlapping spans (preferring tighter/higher sim)
    candidates.sort((a, b) => {
      if (Math.abs(b.similarity - a.similarity) > 0.04) return b.similarity - a.similarity;
      return (b.tokenEnd - b.tokenStart) - (a.tokenEnd - a.tokenStart);
    });

    const usedToken = new Uint8Array(tokens.length);
    const chosen: SlangSpanHit[] = [];
    for (const cand of candidates) {
      let overlaps = false;
      for (let t = cand.tokenStart; t < cand.tokenEnd; t++) {
        if (usedToken[t]) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) {
        chosen.push(cand);
        for (let t = cand.tokenStart; t < cand.tokenEnd; t++) usedToken[t] = 1;
      }
    }

    chosen.sort((a, b) => a.tokenStart - b.tokenStart);
    if (chosen.some((hit) => {
      for (let t = hit.tokenStart - 1; t >= 0; t--) {
        if (["and", "with", "plus", ",", ";"].includes(tokens[t].norm)) break;
        if (LEX_NEGATE_SET.has(tokens[t].norm)) return true;
      }
      return false;
    })) {
      return { matched: false, mode: "none", hits: [], residualQuery: text, splicedClauses: [], bestSimilarity: globalBestSim, bestIdiomId: globalBestIdiom };
    }
    const residualTokens: string[] = [];
    for (let t = 0; t < tokens.length; t++) {
      if (!usedToken[t]) {
        residualTokens.push(tokens[t].text);
      }
    }

    const residualQuery = residualTokens
      .join(" ")
      .replace(/\b(?:with|and|plus|also|who|wants|to|where|that|have)\s*$/i, "")
      .replace(/^\s*(?:with|and|plus|also)\b/i, "")
      .trim();

    const splicedClauses: FilterClause[] = [];
    const seen = new Set<string>();
    for (const h of chosen) {
      for (const c of h.clauses) {
        const key = canonicalizeClause(c);
        if (!seen.has(key)) {
          seen.add(key);
          splicedClauses.push({ ...c });
        }
      }
    }

    return {
      matched: true,
      mode: residualQuery.length === 0 ? "whole" : "subspan",
      hits: chosen,
      residualQuery,
      splicedClauses,
      bestSimilarity: chosen[0]?.similarity ?? globalBestSim,
      bestIdiomId: chosen[0]?.idiomId ?? globalBestIdiom,
    };
  }

  listPrototypes(schemaName?: string): CentroidPrototype[] {
    const all = Array.from(this.prototypes.values());
    return schemaName ? all.filter((p) => p.schemaName === schemaName) : all;
  }
}

export function canonicalizeClause(c: FilterClause): string {
  let cmp = c.cmp;
  let neg = c.negated;
  if (cmp === "neq") {
    cmp = "eq";
    neg = !neg;
  }
  return `${c.field}|${cmp}|${c.value.toLowerCase()}|${neg ? "1" : "0"}`;
}

export function astExactMatch(pred: readonly FilterClause[], target: readonly FilterClause[]): boolean {
  if (pred.length !== target.length) return false;
  for (let i = 0; i < pred.length; i++) {
    if (canonicalizeClause(pred[i]) !== canonicalizeClause(target[i])) return false;
  }
  return true;
}
