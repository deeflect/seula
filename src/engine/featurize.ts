import { FieldDef, FieldKind, SchemaDef } from "./schema";

export const MAX_TOKENS = 24;
export const FEATURES_PER_TOKEN = 8;
export const NUM_SPARSE_FEATURES = 52;

export const LEX_EQ_SET = new Set(["=", "==", ":", "is", "equals", "equal", "eq", "matching", "as"]);
export const LEX_NEQ_SET = new Set(["!=", "!==", "<>", "isnt", "neq", "except", "excluding", "other_than"]);
export const LEX_GT_SET = new Set([">", ">=", "over", "above", "greater", "more", "exceeding", "gt", "gte", "min", "at_least", "atleast", "exceeds"]);
export const LEX_LT_SET = new Set(["<", "<=", "under", "below", "less", "fewer", "lt", "lte", "max", "at_most", "atmost", "within"]);
export const LEX_CONTAINS_SET = new Set(["~", "contains", "containing", "has", "including", "like", "about", "includes"]);
export const LEX_BEFORE_SET = new Set(["before", "prior", "earlier", "until", "older", "pre"]);
export const LEX_AFTER_SET = new Set(["after", "since", "newer", "later", "post", "starting"]);
export const LEX_AND_SET = new Set(["and", "&", "&&", ",", ";", "plus", "also", "with", "having"]);
export const LEX_NEGATE_SET = new Set(["not", "no", "non", "never", "!", "without", "arent", "wasnt", "werent", "cannot"]);
export const LEX_BOOL_TRUE_SET = new Set(["true", "yes", "on", "enabled", "active"]);
export const LEX_BOOL_FALSE_SET = new Set(["false", "off", "disabled", "inactive"]);
export const LEX_FILLER_SET = new Set([
  "show", "me", "all", "find", "list", "get", "where", "that", "are", "was", "were",
  "the", "for", "please", "only", "records", "items", "entries", "rows", "than",
  "to", "by", "from", "in", "of", "at", "any", "which", "whose", "been", "be", "have", "had", "went",
]);

export const PROTECTED_GRAMMAR_WORDS = new Set<string>([
  ...LEX_EQ_SET,
  ...LEX_NEQ_SET,
  ...LEX_GT_SET,
  ...LEX_LT_SET,
  ...LEX_CONTAINS_SET,
  ...LEX_BEFORE_SET,
  ...LEX_AFTER_SET,
  ...LEX_AND_SET,
  ...LEX_NEGATE_SET,
  ...LEX_BOOL_TRUE_SET,
  ...LEX_BOOL_FALSE_SET,
  ...LEX_FILLER_SET,
]);

const RELATIVE_DATE_WORDS = new Set([
  "today", "yesterday", "tomorrow", "last_week", "last_month", "this_week",
  "this_month", "now", "recent", "q1", "q2", "q3", "q4",
]);

const TOKEN_RE = /!==|!=|>=|<=|==|<>|&&|[=:><~!,;&]|"[^"]+"|'[^']+'|\d{4}-\d{2}-\d{2}|\d{4}\/\d{2}\/\d{2}|[@$/\w.-]+(?:'[\w]+)?/gu;

export interface TokenSpan {
  text: string;
  norm: string;
  start: number;
  end: number;
}

export interface TokenFeatureMeta {
  token: TokenSpan;
  featureIds: number[];
  matchedField: string | null;
  matchedFieldKind: FieldKind | null;
  fieldMatchQuality: string;
  boolNegPrefix: boolean;
  matchedEnumField: string | null;
  matchedEnumValue: string | null;
  enumMatchQuality: string;
  distanceToField: number | null;
}

export function normalizeToken(raw: string): string {
  let s = raw.trim().toLowerCase();
  if (s.length >= 2 && ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))) {
    s = s.slice(1, -1).trim();
  }
  if (s === "isn't") s = "isnt";
  if (s === "aren't") s = "arent";
  if (s === "wasn't") s = "wasnt";
  if (s === "weren't") s = "werent";
  return s;
}

export function tokenizeQuery(query: string, maxTokens: number = MAX_TOKENS): TokenSpan[] {
  const spans: TokenSpan[] = [];
  for (const match of query.matchAll(TOKEN_RE)) {
    const raw = match[0];
    const norm = normalizeToken(raw);
    if (!norm) continue;
    const start = match.index ?? 0;
    spans.push({ text: raw, norm, start, end: start + raw.length });
    if (spans.length >= maxTokens) break;
  }
  return spans;
}

export function simpleStem(word: string): string {
  const w = word.toLowerCase().replace(/-/g, "_");
  if (w.length > 5 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 5 && w.endsWith("ing")) return w.slice(0, -3);
  if (w.length > 4 && w.endsWith("ed")) return w.slice(0, -2);
  if (w.length > 4 && w.endsWith("es")) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss") && !w.endsWith("ms")) return w.slice(0, -1);
  return w;
}

export function damerauLevenshtein(a: string, b: string): number {
  if (a === b) return 0;
  const la = a.length;
  const lb = b.length;
  if (Math.abs(la - lb) > 3) return Math.abs(la - lb);
  const dp: number[][] = Array.from({ length: la + 1 }, () => new Array<number>(lb + 1).fill(0));
  for (let i = 0; i <= la; i++) dp[i][0] = i;
  for (let j = 0; j <= lb; j++) dp[0][j] = j;
  for (let i = 1; i <= la; i++) {
    for (let j = 1; j <= lb; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let best = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        best = Math.min(best, dp[i - 2][j - 2] + 1);
      }
      dp[i][j] = best;
    }
  }
  return dp[la][lb];
}

export function matchSchemaField(
  normToken: string,
  schema: SchemaDef,
): { field: FieldDef | null; quality: string; isBoolNegPrefix: boolean } {
  const t = normToken.replace(/-/g, "_");
  for (const f of schema.fields) {
    if (t === f.name.toLowerCase()) return { field: f, quality: "EXACT", isBoolNegPrefix: false };
  }
  const aliasHits = schema.fields.filter((f) => f.aliases.some((a) => a.toLowerCase() === t));
  if (aliasHits.length === 1) return { field: aliasHits[0], quality: "EXACT", isBoolNegPrefix: false };
  if (aliasHits.length > 1) return { field: null, quality: "NONE", isBoolNegPrefix: false };

  if (PROTECTED_GRAMMAR_WORDS.has(t)) {
    return { field: null, quality: "NONE", isBoolNegPrefix: false };
  }

  for (const prefix of ["un_", "un", "non_", "non"]) {
    if (t.startsWith(prefix) && t.length > prefix.length + 2) {
      const rest = t.slice(prefix.length);
      for (const f of schema.fields) {
        if (f.kind !== "BOOL") continue;
        const candidates = [f.name.toLowerCase(), ...f.aliases.map((a) => a.toLowerCase())];
        for (const cand of candidates) {
          if (rest === cand || simpleStem(rest) === simpleStem(cand) || damerauLevenshtein(rest, cand) <= 1) {
            return { field: f, quality: "BOOL_NEG_PREFIX", isBoolNegPrefix: true };
          }
        }
      }
    }
  }

  const tStem = simpleStem(t);
  for (const f of schema.fields) {
    const candidates = [f.name.toLowerCase(), ...f.aliases.map((a) => a.toLowerCase())];
    for (const cand of candidates) {
      const cStem = simpleStem(cand);
      if (tStem.length >= 3 && tStem === cStem) {
        return { field: f, quality: "STEM_OR_PREFIX", isBoolNegPrefix: false };
      }
      if (t.length >= 4 && (cand.startsWith(t) || t.startsWith(cand))) {
        return { field: f, quality: "STEM_OR_PREFIX", isBoolNegPrefix: false };
      }
    }
  }

  let bestField: FieldDef | null = null;
  let bestDist = 99;
  for (const f of schema.fields) {
    const candidates = [f.name.toLowerCase(), ...f.aliases.map((a) => a.toLowerCase())];
    for (const cand of candidates) {
      if (t.length < 4 || cand.length < 4) continue;
      const dist = damerauLevenshtein(t, cand);
      const maxAllowed = cand.length >= 6 && t.length >= 5 && t[0] === cand[0] ? 2 : 1;
      if (dist <= maxAllowed && dist < bestDist) {
        bestDist = dist;
        bestField = f;
      }
    }
  }
  if (bestField !== null) {
    return { field: bestField, quality: "TYPO", isBoolNegPrefix: false };
  }

  return { field: null, quality: "NONE", isBoolNegPrefix: false };
}

export function matchSchemaEnum(
  normToken: string,
  schema: SchemaDef,
): { field: FieldDef | null; enumValue: string | null; quality: string } {
  const t = normToken.replace(/-/g, "_");
  for (const f of schema.fields) {
    if (f.kind !== "ENUM") continue;
    for (const v of f.enumValues) {
      if (t === v.toLowerCase()) return { field: f, enumValue: v, quality: "EXACT" };
    }
  }

  if (PROTECTED_GRAMMAR_WORDS.has(t)) {
    return { field: null, enumValue: null, quality: "NONE" };
  }

  const tStem = simpleStem(t);
  for (const f of schema.fields) {
    if (f.kind !== "ENUM") continue;
    for (const v of f.enumValues) {
      const vl = v.toLowerCase();
      if (tStem.length >= 3 && tStem === simpleStem(vl)) {
        return { field: f, enumValue: v, quality: "STEM_OR_PREFIX" };
      }
      if (t.length >= 4 && (vl.startsWith(t) || t.startsWith(vl))) {
        return { field: f, enumValue: v, quality: "STEM_OR_PREFIX" };
      }
    }
  }

  let bestF: FieldDef | null = null;
  let bestV: string | null = null;
  let bestDist = 99;
  for (const f of schema.fields) {
    if (f.kind !== "ENUM") continue;
    for (const v of f.enumValues) {
      const vl = v.toLowerCase();
      if (t.length < 4 || vl.length < 4) continue;
      const dist = damerauLevenshtein(t, vl);
      const maxAllowed = vl.length >= 6 && t.length >= 5 && t[0] === vl[0] ? 2 : 1;
      if (dist <= maxAllowed && dist < bestDist) {
        bestDist = dist;
        bestF = f;
        bestV = v;
      }
    }
  }
  if (bestF !== null && bestV !== null) {
    return { field: bestF, enumValue: bestV, quality: "TYPO" };
  }

  return { field: null, enumValue: null, quality: "NONE" };
}

function charShapeBucket(raw: string, norm: string): number {
  if (["=", "==", "!=", "!==", ">", ">=", "<", "<=", ":", "~", "!", "&", "&&", ",", ";", "<>"].includes(raw)) {
    return 8;
  }
  if (
    (raw.startsWith('"') && raw.endsWith('"')) ||
    (raw.startsWith("'") && raw.endsWith("'")) ||
    raw.includes("/") ||
    raw.startsWith("@")
  ) {
    return 9;
  }
  if (/^\d{4}[-/]\d{2}[-/]\d{2}$/.test(norm) || RELATIVE_DATE_WORDS.has(norm) || /^\d+[dhwmy]$/.test(norm)) {
    return 7;
  }
  if (/^-?\d+(\.\d+)?$/.test(norm)) return 5;
  if (/^\$?-?\d+(\.\d+)?(ms|kb|mb|gb|s|m|h|usd|k)?$/.test(norm)) return 6;
  if (/^[a-z]+$/.test(norm)) return 2;
  if (/^[a-z]+([_-][a-z0-9]+)+$/.test(norm)) return 3;
  if (/^[a-z]+\d+[a-z0-9]*$/.test(norm)) return 4;
  return 10;
}

function lengthBucket(norm: string): number {
  const n = norm.length;
  if (n <= 1) return 11;
  if (n <= 3) return 12;
  if (n <= 6) return 13;
  if (n <= 10) return 14;
  return 15;
}

function lexiconBucket(norm: string): number {
  if (LEX_EQ_SET.has(norm)) return 17;
  if (LEX_NEQ_SET.has(norm)) return 18;
  if (LEX_GT_SET.has(norm)) return 19;
  if (LEX_LT_SET.has(norm)) return 20;
  if (LEX_CONTAINS_SET.has(norm)) return 21;
  if (LEX_BEFORE_SET.has(norm)) return 22;
  if (LEX_AFTER_SET.has(norm)) return 23;
  if (LEX_AND_SET.has(norm)) return 24;
  if (LEX_NEGATE_SET.has(norm)) return 25;
  if (LEX_BOOL_TRUE_SET.has(norm)) return 26;
  if (LEX_BOOL_FALSE_SET.has(norm)) return 27;
  if (LEX_FILLER_SET.has(norm)) return 28;
  return 16;
}

function fieldKindBucket(kind: FieldKind | null): number {
  if (kind === "BOOL") return 30;
  if (kind === "ENUM") return 31;
  if (kind === "DATE") return 32;
  if (kind === "NUMBER") return 33;
  if (kind === "TEXT") return 34;
  return 29;
}

function fieldQualityBucket(quality: string): number {
  if (quality === "EXACT") return 36;
  if (quality === "STEM_OR_PREFIX") return 37;
  if (quality === "TYPO") return 38;
  if (quality === "BOOL_NEG_PREFIX") return 39;
  return 35;
}

function enumQualityBucket(quality: string): number {
  if (quality === "EXACT") return 41;
  if (quality === "STEM_OR_PREFIX") return 42;
  if (quality === "TYPO") return 43;
  return 40;
}

function distanceBucket(dist: number | null, hasEnumMatch: boolean): number {
  if (dist === null) return hasEnumMatch ? 50 : 51;
  if (dist === 0) return 44;
  if (dist === 1) return 45;
  if (dist === 2) return 46;
  if (dist >= 3) return 47;
  if (dist === -1) return 48;
  return 49;
}

export function featurizeQuery(
  query: string,
  schema: SchemaDef,
  useWordIdentity: boolean = false,
): TokenFeatureMeta[] {
  const tokens = tokenizeQuery(query);
  if (tokens.length === 0) return [];

  const rawMatches = tokens.map((tok, idx) => {
    if (/^['"]/.test(tok.text)) return { idx, fMatch: { field: null, quality: "NONE", isBoolNegPrefix: false }, eMatch: { field: null, enumValue: null, quality: "NONE" } };
    let fMatch = matchSchemaField(tok.norm, schema);
    const eMatch = matchSchemaEnum(tok.norm, schema);
    if (eMatch.quality === "EXACT" && (fMatch.quality === "TYPO" || fMatch.quality === "STEM_OR_PREFIX")) {
      fMatch = { field: null, quality: "NONE", isBoolNegPrefix: false };
    }
    return { idx, fMatch, eMatch };
  });

  const fieldPositions = rawMatches.filter((m) => m.fMatch.field !== null).map((m) => m.idx);

  return tokens.map((tok, idx) => {
    const { fMatch, eMatch } = rawMatches[idx];
    let nearestDist: number | null = null;
    if (fieldPositions.length > 0) {
      let bestPos = fieldPositions[0];
      let bestAbs = Math.abs(idx - bestPos);
      let bestTie = idx >= bestPos ? 0 : 1;
      for (const p of fieldPositions.slice(1)) {
        const dAbs = Math.abs(idx - p);
        const tie = idx >= p ? 0 : 1;
        if (dAbs < bestAbs || (dAbs === bestAbs && tie < bestTie)) {
          bestPos = p;
          bestAbs = dAbs;
          bestTie = tie;
        }
      }
      nearestDist = idx - bestPos;
    }

    const wordSlot = useWordIdentity ? 1 : 1;
    const featureIds = [
      wordSlot,
      charShapeBucket(tok.text, tok.norm),
      lengthBucket(tok.norm),
      /^['"]/.test(tok.text) ? lexiconBucket("quoted_value") : lexiconBucket(tok.norm),
      fieldKindBucket(fMatch.field?.kind ?? null),
      fieldQualityBucket(fMatch.quality),
      enumQualityBucket(eMatch.quality),
      distanceBucket(nearestDist, eMatch.field !== null),
    ];

    return {
      token: tok,
      featureIds,
      matchedField: fMatch.field?.name ?? null,
      matchedFieldKind: fMatch.field?.kind ?? null,
      fieldMatchQuality: fMatch.quality,
      boolNegPrefix: fMatch.isBoolNegPrefix,
      matchedEnumField: eMatch.field?.name ?? null,
      matchedEnumValue: eMatch.enumValue ?? null,
      enumMatchQuality: eMatch.quality,
      distanceToField: nearestDist,
    };
  });
}
