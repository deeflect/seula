from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Sequence

from spike.schema import FieldDef, SchemaDef

MAX_TOKENS = 24
FEATURES_PER_TOKEN = 8
NUM_SPARSE_FEATURES = 52

# Feature group offsets (total = 52 sparse feature rows)
# 0..1   (2): word identity folded slot (0=pad, 1=active token when use_word_identity=False)
# 2..10  (9): character shape bucket
# 11..15 (5): length bucket
# 16..28 (13): operator & structural lexicon bucket
# 29..34 (6): schema field match kind (NONE, BOOL, ENUM, DATE, NUMBER, TEXT)
# 35..39 (5): schema field match quality (NONE, EXACT, STEM_OR_PREFIX, TYPO, BOOL_NEG_PREFIX)
# 40..43 (4): schema enum match quality (NONE, EXACT, STEM_OR_PREFIX, TYPO)
# 44..51 (8): signed token distance to nearest matched field

LEX_EQ_SET = {"=", "==", ":", "is", "equals", "equal", "eq", "matching", "as"}
LEX_NEQ_SET = {"!=", "!==", "<>", "isnt", "neq", "except", "excluding", "other_than"}
LEX_GT_SET = {">", ">=", "over", "above", "greater", "more", "exceeding", "gt", "gte", "min", "at_least", "atleast", "exceeds"}
LEX_LT_SET = {"<", "<=", "under", "below", "less", "fewer", "lt", "lte", "max", "at_most", "atmost", "within"}
LEX_CONTAINS_SET = {"~", "contains", "containing", "has", "including", "like", "about", "includes"}
LEX_BEFORE_SET = {"before", "prior", "earlier", "until", "older", "pre"}
LEX_AFTER_SET = {"after", "since", "newer", "later", "post", "starting"}
LEX_AND_SET = {"and", "&", "&&", ",", ";", "plus", "also", "with", "having"}
LEX_NEGATE_SET = {"not", "no", "non", "never", "!", "without", "arent", "wasnt", "werent", "cannot"}
LEX_BOOL_TRUE_SET = {"true", "yes", "on", "enabled", "active"}
LEX_BOOL_FALSE_SET = {"false", "off", "disabled", "inactive"}
LEX_FILLER_SET = {
    "show", "me", "all", "find", "list", "get", "where", "that", "are", "was", "were",
    "the", "for", "please", "only", "records", "items", "entries", "rows", "than",
    "to", "by", "from", "in", "of", "at", "any", "which", "whose", "been", "be", "have", "had", "went"
}

PROTECTED_GRAMMAR_WORDS = (
    LEX_EQ_SET
    | LEX_NEQ_SET
    | LEX_GT_SET
    | LEX_LT_SET
    | LEX_CONTAINS_SET
    | LEX_BEFORE_SET
    | LEX_AFTER_SET
    | LEX_AND_SET
    | LEX_NEGATE_SET
    | LEX_BOOL_TRUE_SET
    | LEX_BOOL_FALSE_SET
    | LEX_FILLER_SET
)

RELATIVE_DATE_WORDS = {
    "today", "yesterday", "tomorrow", "last_week", "last_month", "this_week",
    "this_month", "now", "recent", "q1", "q2", "q3", "q4",
}

_TOKEN_RE = re.compile(
    r"""!=|!==|>=|<=|==|<>|&&|[=:><~!,;&]|"[^"]+"|'[^']+'|\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|[@$/\w.-]+""",
    re.UNICODE,
)


@dataclass(frozen=True)
class TokenSpan:
    text: str
    norm: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenFeatureMeta:
    token: TokenSpan
    feature_ids: tuple[int, ...]
    matched_field: str | None
    matched_field_kind: str | None
    field_match_quality: str
    bool_neg_prefix: bool
    matched_enum_field: str | None
    matched_enum_value: str | None
    enum_match_quality: str
    distance_to_field: int | None


def normalize_token(raw: str) -> str:
    s = raw.strip().lower()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def tokenize_query(query: str, max_tokens: int = MAX_TOKENS) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for m in _TOKEN_RE.finditer(query):
        raw = m.group(0)
        norm = normalize_token(raw)
        if not norm:
            continue
        spans.append(TokenSpan(text=raw, norm=norm, start=m.start(), end=m.end()))
        if len(spans) >= max_tokens:
            break
    return spans


@lru_cache(maxsize=8192)
def simple_stem(word: str) -> str:
    w = word.lower().replace("-", "_")
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("ms"):
        return w[:-1]
    return w


@lru_cache(maxsize=16384)
def damerau_levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return abs(la - lb)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[la][lb]


@lru_cache(maxsize=8192)
def match_schema_field(norm_token: str, schema: SchemaDef) -> tuple[FieldDef | None, str, bool]:
    """Return (FieldDef | None, quality, is_bool_neg_prefix)."""
    t = norm_token.replace("-", "_")
    # 1. Exact match on field name or alias
    for f in schema.fields:
        if t == f.name.lower():
            return f, "EXACT", False
        for a in f.aliases:
            if t == a.lower():
                return f, "EXACT", False

    if t in PROTECTED_GRAMMAR_WORDS:
        return None, "NONE", False

    # 2. Negative prefix on BOOL fields (e.g. unmitigated, unlocked, unverified, unsubscribed)
    for prefix in ("un_", "un", "non_", "non"):
        if t.startswith(prefix) and len(t) > len(prefix) + 2:
            rest = t[len(prefix):]
            for f in schema.fields:
                if f.kind != "BOOL":
                    continue
                candidates = (f.name.lower(), *(a.lower() for a in f.aliases))
                for cand in candidates:
                    if rest == cand or simple_stem(rest) == simple_stem(cand) or damerau_levenshtein(rest, cand) <= 1:
                        return f, "BOOL_NEG_PREFIX", True

    # 3. Stem or prefix match
    t_stem = simple_stem(t)
    for f in schema.fields:
        candidates = (f.name.lower(), *(a.lower() for a in f.aliases))
        for cand in candidates:
            c_stem = simple_stem(cand)
            if len(t_stem) >= 3 and t_stem == c_stem:
                return f, "STEM_OR_PREFIX", False
            if len(t) >= 4 and (cand.startswith(t) or t.startswith(cand)):
                return f, "STEM_OR_PREFIX", False

    # 4. Typo match (Damerau-Levenshtein)
    best_field: FieldDef | None = None
    best_dist = 99
    for f in schema.fields:
        candidates = (f.name.lower(), *(a.lower() for a in f.aliases))
        for cand in candidates:
            if len(t) < 4 or len(cand) < 4:
                continue
            dist = damerau_levenshtein(t, cand)
            max_allowed = 2 if (len(cand) >= 6 and len(t) >= 5 and t[0] == cand[0]) else 1
            if dist <= max_allowed and dist < best_dist:
                best_dist = dist
                best_field = f
    if best_field is not None:
        return best_field, "TYPO", False

    return None, "NONE", False


@lru_cache(maxsize=8192)
def match_schema_enum(norm_token: str, schema: SchemaDef) -> tuple[FieldDef | None, str | None, str]:
    """Return (FieldDef | None, canonical_enum_value | None, quality)."""
    t = norm_token.replace("-", "_")
    # 1. Exact enum value match
    for f in schema.fields:
        if f.kind != "ENUM":
            continue
        for v in f.enum_values:
            if t == v.lower():
                return f, v, "EXACT"

    if t in PROTECTED_GRAMMAR_WORDS:
        return None, None, "NONE"

    # 2. Stem or prefix match on enum values
    t_stem = simple_stem(t)
    for f in schema.fields:
        if f.kind != "ENUM":
            continue
        for v in f.enum_values:
            vl = v.lower()
            if len(t_stem) >= 3 and t_stem == simple_stem(vl):
                return f, v, "STEM_OR_PREFIX"
            if len(t) >= 4 and (vl.startswith(t) or t.startswith(vl)):
                return f, v, "STEM_OR_PREFIX"

    # 3. Typo match on enum values
    best_f: FieldDef | None = None
    best_v: str | None = None
    best_dist = 99
    for f in schema.fields:
        if f.kind != "ENUM":
            continue
        for v in f.enum_values:
            vl = v.lower()
            if len(t) < 4 or len(vl) < 4:
                continue
            dist = damerau_levenshtein(t, vl)
            max_allowed = 2 if (len(vl) >= 6 and len(t) >= 5 and t[0] == vl[0]) else 1
            if dist <= max_allowed and dist < best_dist:
                best_dist = dist
                best_f = f
                best_v = v
    if best_f is not None and best_v is not None:
        return best_f, best_v, "TYPO"

    return None, None, "NONE"


def char_shape_bucket(raw: str, norm: str) -> int:
    """Return shape index in 2..10."""
    if raw in {"=", "==", "!=", "!==", ">", ">=", "<", "<=", ":", "~", "!", "&", "&&", ",", ";", "<>"}:
        return 8  # SYMBOL_OP
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")) or "/" in raw or raw.startswith("@"):
        return 9  # QUOTED_OR_PATH
    if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", norm) or norm in RELATIVE_DATE_WORDS or re.fullmatch(r"\d+[dhwmy]", norm):
        return 7  # ISO_OR_REL_DATE
    if re.fullmatch(r"-?\d+(\.\d+)?", norm):
        return 5  # PURE_NUMBER
    if re.fullmatch(r"\$?-?\d+(\.\d+)?(ms|kb|mb|gb|s|m|h|usd|k)?", norm):
        return 6  # NUMBER_WITH_UNIT
    if re.fullmatch(r"[a-z]+", norm):
        return 2  # ALL_ALPHA
    if re.fullmatch(r"[a-z]+([_-][a-z0-9]+)+", norm):
        return 3  # ALPHA_UNDERSCORE_OR_HYPHEN
    if re.fullmatch(r"[a-z]+\d+[a-z0-9]*", norm):
        return 4  # ALPHANUM_CODE (p0, sev1, int6)
    return 10  # OTHER_SHAPE


def length_bucket(norm: str) -> int:
    """Return length index in 11..15."""
    n = len(norm)
    if n <= 1:
        return 11
    if n <= 3:
        return 12
    if n <= 6:
        return 13
    if n <= 10:
        return 14
    return 15


def lexicon_bucket(norm: str) -> int:
    """Return lexicon index in 16..28."""
    if norm in LEX_EQ_SET:
        return 17
    if norm in LEX_NEQ_SET:
        return 18
    if norm in LEX_GT_SET:
        return 19
    if norm in LEX_LT_SET:
        return 20
    if norm in LEX_CONTAINS_SET:
        return 21
    if norm in LEX_BEFORE_SET:
        return 22
    if norm in LEX_AFTER_SET:
        return 23
    if norm in LEX_AND_SET:
        return 24
    if norm in LEX_NEGATE_SET:
        return 25
    if norm in LEX_BOOL_TRUE_SET:
        return 26
    if norm in LEX_BOOL_FALSE_SET:
        return 27
    if norm in LEX_FILLER_SET:
        return 28
    return 16  # LEX_NONE


def field_kind_bucket(kind: str | None) -> int:
    """Return field match kind index in 29..34."""
    if kind == "BOOL":
        return 30
    if kind == "ENUM":
        return 31
    if kind == "DATE":
        return 32
    if kind == "NUMBER":
        return 33
    if kind == "TEXT":
        return 34
    return 29


def field_quality_bucket(quality: str) -> int:
    """Return field match quality index in 35..39."""
    if quality == "EXACT":
        return 36
    if quality == "STEM_OR_PREFIX":
        return 37
    if quality == "TYPO":
        return 38
    if quality == "BOOL_NEG_PREFIX":
        return 39
    return 35


def enum_quality_bucket(quality: str) -> int:
    """Return enum match quality index in 40..43."""
    if quality == "EXACT":
        return 41
    if quality == "STEM_OR_PREFIX":
        return 42
    if quality == "TYPO":
        return 43
    return 40


def distance_bucket(dist: int | None, has_enum_match: bool) -> int:
    """Return distance index in 44..51."""
    if dist is None:
        return 50 if has_enum_match else 51
    if dist == 0:
        return 44
    if dist == 1:
        return 45
    if dist == 2:
        return 46
    if dist >= 3:
        return 47
    if dist == -1:
        return 48
    return 49  # dist <= -2


def featurize_query(
    query: str,
    schema: SchemaDef,
    use_word_identity: bool = False,
) -> list[TokenFeatureMeta]:
    """Zero-vocabulary structural featurizer.
    By default use_word_identity=False folds all word identity onto slot 1 (active token).
    """
    tokens = tokenize_query(query)
    if not tokens:
        return []

    raw_matches: list[tuple[FieldDef | None, str, bool, FieldDef | None, str | None, str]] = []
    field_positions: list[int] = []

    for idx, tok in enumerate(tokens):
        f_def, f_qual, is_neg_bool = match_schema_field(tok.norm, schema)
        e_field, e_val, e_qual = match_schema_enum(tok.norm, schema)
        # If token matched both an exact enum value and only a typo/prefix field, prefer exact enum
        if e_qual == "EXACT" and f_qual in {"TYPO", "STEM_OR_PREFIX"}:
            f_def, f_qual, is_neg_bool = None, "NONE", False
        if f_def is not None:
            field_positions.append(idx)
        raw_matches.append((f_def, f_qual, is_neg_bool, e_field, e_val, e_qual))

    results: list[TokenFeatureMeta] = []
    for idx, tok in enumerate(tokens):
        f_def, f_qual, is_neg_bool, e_field, e_val, e_qual = raw_matches[idx]

        # Nearest matched field distance (prefer preceding field in same clause if tied)
        nearest_dist: int | None = None
        if field_positions:
            best_pos = min(
                field_positions,
                key=lambda p: (abs(idx - p), 0 if idx >= p else 1),
            )
            nearest_dist = idx - best_pos

        word_slot = 1 if not use_word_identity else (1 + (hash(tok.norm) % 1))
        f_ids = (
            word_slot,
            char_shape_bucket(tok.text, tok.norm),
            length_bucket(tok.norm),
            lexicon_bucket(tok.norm),
            field_kind_bucket(f_def.kind if f_def is not None else None),
            field_quality_bucket(f_qual),
            enum_quality_bucket(e_qual),
            distance_bucket(nearest_dist, e_field is not None),
        )
        results.append(
            TokenFeatureMeta(
                token=tok,
                feature_ids=f_ids,
                matched_field=f_def.name if f_def is not None else None,
                matched_field_kind=f_def.kind if f_def is not None else None,
                field_match_quality=f_qual,
                bool_neg_prefix=is_neg_bool,
                matched_enum_field=e_field.name if e_field is not None else None,
                matched_enum_value=e_val,
                enum_match_quality=e_qual,
                distance_to_field=nearest_dist,
            )
        )
    return results
