from __future__ import annotations
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from spike.featurize import (
    LEX_AFTER_SET,
    LEX_AND_SET,
    LEX_BEFORE_SET,
    LEX_BOOL_FALSE_SET,
    LEX_BOOL_TRUE_SET,
    LEX_CONTAINS_SET,
    LEX_EQ_SET,
    LEX_GT_SET,
    LEX_LT_SET,
    LEX_NEGATE_SET,
    LEX_NEQ_SET,
    TokenFeatureMeta,
    featurize_query,
    simple_stem,
    tokenize_query,
)
from spike.model import ROLES
from spike.schema import FieldDef, SchemaDef

ROLE_TO_CMP: dict[str, str] = {
    "OP_EQ": "eq",
    "OP_NEQ": "neq",
    "OP_GT": "gt",
    "OP_LT": "lt",
    "OP_CONTAINS": "contains",
    "OP_BEFORE": "before",
    "OP_AFTER": "after",
}


@dataclass(frozen=True)
class FilterClause:
    field: str
    cmp: str
    value: str
    negated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "cmp": self.cmp,
            "value": self.value,
            "negated": self.negated,
        }


def clean_numeric_value(raw: str) -> str | None:
    s = raw.strip().lower().lstrip("$")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(ms|kb|mb|gb|s|m|h|usd|k)?", s)
    if not m:
        return None
    num_str, suffix = m.group(1), m.group(2)
    if suffix == "k":
        val = float(num_str) * 1000.0
        return str(int(val)) if val.is_integer() else str(val)
    return num_str


def infer_operator_from_token(norm: str) -> str | None:
    if norm in LEX_NEQ_SET:
        return "neq"
    if norm in LEX_GT_SET:
        return "gt"
    if norm in LEX_LT_SET:
        return "lt"
    if norm in LEX_CONTAINS_SET:
        return "contains"
    if norm in LEX_BEFORE_SET:
        return "before"
    if norm in LEX_AFTER_SET:
        return "after"
    if norm in LEX_EQ_SET:
        return "eq"
    return None


def _resolve_clause_span(
    span: list[tuple[TokenFeatureMeta, str]],
    schema: SchemaDef,
) -> FilterClause | None:
    if not span:
        return None

    field_map = schema.field_map()
    negated = False
    cmp_op: str | None = None
    explicit_field: str | None = None
    value_metas: list[TokenFeatureMeta] = []

    for meta, role in span:
        if meta.bool_neg_prefix:
            negated = True
        if role == "NEGATE" or meta.token.norm in LEX_NEGATE_SET:
            negated = True
            continue
        if role in ROLE_TO_CMP:
            if cmp_op is None or cmp_op == "eq":
                cmp_op = ROLE_TO_CMP[role]
            continue
        if role == "FIELD":
            if explicit_field is None and meta.matched_field is not None:
                explicit_field = meta.matched_field
            elif explicit_field is None and meta.matched_enum_field is not None:
                # Model tagged an implicit enum token as FIELD
                explicit_field = meta.matched_enum_field
                value_metas.append(meta)
            continue
        if role == "VALUE":
            value_metas.append(meta)

    # Fallback: if no operator was tagged by role, inspect operator lexicon tokens in span
    if cmp_op is None:
        for meta, role in span:
            if role not in {"FIELD", "VALUE"}:
                inferred = infer_operator_from_token(meta.token.norm)
                if inferred is not None:
                    cmp_op = inferred
                    break

    # Fallback: if explicit_field wasn't tagged as FIELD, check if any token in span matched a schema field
    if explicit_field is None:
        for meta, role in span:
            if meta.matched_field is not None and role != "NONE":
                explicit_field = meta.matched_field
                break

    # Case 1: We have a matched schema field
    if explicit_field is not None and explicit_field in field_map:
        f_def = field_map[explicit_field]
        final_cmp = cmp_op or "eq"

        if f_def.kind == "BOOL":
            # Check if there is an explicit true/false value token
            bool_val = True
            for vm in value_metas:
                if vm.token.norm in LEX_BOOL_FALSE_SET or vm.token.norm == "0":
                    bool_val = False
                elif vm.token.norm in LEX_BOOL_TRUE_SET or vm.token.norm == "1":
                    bool_val = True
            if final_cmp == "neq":
                final_cmp = "eq"
                negated = not negated
            if not bool_val:
                negated = not negated
            return FilterClause(field=f_def.name, cmp="eq", value="true", negated=negated)

        if f_def.kind == "ENUM":
            # Prefer a value token whose enum value belongs to this field
            chosen_val: str | None = None
            for vm in value_metas:
                if vm.matched_enum_field == f_def.name and vm.matched_enum_value is not None:
                    chosen_val = vm.matched_enum_value
                    break
            if chosen_val is None:
                # Check any token in the span that matched this field's enum values
                for meta, _ in span:
                    if meta.matched_enum_field == f_def.name and meta.matched_enum_value is not None:
                        chosen_val = meta.matched_enum_value
                        break
            if chosen_val is None and value_metas:
                raw_cand = value_metas[0].token.norm
                if raw_cand in f_def.enum_values:
                    chosen_val = raw_cand
            if chosen_val is None:
                return None
            return FilterClause(field=f_def.name, cmp=final_cmp, value=chosen_val, negated=negated)

        if f_def.kind == "NUMBER":
            for vm in value_metas:
                cleaned = clean_numeric_value(vm.token.norm)
                if cleaned is not None:
                    return FilterClause(field=f_def.name, cmp=final_cmp, value=cleaned, negated=negated)
            # Check any token in span if model missed VALUE tag on number
            for meta, role in span:
                if role != "FIELD":
                    cleaned = clean_numeric_value(meta.token.norm)
                    if cleaned is not None:
                        return FilterClause(field=f_def.name, cmp=final_cmp, value=cleaned, negated=negated)
            return None

        if f_def.kind == "DATE":
            if not value_metas:
                return None
            val_str = value_metas[0].token.norm
            if val_str in LEX_AND_SET or val_str in LEX_EQ_SET:
                return None
            return FilterClause(field=f_def.name, cmp=final_cmp, value=val_str, negated=negated)

        if f_def.kind == "TEXT":
            filtered_vals = [
                vm.token.norm
                for vm in value_metas
                if vm.matched_field != f_def.name and vm.token.norm not in LEX_EQ_SET
            ]
            if not filtered_vals:
                return None
            return FilterClause(
                field=f_def.name,
                cmp=final_cmp,
                value=" ".join(filtered_vals),
                negated=negated,
            )

    # Case 2: Implicit ENUM clause (e.g. "critical", "not prod", "sev1")
    for vm in value_metas:
        if vm.matched_enum_field is not None and vm.matched_enum_value is not None:
            return FilterClause(
                field=vm.matched_enum_field,
                cmp=cmp_op or "eq",
                value=vm.matched_enum_value,
                negated=negated,
            )

    return None


def compile_tokens(
    metas: Sequence[TokenFeatureMeta],
    role_ids: Sequence[int],
    boundary_flags: Sequence[int | bool],
    schema: SchemaDef,
) -> list[FilterClause]:
    """Deterministically compile token role predictions + clause boundaries into FilterClauses."""
    if not metas:
        return []

    spans: list[list[tuple[TokenFeatureMeta, str]]] = []
    current: list[tuple[TokenFeatureMeta, str]] = []

    def current_has_content() -> bool:
        return any(r in {"FIELD", "VALUE"} for _, r in current)

    def current_is_complete() -> bool:
        has_field = any(r == "FIELD" for _, r in current)
        has_val = any(r == "VALUE" for _, r in current)
        if has_field and has_val:
            return True
        # Standalone BOOL field
        if has_field and any(m.matched_field_kind == "BOOL" for m, r in current if r == "FIELD"):
            return True
        # Standalone implicit ENUM value
        if not has_field and has_val and any(m.matched_enum_field is not None for m, r in current if r == "VALUE"):
            return True
        return False

    for idx, meta in enumerate(metas):
        role = ROLES[role_ids[idx]] if 0 <= role_ids[idx] < len(ROLES) else "NONE"
        is_boundary = bool(boundary_flags[idx]) if idx < len(boundary_flags) else False

        if role == "AND" or meta.token.norm in {",", ";", "and", "&&", "&"}:
            if current:
                spans.append(current)
                current = []
            continue

        if role == "NONE":
            continue

        # Check if a new clause should start
        if current:
            if is_boundary and current_has_content():
                spans.append(current)
                current = []
            elif role == "FIELD" and current_is_complete():
                spans.append(current)
                current = []
            elif role == "NEGATE" and current_is_complete():
                spans.append(current)
                current = []
            elif (
                role == "VALUE"
                and meta.matched_enum_field is not None
                and current_is_complete()
                and not any(
                    r == "FIELD" and m.matched_field == meta.matched_enum_field
                    for m, r in current
                )
            ):
                spans.append(current)
                current = []

        current.append((meta, role))

    if current:
        spans.append(current)

    clauses: list[FilterClause] = []
    seen_keys: set[tuple[str, str, str, bool]] = set()
    for span in spans:
        clause = _resolve_clause_span(span, schema)
        if clause is not None:
            key = (clause.field, clause.cmp, clause.value, clause.negated)
            if key not in seen_keys:
                seen_keys.add(key)
                clauses.append(clause)
    return clauses


def canonicalize_clause(c: FilterClause | dict[str, Any]) -> tuple[str, str, str, bool]:
    if isinstance(c, FilterClause):
        field, cmp_op, val, neg = c.field, c.cmp, c.value, c.negated
    else:
        field = str(c["field"])
        cmp_op = str(c["cmp"])
        val = str(c["value"])
        neg = bool(c.get("negated", False))
    # Normalize (cmp="neq", negated=False) -> (cmp="eq", negated=True) for canonical AST equivalence
    if cmp_op == "neq":
        cmp_op = "eq"
        neg = not neg
    return (field, cmp_op, val.lower(), neg)


def ast_exact_match(
    pred: Sequence[FilterClause | dict[str, Any]],
    target: Sequence[FilterClause | dict[str, Any]],
) -> bool:
    if len(pred) != len(target):
        return False
    return [canonicalize_clause(c) for c in pred] == [canonicalize_clause(c) for c in target]


# ============================================================================
# Baseline 1: Deterministic Rule / Regex Parser (~120 lines, exact vocabulary)
# ============================================================================

_RULE_SPLIT_RE = re.compile(r"\s+(?:and|&&|&|plus|also|where|with|that\s+have)\s+|\s*[,;]\s*", re.IGNORECASE)
_RULE_INTRO_RE = re.compile(
    r"^(?:show\s+me|show\s+all|find\s+all\s+records\s+where|please\s+list\s+items\s+with|get\s+entries\s+that\s+have|find\s+rows\s+for|only\s+show)\s+",
    re.IGNORECASE,
)
_RULE_BIN_RE = re.compile(
    r"^\s*(?:(not)\s+)?([a-zA-Z0-9_]+)\s*"
    r"(!==|!=|==|=|>=|<=|>|<|:|~|\bis\s+not\b|\bis\b|\bequals\b|\bcontains\b|\bbefore\b|\bafter\b|\bover\b|\bunder\b)\s*"
    r"(?:(not)\s+)?([a-zA-Z0-9_./:@$-]+)\s*$",
    re.IGNORECASE,
)
_RULE_SINGLE_RE = re.compile(r"^\s*(?:(not|no|!)\s+)?([a-zA-Z0-9_]+)\s*$", re.IGNORECASE)


def baseline_parse(query: str, schema: SchemaDef) -> list[FilterClause]:
    """Deterministic Rule/Regex Parser (Baseline 1).
    Matches clean `<field> <op> <value>`, `<bool_field>`, and exact `<enum_value>` segments.
    Does not handle conversational filler within clauses, inverted syntax, `un-` prefixes, units, or typos.
    """
    exact_fields: dict[str, FieldDef] = {}
    exact_enums: dict[str, tuple[FieldDef, str]] = {}
    for f in schema.fields:
        exact_fields[f.name.lower()] = f
        for a in f.aliases:
            exact_fields[a.lower()] = f
        if f.kind == "ENUM":
            for v in f.enum_values:
                exact_enums[v.lower()] = (f, v)

    op_map = {
        "=": "eq",
        "==": "eq",
        ":": "eq",
        "is": "eq",
        "equals": "eq",
        "!=": "neq",
        "!==": "neq",
        "is not": "neq",
        ">": "gt",
        ">=": "gt",
        "over": "gt",
        "<": "lt",
        "<=": "lt",
        "under": "lt",
        "~": "contains",
        "contains": "contains",
        "before": "before",
        "after": "after",
    }

    cleaned_query = _RULE_INTRO_RE.sub("", query.strip())
    clauses: list[FilterClause] = []
    segments = _RULE_SPLIT_RE.split(cleaned_query)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m_bin = _RULE_BIN_RE.match(seg)
        if m_bin:
            neg1, f_tok, raw_op, neg2, v_tok = m_bin.groups()
            f_def = exact_fields.get(f_tok.lower())
            if f_def is None:
                continue
            op_norm = re.sub(r"\s+", " ", raw_op.lower().strip())
            cmp_op = op_map.get(op_norm, "eq")
            negated = bool(neg1 or neg2)
            v_clean = v_tok.strip().strip("\"'").lower()

            if f_def.kind == "BOOL":
                if cmp_op == "neq":
                    cmp_op = "eq"
                    negated = not negated
                if v_clean in {"false", "no", "off", "0"}:
                    negated = not negated
                clauses.append(FilterClause(f_def.name, "eq", "true", negated))
            elif f_def.kind == "ENUM":
                matched_v = next((ev for ev in f_def.enum_values if ev.lower() == v_clean), None)
                if matched_v is not None:
                    clauses.append(FilterClause(f_def.name, cmp_op, matched_v, negated))
            elif f_def.kind == "NUMBER":
                if re.fullmatch(r"-?\d+(?:\.\d+)?", v_clean):
                    clauses.append(FilterClause(f_def.name, cmp_op, v_clean, negated))
            else:
                clauses.append(FilterClause(f_def.name, cmp_op, v_clean, negated))
            continue

        m_single = _RULE_SINGLE_RE.match(seg)
        if m_single:
            neg_tok, word = m_single.groups()
            negated = bool(neg_tok)
            w_low = word.lower()
            if w_low in exact_fields and exact_fields[w_low].kind == "BOOL":
                clauses.append(FilterClause(exact_fields[w_low].name, "eq", "true", negated))
            elif w_low in exact_enums:
                ef, ev = exact_enums[w_low]
                clauses.append(FilterClause(ef.name, "eq", ev, negated))

    return clauses


# ============================================================================
# Baseline 3: Personal Centroid / Example Matcher (Zero-Retrain Adaptation)
# ============================================================================

CENTROID_DIM = 96


def fnv1a_32(text: str) -> int:
    """Deterministic 32-bit FNV-1a hash matching TypeScript unsigned 32-bit math."""
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def embed_personal_phrase(
    query: str,
    schema: SchemaDef | None = None,
    dim: int = CENTROID_DIM,
) -> list[float]:
    """Compute a 96-dim L2-normalized structural + character-ngram feature vector."""
    vec = [0.0] * dim
    tokens = tokenize_query(query)
    if not tokens:
        return vec

    def accumulate(feature_key: str, weight: float) -> None:
        h = fnv1a_32(feature_key)
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norms = [t.norm for t in tokens]
    for i, norm in enumerate(norms):
        stem = simple_stem(norm)
        accumulate(f"tok:{norm}", 1.5)
        accumulate(f"stem:{stem}", 1.2)
        if i + 1 < len(norms):
            accumulate(f"bi:{stem}_{simple_stem(norms[i + 1])}", 1.4)
        padded = f"^{norm}$"
        for k in range(len(padded) - 2):
            accumulate(f"tri:{padded[k:k+3]}", 0.5)

    if schema is not None:
        metas = featurize_query(query, schema)
        for m in metas:
            for fid in m.feature_ids[1:]:
                accumulate(f"struct:{fid}", 0.4)

    norm_sq = sum(v * v for v in vec)
    if norm_sq > 1e-12:
        inv = 1.0 / math.sqrt(norm_sq)
        vec = [v * inv for v in vec]
    return vec


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class CentroidPrototype:
    idiom_id: str
    schema_name: str
    examples: list[str]
    clauses: list[FilterClause]
    centroid: list[float]


@dataclass
class CentroidMatch:
    matched: bool
    idiom_id: str | None
    similarity: float
    clauses: list[FilterClause]


class CentroidMemory:
    """Personal Centroid/Example Matcher (Learnalist-style zero-retrain memory).
    Lets a user teach custom shorthand idioms (e.g. 'p0 fires' -> severity=critical & not mitigated)
    from 1-3 examples with cosine similarity + rejection threshold.
    """

    def __init__(self, dim: int = CENTROID_DIM, threshold: float = 0.62) -> None:
        self.dim = dim
        self.threshold = threshold
        self.prototypes: dict[str, CentroidPrototype] = {}

    def teach(
        self,
        idiom_id: str,
        schema: SchemaDef,
        examples: Sequence[str],
        clauses: Sequence[FilterClause | dict[str, Any]],
    ) -> CentroidPrototype:
        assert 1 <= len(examples) <= 5, "Provide 1-5 examples per personal idiom"
        avg = [0.0] * self.dim
        for ex in examples:
            emb = embed_personal_phrase(ex, schema, self.dim)
            for i in range(self.dim):
                avg[i] += emb[i]
        norm_sq = sum(v * v for v in avg)
        if norm_sq > 1e-12:
            inv = 1.0 / math.sqrt(norm_sq)
            avg = [v * inv for v in avg]

        parsed_clauses = [
            c if isinstance(c, FilterClause) else FilterClause(
                field=str(c["field"]),
                cmp=str(c["cmp"]),
                value=str(c["value"]),
                negated=bool(c.get("negated", False)),
            )
            for c in clauses
        ]
        proto = CentroidPrototype(
            idiom_id=idiom_id,
            schema_name=schema.name,
            examples=list(examples),
            clauses=parsed_clauses,
            centroid=avg,
        )
        self.prototypes[f"{schema.name}:{idiom_id}"] = proto
        return proto

    def query(self, text: str, schema: SchemaDef) -> CentroidMatch:
        q_emb = embed_personal_phrase(text, schema, self.dim)
        best_sim = -1.0
        best_proto: CentroidPrototype | None = None
        for proto in self.prototypes.values():
            if proto.schema_name != schema.name:
                continue
            sim = cosine_similarity(q_emb, proto.centroid)
            if sim > best_sim:
                best_sim = sim
                best_proto = proto

        if best_proto is not None and best_sim >= self.threshold:
            return CentroidMatch(
                matched=True,
                idiom_id=best_proto.idiom_id,
                similarity=round(best_sim, 4),
                clauses=list(best_proto.clauses),
            )
        return CentroidMatch(
            matched=False,
            idiom_id=best_proto.idiom_id if best_proto else None,
            similarity=round(max(best_sim, 0.0), 4),
            clauses=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "threshold": self.threshold,
            "prototypes": [
                {
                    "idiom_id": p.idiom_id,
                    "schema_name": p.schema_name,
                    "examples": p.examples,
                    "clauses": [c.to_dict() for c in p.clauses],
                    "centroid": [round(x, 6) for x in p.centroid],
                }
                for p in self.prototypes.values()
            ],
        }
