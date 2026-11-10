from __future__ import annotations
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from spike.compile_query import (
    CentroidMemory,
    FilterClause,
    ast_exact_match,
    baseline_parse,
    compile_tokens,
)
from spike.featurize import (
    FEATURES_PER_TOKEN,
    MAX_TOKENS,
    featurize_query,
)
from spike.model import (
    NUM_ROLES,
    QUANTIZATION_BITS,
    ROLE_TO_ID,
    ROLES,
    QueryTagger,
    quantize_tensor_int6,
)
from spike.schema import (
    ALL_SCHEMAS,
    TRAIN_SCHEMAS,
    TRANSFER_SCHEMAS,
    FieldDef,
    SchemaDef,
    assert_disjoint_schemas,
)

TEXT_VALUES_TRAIN = ("alice", "bob", "charlie", "dana", "auth_bug", "invoice_q3", "src/main", "core/api", "readme", "hotfix_v2")
TEXT_VALUES_TRANSFER = ("elena", "frank", "grace", "heidi", " Marcus", "k8s_node", "checkout_v2", "west_cluster", "acme_corp", "globex")

DATE_VALUES = ("today", "yesterday", "last_week", "last_month", "2026-10-01", "2026-10-15", "2026-11-01", "7d", "24h")
NUMBER_VALUES = ("5", "10", "25", "45", "100", "200", "500", "1000", "50000")

INTRO_FILLERS = (
    ["show", "me"],
    ["show", "all"],
    ["find", "all", "records", "where"],
    ["please", "list", "items", "with"],
    ["get", "entries", "that", "have"],
    ["find", "rows", "for"],
    ["only", "show"],
)


def make_typo(word: str, rng: random.Random) -> str:
    """Create a realistic 1-edit typo on words of length >= 5 while preserving the first character."""
    w = word.lower()
    if len(w) < 5:
        return w
    choice = rng.choice(("drop_vowel", "swap_mid", "drop_mid"))
    if choice == "drop_vowel":
        for i in range(2, len(w) - 1):
            if w[i] in "aeiou":
                return w[:i] + w[i + 1 :]
    if choice == "swap_mid" and len(w) >= 6:
        idx = rng.randint(1, len(w) - 3)
        if w[idx] != w[idx + 1]:
            return w[:idx] + w[idx + 1] + w[idx] + w[idx + 2 :]
    idx = rng.randint(2, len(w) - 2)
    return w[:idx] + w[idx + 1 :]


def make_stem_or_plural(word: str) -> str:
    w = word.lower()
    if w.endswith("y") and len(w) >= 5:
        return w[:-1] + "ies"
    if not w.endswith("s") and len(w) >= 4:
        return w + "s"
    return w


def pick_field_surface(f: FieldDef, rng: random.Random, messy: bool) -> str:
    candidates = [f.name, *f.aliases]
    base = rng.choice(candidates)
    if not messy:
        return base
    roll = rng.random()
    if roll < 0.38 and len(base) >= 5:
        return make_typo(base, rng)
    if roll < 0.60 and len(base) >= 4:
        return make_stem_or_plural(base)
    return base


def build_clause_tokens(
    f: FieldDef,
    schema: SchemaDef,
    rng: random.Random,
    messy: bool,
) -> tuple[list[tuple[str, str]], FilterClause]:
    """Return ([(token_str, role_name), ...], ground_truth_clause)."""
    is_transfer = schema.split == "transfer"
    text_pool = TEXT_VALUES_TRANSFER if is_transfer else TEXT_VALUES_TRAIN

    if f.kind == "BOOL":
        negated = rng.random() < 0.45
        if not messy:
            style = rng.choice(("bare", "explicit_bool", "eq_bool"))
            f_surf = rng.choice([f.name, *f.aliases])
            if style == "bare":
                if negated:
                    return [("not", "NEGATE"), (f_surf, "FIELD")], FilterClause(f.name, "eq", "true", True)
                return [(f_surf, "FIELD")], FilterClause(f.name, "eq", "true", False)
            if style == "explicit_bool":
                val_tok = "false" if negated else "true"
                op_tok = rng.choice(("=", ":", "is"))
                return [(f_surf, "FIELD"), (op_tok, "OP_EQ"), (val_tok, "VALUE")], FilterClause(f.name, "eq", "true", negated)
            val_tok = "false" if negated else "true"
            return [(f_surf, "FIELD"), ("=", "OP_EQ"), (val_tok, "VALUE")], FilterClause(f.name, "eq", "true", negated)
        else:
            style = rng.choice(("un_prefix", "without_bool", "conversational_bool", "typo_bool"))
            base_name = rng.choice([f.name, *f.aliases])
            if style == "un_prefix" and not base_name.startswith("un"):
                return [(f"un{base_name}", "FIELD")], FilterClause(f.name, "eq", "true", True)
            if style == "without_bool":
                surf = make_typo(base_name, rng) if len(base_name) >= 5 and rng.random() < 0.5 else base_name
                return [("without", "NEGATE"), (surf, "FIELD")], FilterClause(f.name, "eq", "true", True)
            if style == "conversational_bool":
                surf = pick_field_surface(f, rng, messy=True)
                if negated:
                    return [("that", "NONE"), ("are", "NONE"), ("not", "NEGATE"), (surf, "FIELD")], FilterClause(f.name, "eq", "true", True)
                return [("that", "NONE"), ("are", "NONE"), (surf, "FIELD")], FilterClause(f.name, "eq", "true", False)
            surf = make_typo(base_name, rng) if len(base_name) >= 5 else base_name
            if negated:
                return [("not", "NEGATE"), (surf, "FIELD")], FilterClause(f.name, "eq", "true", True)
            return [(surf, "FIELD")], FilterClause(f.name, "eq", "true", False)

    if f.kind == "ENUM":
        canon_val = rng.choice(f.enum_values)
        negated = rng.random() < 0.25
        if not messy:
            style = rng.choice(("explicit_eq", "explicit_colon", "implicit_enum", "explicit_neq"))
            f_surf = rng.choice([f.name, *f.aliases])
            if style == "implicit_enum":
                if negated:
                    return [("not", "NEGATE"), (canon_val, "VALUE")], FilterClause(f.name, "eq", canon_val, True)
                return [(canon_val, "VALUE")], FilterClause(f.name, "eq", canon_val, False)
            if style == "explicit_neq" and negated:
                return [(f_surf, "FIELD"), ("!=", "OP_NEQ"), (canon_val, "VALUE")], FilterClause(f.name, "neq", canon_val, False)
            op_tok = ":" if style == "explicit_colon" else rng.choice(("=", "is"))
            if negated:
                return [(f_surf, "FIELD"), ("is", "OP_EQ"), ("not", "NEGATE"), (canon_val, "VALUE")], FilterClause(f.name, "eq", canon_val, True)
            return [(f_surf, "FIELD"), (op_tok, "OP_EQ"), (canon_val, "VALUE")], FilterClause(f.name, "eq", canon_val, False)
        else:
            f_surf = pick_field_surface(f, rng, messy=True)
            v_surf = make_typo(canon_val, rng) if (len(canon_val) >= 5 and rng.random() < 0.45) else canon_val
            style = rng.choice(("conv_mid", "typo_pair", "implicit_typo", "excluding_enum"))
            if style == "conv_mid":
                if negated:
                    return [("the", "NONE"), (f_surf, "FIELD"), ("was", "NONE"), ("not", "NEGATE"), (v_surf, "VALUE")], FilterClause(f.name, "eq", canon_val, True)
                return [("the", "NONE"), (f_surf, "FIELD"), ("was", "NONE"), (v_surf, "VALUE")], FilterClause(f.name, "eq", canon_val, False)
            if style == "implicit_typo":
                v_typo = make_typo(canon_val, rng) if len(canon_val) >= 5 else canon_val
                if negated:
                    return [("not", "NEGATE"), (v_typo, "VALUE")], FilterClause(f.name, "eq", canon_val, True)
                return [(v_typo, "VALUE")], FilterClause(f.name, "eq", canon_val, False)
            if style == "excluding_enum":
                return [(f_surf, "FIELD"), ("excluding", "OP_NEQ"), (v_surf, "VALUE")], FilterClause(f.name, "neq", canon_val, False)
            return [(f_surf, "FIELD"), ("is", "OP_EQ"), (v_surf, "VALUE")], FilterClause(f.name, "eq", canon_val, False)

    if f.kind == "NUMBER":
        num_val = rng.choice(NUMBER_VALUES)
        cmp_choice = rng.choice(("gt", "lt", "eq"))
        role_op = {"gt": "OP_GT", "lt": "OP_LT", "eq": "OP_EQ"}[cmp_choice]
        if not messy:
            f_surf = rng.choice([f.name, *f.aliases])
            op_tok = {"gt": rng.choice((">", "over")), "lt": rng.choice(("<", "under")), "eq": "="}[cmp_choice]
            return [(f_surf, "FIELD"), (op_tok, role_op), (num_val, "VALUE")], FilterClause(f.name, cmp_choice, num_val, False)
        else:
            f_surf = pick_field_surface(f, rng, messy=True)
            unit = rng.choice(("ms", "kb", "m", "usd", ""))
            v_surf = f"{num_val}{unit}"
            style = rng.choice(("went_over", "inverted_num", "typo_num"))
            op_word = {"gt": rng.choice(("over", "above", "exceeding")), "lt": rng.choice(("under", "below", "within")), "eq": "equals"}[cmp_choice]
            if style == "went_over":
                return [(f_surf, "FIELD"), ("went", "NONE"), (op_word, role_op), (v_surf, "VALUE")], FilterClause(f.name, cmp_choice, num_val, False)
            if style == "inverted_num" and cmp_choice in {"gt", "lt"}:
                return [(op_word, role_op), (v_surf, "VALUE"), (f_surf, "FIELD")], FilterClause(f.name, cmp_choice, num_val, False)
            return [(f_surf, "FIELD"), (op_word, role_op), (v_surf, "VALUE")], FilterClause(f.name, cmp_choice, num_val, False)

    if f.kind == "DATE":
        d_val = rng.choice(DATE_VALUES)
        cmp_choice = rng.choice(("before", "after", "eq"))
        role_op = {"before": "OP_BEFORE", "after": "OP_AFTER", "eq": "OP_EQ"}[cmp_choice]
        if not messy:
            f_surf = rng.choice([f.name, *f.aliases])
            op_tok = {"before": "before", "after": "after", "eq": ":"}[cmp_choice]
            return [(f_surf, "FIELD"), (op_tok, role_op), (d_val, "VALUE")], FilterClause(f.name, cmp_choice, d_val, False)
        else:
            f_surf = pick_field_surface(f, rng, messy=True)
            op_word = {"before": rng.choice(("before", "prior", "earlier")), "after": rng.choice(("after", "since", "later")), "eq": "on"}[cmp_choice]
            if cmp_choice == "eq":
                return [(f_surf, "FIELD"), ("is", "OP_EQ"), (d_val, "VALUE")], FilterClause(f.name, "eq", d_val, False)
            if rng.random() < 0.45:
                return [(op_word, role_op), (d_val, "VALUE"), (f_surf, "FIELD")], FilterClause(f.name, cmp_choice, d_val, False)
            return [(f_surf, "FIELD"), ("was", "NONE"), (op_word, role_op), (d_val, "VALUE")], FilterClause(f.name, cmp_choice, d_val, False)

    # TEXT field
    t_val = rng.choice(text_pool).strip().lower()
    cmp_choice = rng.choice(("eq", "contains"))
    role_op = "OP_CONTAINS" if cmp_choice == "contains" else "OP_EQ"
    if not messy:
        f_surf = rng.choice([f.name, *f.aliases])
        op_tok = rng.choice(("contains", "~")) if cmp_choice == "contains" else rng.choice(("=", ":", "is"))
        return [(f_surf, "FIELD"), (op_tok, role_op), (t_val, "VALUE")], FilterClause(f.name, cmp_choice, t_val, False)
    else:
        f_surf = pick_field_surface(f, rng, messy=True)
        if cmp_choice == "contains":
            op_word = rng.choice(("contains", "including", "like", "has"))
            return [("the", "NONE"), (f_surf, "FIELD"), (op_word, role_op), (t_val, "VALUE")], FilterClause(f.name, "contains", t_val, False)
        return [("the", "NONE"), (f_surf, "FIELD"), ("was", "NONE"), ("is", "OP_EQ"), (t_val, "VALUE")], FilterClause(f.name, "eq", t_val, False)


def synthesize_sample(
    schemas: tuple[SchemaDef, ...],
    rng: random.Random,
    messy_prob: float = 0.55,
) -> dict[str, Any]:
    schema = rng.choice(schemas)
    num_clauses = rng.choices((1, 2, 3), weights=(0.30, 0.50, 0.20))[0]
    chosen_fields = rng.sample(list(schema.fields), k=min(num_clauses, len(schema.fields)))
    is_messy = rng.random() < messy_prob

    tokens_roles_bounds: list[tuple[str, str, int]] = []
    clauses: list[FilterClause] = []

    if is_messy and rng.random() < 0.65:
        intro = rng.choice(INTRO_FILLERS)
        for w in intro:
            tokens_roles_bounds.append((w, "NONE", 0))

    for c_idx, f_def in enumerate(chosen_fields):
        if c_idx > 0:
            join_style = rng.choice(("and", "comma", "also", "space") if is_messy else ("and", "comma", "also"))
            if join_style == "and":
                tokens_roles_bounds.append(("and", "AND", 0))
            elif join_style == "comma":
                tokens_roles_bounds.append((",", "AND", 0))
            elif join_style == "also":
                tokens_roles_bounds.append(("also", "AND", 0))

        c_tokens, clause = build_clause_tokens(f_def, schema, rng, messy=is_messy)
        marked_boundary = False
        for tok_str, role_name in c_tokens:
            b_flag = 0
            if not marked_boundary and role_name != "NONE":
                b_flag = 1
                marked_boundary = True
            tokens_roles_bounds.append((tok_str, role_name, b_flag))
        clauses.append(clause)

    # Assemble query string with realistic spacing around punctuation
    parts: list[str] = []
    for tok_str, _, _ in tokens_roles_bounds:
        if tok_str in {",", ";"} and parts:
            parts[-1] = parts[-1] + tok_str
        elif tok_str == ":" and parts and rng.random() < 0.5:
            parts[-1] = parts[-1] + ":"
        else:
            parts.append(tok_str)
    query_str = " ".join(parts)

    metas = featurize_query(query_str, schema, use_word_identity=False)
    n_tok = min(len(metas), len(tokens_roles_bounds), MAX_TOKENS)
    metas = metas[:n_tok]

    feat_ids = [[0] * FEATURES_PER_TOKEN for _ in range(MAX_TOKENS)]
    role_targets = [0] * MAX_TOKENS
    bound_targets = [0.0] * MAX_TOKENS
    mask = [0.0] * MAX_TOKENS

    for i in range(n_tok):
        feat_ids[i] = list(metas[i].feature_ids)
        role_targets[i] = ROLE_TO_ID[tokens_roles_bounds[i][1]]
        bound_targets[i] = float(tokens_roles_bounds[i][2])
        mask[i] = 1.0

    return {
        "query": query_str,
        "schema_name": schema.name,
        "messy": is_messy,
        "clauses": clauses,
        "feat_ids": feat_ids,
        "role_targets": role_targets,
        "bound_targets": bound_targets,
        "mask": mask,
    }


def make_tensor_dataset(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "feat_ids": torch.tensor([s["feat_ids"] for s in samples], dtype=torch.long, device=device),
        "role_targets": torch.tensor([s["role_targets"] for s in samples], dtype=torch.long, device=device),
        "bound_targets": torch.tensor([s["bound_targets"] for s in samples], dtype=torch.float32, device=device),
        "mask": torch.tensor([s["mask"] for s in samples], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def evaluate_split(
    model: QueryTagger,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    ds = make_tensor_dataset(samples, device)
    out = model(ds["feat_ids"], ds["mask"], return_traces=False)
    pred_roles = out["role_logits"].argmax(dim=-1).cpu().tolist()
    pred_bounds = (torch.sigmoid(out["boundary_logits"]) >= 0.5).int().cpu().tolist()

    neural_hits = 0
    baseline_hits = 0
    token_role_correct = 0
    token_total = 0

    per_schema: dict[str, dict[str, int]] = {}
    sample_failures: list[dict[str, Any]] = []

    for idx, s in enumerate(samples):
        schema = ALL_SCHEMAS[s["schema_name"]]
        if schema.name not in per_schema:
            per_schema[schema.name] = {"total": 0, "neural_hits": 0, "baseline_hits": 0}
        per_schema[schema.name]["total"] += 1

        metas = featurize_query(s["query"], schema, use_word_identity=False)
        n = len(metas)
        for t_i in range(n):
            token_total += 1
            if pred_roles[idx][t_i] == s["role_targets"][t_i]:
                token_role_correct += 1

        neural_clauses = compile_tokens(metas, pred_roles[idx][:n], pred_bounds[idx][:n], schema)
        base_clauses = baseline_parse(s["query"], schema)

        n_ok = ast_exact_match(neural_clauses, s["clauses"])
        b_ok = ast_exact_match(base_clauses, s["clauses"])

        if n_ok:
            neural_hits += 1
            per_schema[schema.name]["neural_hits"] += 1
        elif len(sample_failures) < 5:
            sample_failures.append({
                "query": s["query"],
                "schema": schema.name,
                "expected": [c.to_dict() for c in s["clauses"]],
                "predicted": [c.to_dict() for c in neural_clauses],
            })
        if b_ok:
            baseline_hits += 1
            per_schema[schema.name]["baseline_hits"] += 1

    total = max(len(samples), 1)
    return {
        "count": len(samples),
        "neural_ast_accuracy": round(neural_hits / total, 4),
        "baseline_ast_accuracy": round(baseline_hits / total, 4),
        "token_role_accuracy": round(token_role_correct / max(token_total, 1), 4),
        "per_schema": {
            k: {
                "count": v["total"],
                "neural_ast_accuracy": round(v["neural_hits"] / max(v["total"], 1), 4),
                "baseline_ast_accuracy": round(v["baseline_hits"] / max(v["total"], 1), 4),
            }
            for k, v in per_schema.items()
        },
        "sample_failures": sample_failures,
    }


def evaluate_personal_centroids(model: QueryTagger, device: torch.device) -> dict[str, Any]:
    """Evaluate Baseline 3 (CentroidMemory) on taught shorthand idioms vs Baseline 1 and standalone QueryTagger."""
    memory = CentroidMemory(threshold=0.58)

    idiom_specs = [
        {
            "idiom_id": "p0_fires",
            "schema": ALL_SCHEMAS["incidents"],
            "teach_examples": ["p0 fires", "active p0 fire", "unmitigated p0 fires"],
            "test_phrases": ["p0 fires", "current p0 fire", "any active p0 fires", "unmitigated p0 fire"],
            "clauses": [
                FilterClause("severity", "eq", "critical", False),
                FilterClause("mitigated", "eq", "true", True),
            ],
        },
        {
            "idiom_id": "auth_meltdown",
            "schema": ALL_SCHEMAS["incidents"],
            "teach_examples": ["auth meltdown", "login outage meltdown", "auth service meltdown"],
            "test_phrases": ["auth meltdown", "active auth meltdown", "login meltdown outage", "auth service meltdowns"],
            "clauses": [
                FilterClause("service", "eq", "auth", False),
                FilterClause("severity", "eq", "sev1", False),
            ],
        },
        {
            "idiom_id": "bad_prod_rollout",
            "schema": ALL_SCHEMAS["deployments"],
            "teach_examples": ["bad prod push", "flaky prod rollout", "botched prod deploy"],
            "test_phrases": ["bad prod push", "flaky prod rollouts", "botched prod deployment", "bad prod rollout"],
            "clauses": [
                FilterClause("environment", "eq", "prod", False),
                FilterClause("outcome", "eq", "rolled_back", False),
            ],
        },
        {
            "idiom_id": "whale_accounts",
            "schema": ALL_SCHEMAS["contacts"],
            "teach_examples": ["whale accounts", "big enterprise whales", "top tier whales"],
            "test_phrases": ["whale accounts", "enterprise whale account", "big whale accounts", "top tier whale"],
            "clauses": [
                FilterClause("segment", "eq", "enterprise", False),
                FilterClause("arr_usd", "gt", "100000", False),
            ],
        },
        {
            "idiom_id": "oom_diverged_runs",
            "schema": ALL_SCHEMAS["experiments"],
            "teach_examples": ["nan blowups", "diverged nan runs", "exploded loss runs"],
            "test_phrases": ["nan blowups", "diverged nan run", "exploded loss run", "nan blowup runs"],
            "clauses": [
                FilterClause("metric_status", "eq", "diverged", False),
            ],
        },
        {
            "idiom_id": "my_hot_bugs",
            "schema": ALL_SCHEMAS["issues"],
            "teach_examples": ["triage hotlist", "urgent bug queue", "blocker bug hotlist"],
            "test_phrases": ["triage hotlist", "urgent bug queues", "blocker bug hotlist", "my triage hotlist"],
            "clauses": [
                FilterClause("labels", "eq", "bug", False),
                FilterClause("priority", "eq", "urgent", False),
                FilterClause("status", "eq", "open", False),
            ],
        },
    ]

    for spec in idiom_specs:
        memory.teach(
            idiom_id=spec["idiom_id"],
            schema=spec["schema"],
            examples=spec["teach_examples"],
            clauses=spec["clauses"],
        )

    centroid_hits = 0
    neural_only_hits = 0
    baseline_hits = 0
    total_pos = 0

    for spec in idiom_specs:
        schema = spec["schema"]
        target = spec["clauses"]
        for phrase in spec["test_phrases"]:
            total_pos += 1
            c_res = memory.query(phrase, schema)
            if c_res.matched and ast_exact_match(c_res.clauses, target):
                centroid_hits += 1

            b_clauses = baseline_parse(phrase, schema)
            if ast_exact_match(b_clauses, target):
                baseline_hits += 1

            metas = featurize_query(phrase, schema)
            feat_ids = [[0] * FEATURES_PER_TOKEN for _ in range(MAX_TOKENS)]
            mask = [0.0] * MAX_TOKENS
            for i, m in enumerate(metas[:MAX_TOKENS]):
                feat_ids[i] = list(m.feature_ids)
                mask[i] = 1.0
            out = model(
                torch.tensor([feat_ids], dtype=torch.long, device=device),
                torch.tensor([mask], dtype=torch.float32, device=device),
            )
            r_ids = out["role_logits"][0].argmax(dim=-1).cpu().tolist()[: len(metas)]
            b_ids = (torch.sigmoid(out["boundary_logits"][0]) >= 0.5).int().cpu().tolist()[: len(metas)]
            n_clauses = compile_tokens(metas, r_ids, b_ids, schema)
            if ast_exact_match(n_clauses, target):
                neural_only_hits += 1

    # Also test rejection rate on normal schema queries (should NOT trigger false centroid match)
    rng = random.Random(9001)
    neg_queries = [synthesize_sample(TRANSFER_SCHEMAS, rng, messy_prob=0.5) for _ in range(40)]
    false_positives = 0
    for nq in neg_queries:
        schema = ALL_SCHEMAS[nq["schema_name"]]
        c_res = memory.query(nq["query"], schema)
        if c_res.matched:
            false_positives += 1

    return {
        "idioms_taught": len(idiom_specs),
        "positive_eval_count": total_pos,
        "centroid_accuracy": round(centroid_hits / max(total_pos, 1), 4),
        "neural_only_accuracy": round(neural_only_hits / max(total_pos, 1), 4),
        "baseline_rule_accuracy": round(baseline_hits / max(total_pos, 1), 4),
        "rejection_specificity": round(1.0 - (false_positives / len(neg_queries)), 4),
        "memory_export": memory.to_dict(),
    }


@torch.no_grad()
def build_parity_traces(model: QueryTagger, device: torch.device) -> list[dict[str, Any]]:
    """Generate golden forward-pass traces on quantized weights for TypeScript parity verification."""
    test_cases = [
        ("incidents", "show me incidents where the severty was critcal and unmitigated"),
        ("deployments", "environment is prod and latency_ms went over 200ms and not rolled_back"),
        ("experiments", "the framewrk is pytorch and perplexity under 10 and quantized"),
        ("contacts", "segment: enterprise, arr_usd > 50000, not churned"),
        ("issues", "status = open and priority != p3 and points > 5"),
    ]
    traces: list[dict[str, Any]] = []
    for schema_name, query in test_cases:
        schema = ALL_SCHEMAS[schema_name]
        metas = featurize_query(query, schema, use_word_identity=False)
        n = len(metas)
        feat_ids = [[0] * FEATURES_PER_TOKEN for _ in range(MAX_TOKENS)]
        mask = [0.0] * MAX_TOKENS
        for i in range(n):
            feat_ids[i] = list(metas[i].feature_ids)
            mask[i] = 1.0
        out = model(
            torch.tensor([feat_ids], dtype=torch.long, device=device),
            torch.tensor([mask], dtype=torch.float32, device=device),
            return_traces=True,
        )
        role_logits = out["role_logits"][0, :n, :].cpu()
        role_probs = F.softmax(role_logits, dim=-1)
        role_ids = role_logits.argmax(dim=-1).tolist()
        bound_probs = torch.sigmoid(out["boundary_logits"][0, :n]).cpu().tolist()
        bound_flags = [1 if p >= 0.5 else 0 for p in bound_probs]
        clauses = compile_tokens(metas, role_ids, bound_flags, schema)

        traces.append({
            "schema": schema_name,
            "query": query,
            "tokens": [m.token.text for m in metas],
            "feature_ids": [list(m.feature_ids) for m in metas],
            "role_ids": role_ids,
            "roles": [ROLES[r] for r in role_ids],
            "role_probs_top1": [round(float(role_probs[i, role_ids[i]].item()), 5) for i in range(n)],
            "boundary_probs": [round(float(p), 5) for p in bound_probs],
            "fwd_gate_mean": [round(float(out["fwd_gate"][0, i].mean().item()), 5) for i in range(n)],
            "bwd_gate_mean": [round(float(out["bwd_gate"][0, i].mean().item()), 5) for i in range(n)],
            "clauses": [c.to_dict() for c in clauses],
            "baseline_clauses": [c.to_dict() for c in baseline_parse(query, schema)],
        })
    return traces


def snap_model_to_quantized_weights(model: QueryTagger) -> None:
    """Snap model parameters in-place to their exact 6-bit dequantized float32 values."""
    with torch.no_grad():
        model.embed_weight[0].zero_()
        for name, param in model.named_parameters():
            q_vals, scale = quantize_tensor_int6(param.data, model.quantization_bits)
            deq = torch.tensor(q_vals, dtype=torch.float32, device=param.device).reshape(param.shape) * scale
            if name == "embed_weight":
                deq[0].zero_()
            param.copy_(deq)


def main() -> None:
    t0 = time.perf_counter()
    assert_disjoint_schemas(TRAIN_SCHEMAS, TRANSFER_SCHEMAS)

    torch.manual_seed(42)
    rng = random.Random(42)

    # Generate 24,000 training samples strictly on TRAIN_SCHEMAS
    num_train = 24000
    train_samples = [synthesize_sample(TRAIN_SCHEMAS, rng, messy_prob=0.55) for _ in range(num_train)]

    # Generate evaluation splits (held-out train schemas vs disjoint unseen transfer schemas)
    eval_rng = random.Random(1337)
    held_out_clean = [synthesize_sample(TRAIN_SCHEMAS, eval_rng, messy_prob=0.0) for _ in range(500)]
    held_out_messy = [synthesize_sample(TRAIN_SCHEMAS, eval_rng, messy_prob=1.0) for _ in range(500)]
    transfer_clean = [synthesize_sample(TRANSFER_SCHEMAS, eval_rng, messy_prob=0.0) for _ in range(500)]
    transfer_messy = [synthesize_sample(TRANSFER_SCHEMAS, eval_rng, messy_prob=1.0) for _ in range(500)]

    # Single-thread CPU avoids OpenMP spinlock contention on tiny 32-dim scan tensors (~8x faster)
    torch.set_num_threads(1)
    mps_available = bool(torch.backends.mps.is_available())
    device = torch.device("cpu")

    model = QueryTagger().to(device)
    param_count = model.count_parameters()

    ds_train = make_tensor_dataset(train_samples, device)
    batch_size = 800
    epochs = 25
    steps_per_epoch = (num_train + batch_size - 1) // batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=7.0e-3,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
    )

    train_start = time.perf_counter()
    epoch_log: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        total_loss = 0.0
        for b_idx in range(steps_per_epoch):
            idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
            f_ids = ds_train["feat_ids"][idx]
            r_tgt = ds_train["role_targets"][idx]
            b_tgt = ds_train["bound_targets"][idx]
            mask = ds_train["mask"][idx]

            optimizer.zero_grad(set_to_none=True)
            out = model(f_ids, mask)

            active = mask.reshape(-1) > 0.5
            role_loss = F.cross_entropy(
                out["role_logits"].reshape(-1, NUM_ROLES)[active],
                r_tgt.reshape(-1)[active],
            )
            bound_loss = F.binary_cross_entropy_with_logits(
                out["boundary_logits"].reshape(-1)[active],
                b_tgt.reshape(-1)[active],
            )
            loss = role_loss + 0.5 * bound_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / steps_per_epoch
        if epoch in {1, 5, 10, 15, 20, epochs}:
            epoch_log.append({"epoch": epoch, "loss": round(avg_loss, 5)})
            print(f"Epoch {epoch:02d}/{epochs:02d} | loss={avg_loss:.5f} | elapsed={time.perf_counter() - train_start:.1f}s")

    # Snap weights to exact 6-bit quantized grid before final evaluation & export
    snap_model_to_quantized_weights(model)
    weights_payload = model.export_quantized_weights()
    parity_traces = build_parity_traces(model, device)
    weights_payload["parity_traces"] = parity_traces

    eval_held_clean = evaluate_split(model, held_out_clean, device)
    eval_held_messy = evaluate_split(model, held_out_messy, device)
    eval_transfer_clean = evaluate_split(model, transfer_clean, device)
    eval_transfer_messy = evaluate_split(model, transfer_messy, device)
    eval_centroids = evaluate_personal_centroids(model, device)

    total_time_s = round(time.perf_counter() - t0, 2)
    train_time_s = round(time.perf_counter() - train_start, 2)

    root_dir = Path(__file__).resolve().parent.parent
    engine_dir = root_dir / "src" / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    weights_path = engine_dir / "weights.json"
    weights_json_str = json.dumps(weights_payload, separators=(",", ":"))
    weights_path.write_text(weights_json_str, encoding="utf-8")
    weights_file_bytes = weights_path.stat().st_size

    benchmark_payload = {
        "model": {
            "name": "Sieve QueryTagger (Bidirectional Gated-Affine-Scan)",
            "hidden_dim": model.hidden,
            "param_count": param_count,
            "quantization_bits": QUANTIZATION_BITS,
            "packed_int6_bytes": weights_payload["packed_int6_bytes"],
            "packed_int6_kib": round(weights_payload["packed_int6_bytes"] / 1024.0, 2),
            "weights_json_bytes": weights_file_bytes,
            "weights_json_kib": round(weights_file_bytes / 1024.0, 2),
            "mps_available": mps_available,
            "train_samples": num_train,
            "epochs": epochs,
            "train_time_seconds": train_time_s,
            "total_pipeline_seconds": total_time_s,
            "epoch_log": epoch_log,
        },
        "schema_disjointness_verified": True,
        "train_schemas": [s.name for s in TRAIN_SCHEMAS],
        "transfer_schemas": [s.name for s in TRANSFER_SCHEMAS],
        "arena_3way": {
            "held_out_clean": eval_held_clean,
            "held_out_messy": eval_held_messy,
            "transfer_unseen_clean": eval_transfer_clean,
            "transfer_unseen_messy": eval_transfer_messy,
            "personal_centroids": eval_centroids,
        },
        "parity_traces": parity_traces,
    }

    bench_path = Path(__file__).resolve().parent / "benchmark_results.json"
    bench_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")

    print("\n=== Sieve Training & 3-Way Evaluation Complete ===")
    print(f"Parameters:          {param_count:,} ({weights_payload['packed_int6_bytes']:,} B packed int6 / {weights_file_bytes:,} B weights.json)")
    print(f"Training time:       {train_time_s:.2f}s (total {total_time_s:.2f}s, MPS available={mps_available})")
    print(f"Held-out Clean AST:  Neural={eval_held_clean['neural_ast_accuracy']*100:.2f}% vs Rule={eval_held_clean['baseline_ast_accuracy']*100:.2f}%")
    print(f"Held-out Messy AST:  Neural={eval_held_messy['neural_ast_accuracy']*100:.2f}% vs Rule={eval_held_messy['baseline_ast_accuracy']*100:.2f}%")
    print(f"Transfer Clean AST:  Neural={eval_transfer_clean['neural_ast_accuracy']*100:.2f}% vs Rule={eval_transfer_clean['baseline_ast_accuracy']*100:.2f}%")
    print(f"Transfer Messy AST:  Neural={eval_transfer_messy['neural_ast_accuracy']*100:.2f}% vs Rule={eval_transfer_messy['baseline_ast_accuracy']*100:.2f}%")
    print(f"Personal Centroids:  Centroid={eval_centroids['centroid_accuracy']*100:.2f}% (specificity={eval_centroids['rejection_specificity']*100:.2f}%) vs Neural={eval_centroids['neural_only_accuracy']*100:.2f}% vs Rule={eval_centroids['baseline_rule_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
