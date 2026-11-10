from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

FieldKind = Literal["BOOL", "ENUM", "DATE", "NUMBER", "TEXT"]
SchemaSplit = Literal["train", "transfer"]

FIELD_KINDS: tuple[FieldKind, ...] = ("BOOL", "ENUM", "DATE", "NUMBER", "TEXT")


@dataclass(frozen=True)
class FieldDef:
    name: str
    kind: FieldKind
    aliases: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaDef:
    name: str
    description: str
    split: SchemaSplit
    fields: tuple[FieldDef, ...] = field(default_factory=tuple)

    def field_map(self) -> dict[str, FieldDef]:
        return {f.name: f for f in self.fields}


TRAIN_SCHEMAS: tuple[SchemaDef, ...] = (
    SchemaDef(
        name="issues",
        description="Engineering issue tracker",
        split="train",
        fields=(
            FieldDef("status", "ENUM", aliases=("state", "stage"), enum_values=("open", "closed", "triaged", "blocked")),
            FieldDef("priority", "ENUM", aliases=("prio", "rank"), enum_values=("p0", "p1", "p2", "p3", "urgent", "routine")),
            FieldDef("assignee", "TEXT", aliases=("owner", "assigned", "worker")),
            FieldDef("labels", "ENUM", aliases=("tag", "category", "kind"), enum_values=("bug", "feature", "perf", "docs", "chore")),
            FieldDef("points", "NUMBER", aliases=("estimate", "effort", "storypoints")),
            FieldDef("created", "DATE", aliases=("opened", "filed", "logged")),
            FieldDef("locked", "BOOL", aliases=("frozen", "archived")),
        ),
    ),
    SchemaDef(
        name="mail",
        description="Email client messages",
        split="train",
        fields=(
            FieldDef("folder", "ENUM", aliases=("mailbox", "box"), enum_values=("inbox", "sent", "drafts", "trash", "spam")),
            FieldDef("sender", "TEXT", aliases=("author", "writer", "originator")),
            FieldDef("subject", "TEXT", aliases=("topic", "headline", "title")),
            FieldDef("attachment", "ENUM", aliases=("attachment_type", "payload"), enum_values=("pdf", "csv", "slides", "zip", "invoice")),
            FieldDef("size_kb", "NUMBER", aliases=("weight", "kilobytes", "bytes")),
            FieldDef("received", "DATE", aliases=("arrived", "delivered", "stamped")),
            FieldDef("unread", "BOOL", aliases=("unseen", "fresh")),
        ),
    ),
    SchemaDef(
        name="files",
        description="Repository file index",
        split="train",
        fields=(
            FieldDef("extension", "ENUM", aliases=("ext", "filetype", "format"), enum_values=("ts", "py", "rs", "md", "json", "wgsl")),
            FieldDef("directory", "TEXT", aliases=("folder_path", "dir", "location")),
            FieldDef("lines", "NUMBER", aliases=("loc", "linecount", "length")),
            FieldDef("modified", "DATE", aliases=("touched", "edited", "saved")),
            FieldDef("readonly", "BOOL", aliases=("immutable", "protected")),
        ),
    ),
    SchemaDef(
        name="commits",
        description="Git commit history",
        split="train",
        fields=(
            FieldDef("branch", "ENUM", aliases=("ref", "stream"), enum_values=("main", "release", "canary", "hotfix", "staging")),
            FieldDef("committer", "TEXT", aliases=("dev", "contributor", "coder")),
            FieldDef("additions", "NUMBER", aliases=("insertions", "added_lines", "plus")),
            FieldDef("authored", "DATE", aliases=("committed", "pushed", "merged_at")),
            FieldDef("verified", "BOOL", aliases=("signed", "gpg")),
        ),
    ),
)

TRANSFER_SCHEMAS: tuple[SchemaDef, ...] = (
    SchemaDef(
        name="incidents",
        description="Production incident response",
        split="transfer",
        fields=(
            FieldDef("severity", "ENUM", aliases=("sev", "impact", "tier"), enum_values=("sev1", "sev2", "sev3", "critical", "major", "minor")),
            FieldDef("service", "ENUM", aliases=("subsystem", "component", "microservice"), enum_values=("auth", "billing", "gateway", "search", "storage", "checkout")),
            FieldDef("commander", "TEXT", aliases=("lead", "oncall", "responder")),
            FieldDef("duration_min", "NUMBER", aliases=("downtime", "outage_mins", "elapsed")),
            FieldDef("detected", "DATE", aliases=("triggered", "paged", "started")),
            FieldDef("mitigated", "BOOL", aliases=("resolved", "contained", "healed")),
        ),
    ),
    SchemaDef(
        name="deployments",
        description="Cloud deployment telemetry",
        split="transfer",
        fields=(
            FieldDef("environment", "ENUM", aliases=("env", "region", "cluster"), enum_values=("prod", "preview", "sandbox", "eu_west", "us_east")),
            FieldDef("outcome", "ENUM", aliases=("result", "health", "verdict"), enum_values=("succeeded", "rolled_back", "aborted", "pending", "running")),
            FieldDef("initiator", "TEXT", aliases=("deployer", "triggered_by", "actor")),
            FieldDef("latency_ms", "NUMBER", aliases=("duration", "p99", "milliseconds")),
            FieldDef("deployed", "DATE", aliases=("shipped", "released", "rolled_out")),
            FieldDef("canary_enabled", "BOOL", aliases=("progressive", "shadow")),
        ),
    ),
    SchemaDef(
        name="experiments",
        description="ML training experiment tracker",
        split="transfer",
        fields=(
            FieldDef("framework", "ENUM", aliases=("backend", "runtime", "engine"), enum_values=("pytorch", "jax", "mlx", "triton", "onnx")),
            FieldDef("metric_status", "ENUM", aliases=("phase", "run_state", "disposition"), enum_values=("converged", "diverged", "pruned", "queued", "training")),
            FieldDef("researcher", "TEXT", aliases=("scientist", "runner", "experimenter")),
            FieldDef("perplexity", "NUMBER", aliases=("ppl", "loss", "score")),
            FieldDef("launched", "DATE", aliases=("started_on", "queued_at", "scheduled")),
            FieldDef("quantized", "BOOL", aliases=("int6", "compressed", "lowbit")),
        ),
    ),
    SchemaDef(
        name="contacts",
        description="CRM customer directory",
        split="transfer",
        fields=(
            FieldDef("segment", "ENUM", aliases=("cohort", "plan", "vertical"), enum_values=("enterprise", "midmarket", "startup", "agency", "academic")),
            FieldDef("lifecycle", "ENUM", aliases=("funnel", "relationship", "standing"), enum_values=("prospect", "qualified", "customer", "churned", "partner")),
            FieldDef("account_exec", "TEXT", aliases=("rep", "salesperson", "advisor")),
            FieldDef("arr_usd", "NUMBER", aliases=("revenue", "contract_value", "spend", "budget")),
            FieldDef("contacted", "DATE", aliases=("reached", "emailed", "called")),
            FieldDef("subscribed", "BOOL", aliases=("opted_in", "active_member", "newsletter")),
        ),
    ),
    SchemaDef(
        name="client_work",
        description="Agency client requests & revision chaos",
        split="transfer",
        fields=(
            FieldDef("vibe", "ENUM", aliases=("mood", "tone", "energy", "attitude"), enum_values=("chill", "passive_aggressive", "panicking", "delusional", "ghosting")),
            FieldDef("ask", "ENUM", aliases=("deliverable", "scope", "request", "thing"), enum_values=("logo_bigger", "pop_more", "ai_pivot", "weekend_rush", "quick_tweak")),
            FieldDef("invoice_state", "ENUM", aliases=("payment", "bill", "retainer"), enum_values=("paid", "overdue", "ghosted", "disputed", "net90")),
            FieldDef("revisions", "NUMBER", aliases=("rounds", "iterations", "versions", "rewrites"), enum_values=()),
            FieldDef("founder", "TEXT", aliases=("client", "ceo", "stakeholder"), enum_values=()),
            FieldDef("due", "DATE", aliases=("deadline", "launch_date", "promised"), enum_values=()),
            FieldDef("deposit_cleared", "BOOL", aliases=("escrow", "upfront_paid", "funded"), enum_values=()),
        ),
    ),
)

ALL_SCHEMAS: dict[str, SchemaDef] = {
    s.name: s for s in (*TRAIN_SCHEMAS, *TRANSFER_SCHEMAS)
}


def collect_schema_vocabulary(schemas: tuple[SchemaDef, ...]) -> tuple[set[str], set[str], set[str]]:
    field_names: set[str] = set()
    aliases: set[str] = set()
    enum_values: set[str] = set()
    for schema in schemas:
        for f in schema.fields:
            field_names.add(f.name.lower())
            for a in f.aliases:
                aliases.add(a.lower())
            for v in f.enum_values:
                enum_values.add(v.lower())
    return field_names, aliases, enum_values


def assert_disjoint_schemas(
    train_schemas: tuple[SchemaDef, ...] = TRAIN_SCHEMAS,
    transfer_schemas: tuple[SchemaDef, ...] = TRANSFER_SCHEMAS,
) -> None:
    """Assert zero field names, aliases, or enum values overlap between train and transfer eval."""
    train_fields, train_aliases, train_enums = collect_schema_vocabulary(train_schemas)
    eval_fields, eval_aliases, eval_enums = collect_schema_vocabulary(transfer_schemas)

    train_all_terms = train_fields | train_aliases | train_enums
    eval_all_terms = eval_fields | eval_aliases | eval_enums

    field_overlap = train_fields & eval_fields
    alias_overlap = train_aliases & eval_aliases
    enum_overlap = train_enums & eval_enums
    cross_overlap = train_all_terms & eval_all_terms

    assert not field_overlap, f"Field overlap between train and transfer: {field_overlap}"
    assert not alias_overlap, f"Alias overlap between train and transfer: {alias_overlap}"
    assert not enum_overlap, f"Enum overlap between train and transfer: {enum_overlap}"
    assert not cross_overlap, f"Cross-vocabulary overlap between train and transfer: {cross_overlap}"


def schemas_to_dict() -> dict[str, object]:
    return {
        "train": [
            {
                "name": s.name,
                "description": s.description,
                "split": s.split,
                "fields": [
                    {
                        "name": f.name,
                        "kind": f.kind,
                        "aliases": list(f.aliases),
                        "enum_values": list(f.enum_values),
                    }
                    for f in s.fields
                ],
            }
            for s in TRAIN_SCHEMAS
        ],
        "transfer": [
            {
                "name": s.name,
                "description": s.description,
                "split": s.split,
                "fields": [
                    {
                        "name": f.name,
                        "kind": f.kind,
                        "aliases": list(f.aliases),
                        "enum_values": list(f.enum_values),
                    }
                    for f in s.fields
                ],
            }
            for s in TRANSFER_SCHEMAS
        ],
    }


assert_disjoint_schemas()
