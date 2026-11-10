export type FieldKind = "BOOL" | "ENUM" | "DATE" | "NUMBER" | "TEXT";
export type SchemaSplit = "train" | "transfer";

export interface FieldDef {
  readonly name: string;
  readonly sourceKey?: string;
  readonly kind: FieldKind;
  readonly aliases: readonly string[];
  readonly enumValues: readonly string[];
}

export interface SchemaDef {
  readonly name: string;
  readonly description: string;
  readonly split: SchemaSplit;
  readonly fields: readonly FieldDef[];
}

export interface FilterClause {
  field: string;
  cmp: "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "contains" | "before" | "after";
  value: string;
  negated: boolean;
}

export const TRAIN_SCHEMAS: readonly SchemaDef[] = [
  {
    name: "issues",
    description: "Engineering issue tracker",
    split: "train",
    fields: [
      { name: "status", kind: "ENUM", aliases: ["state", "stage"], enumValues: ["open", "closed", "triaged", "blocked"] },
      { name: "priority", kind: "ENUM", aliases: ["prio", "rank"], enumValues: ["p0", "p1", "p2", "p3", "urgent", "routine"] },
      { name: "assignee", kind: "TEXT", aliases: ["owner", "assigned", "worker"], enumValues: [] },
      { name: "labels", kind: "ENUM", aliases: ["tag", "category", "kind"], enumValues: ["bug", "feature", "perf", "docs", "chore"] },
      { name: "points", kind: "NUMBER", aliases: ["estimate", "effort", "storypoints"], enumValues: [] },
      { name: "created", kind: "DATE", aliases: ["opened", "filed", "logged"], enumValues: [] },
      { name: "locked", kind: "BOOL", aliases: ["frozen", "archived"], enumValues: [] },
    ],
  },
  {
    name: "mail",
    description: "Email client messages",
    split: "train",
    fields: [
      { name: "folder", kind: "ENUM", aliases: ["mailbox", "box"], enumValues: ["inbox", "sent", "drafts", "trash", "spam"] },
      { name: "sender", kind: "TEXT", aliases: ["author", "writer", "originator"], enumValues: [] },
      { name: "subject", kind: "TEXT", aliases: ["topic", "headline", "title"], enumValues: [] },
      { name: "attachment", kind: "ENUM", aliases: ["attachment_type", "payload"], enumValues: ["pdf", "csv", "slides", "zip", "invoice"] },
      { name: "size_kb", kind: "NUMBER", aliases: ["weight", "kilobytes", "bytes"], enumValues: [] },
      { name: "received", kind: "DATE", aliases: ["arrived", "delivered", "stamped"], enumValues: [] },
      { name: "unread", kind: "BOOL", aliases: ["unseen", "fresh"], enumValues: [] },
    ],
  },
  {
    name: "files",
    description: "Repository file index",
    split: "train",
    fields: [
      { name: "extension", kind: "ENUM", aliases: ["ext", "filetype", "format"], enumValues: ["ts", "py", "rs", "md", "json", "wgsl"] },
      { name: "directory", kind: "TEXT", aliases: ["folder_path", "dir", "location"], enumValues: [] },
      { name: "lines", kind: "NUMBER", aliases: ["loc", "linecount", "length"], enumValues: [] },
      { name: "modified", kind: "DATE", aliases: ["touched", "edited", "saved"], enumValues: [] },
      { name: "readonly", kind: "BOOL", aliases: ["immutable", "protected"], enumValues: [] },
    ],
  },
  {
    name: "commits",
    description: "Git commit history",
    split: "train",
    fields: [
      { name: "branch", kind: "ENUM", aliases: ["ref", "stream"], enumValues: ["main", "release", "canary", "hotfix", "staging"] },
      { name: "committer", kind: "TEXT", aliases: ["dev", "contributor", "coder"], enumValues: [] },
      { name: "additions", kind: "NUMBER", aliases: ["insertions", "added_lines", "plus"], enumValues: [] },
      { name: "authored", kind: "DATE", aliases: ["committed", "pushed", "merged_at"], enumValues: [] },
      { name: "verified", kind: "BOOL", aliases: ["signed", "gpg"], enumValues: [] },
    ],
  },
];

export const TRANSFER_SCHEMAS: readonly SchemaDef[] = [
  {
    name: "incidents",
    description: "Production incident response",
    split: "transfer",
    fields: [
      { name: "severity", kind: "ENUM", aliases: ["sev", "impact", "tier"], enumValues: ["sev1", "sev2", "sev3", "critical", "major", "minor"] },
      { name: "service", kind: "ENUM", aliases: ["subsystem", "component", "microservice"], enumValues: ["auth", "billing", "gateway", "search", "storage", "checkout"] },
      { name: "commander", kind: "TEXT", aliases: ["lead", "oncall", "responder"], enumValues: [] },
      { name: "duration_min", kind: "NUMBER", aliases: ["downtime", "outage_mins", "elapsed"], enumValues: [] },
      { name: "detected", kind: "DATE", aliases: ["triggered", "paged", "started"], enumValues: [] },
      { name: "mitigated", kind: "BOOL", aliases: ["resolved", "contained", "healed"], enumValues: [] },
    ],
  },
  {
    name: "deployments",
    description: "Cloud deployment telemetry",
    split: "transfer",
    fields: [
      { name: "environment", kind: "ENUM", aliases: ["env", "region", "cluster"], enumValues: ["prod", "preview", "sandbox", "eu_west", "us_east"] },
      { name: "outcome", kind: "ENUM", aliases: ["result", "health", "verdict"], enumValues: ["succeeded", "rolled_back", "aborted", "pending", "running"] },
      { name: "initiator", kind: "TEXT", aliases: ["deployer", "triggered_by", "actor"], enumValues: [] },
      { name: "latency_ms", kind: "NUMBER", aliases: ["duration", "p99", "milliseconds"], enumValues: [] },
      { name: "deployed", kind: "DATE", aliases: ["shipped", "released", "rolled_out"], enumValues: [] },
      { name: "canary_enabled", kind: "BOOL", aliases: ["progressive", "shadow"], enumValues: [] },
    ],
  },
  {
    name: "experiments",
    description: "ML training experiment tracker",
    split: "transfer",
    fields: [
      { name: "framework", kind: "ENUM", aliases: ["backend", "runtime", "engine"], enumValues: ["pytorch", "jax", "mlx", "triton", "onnx"] },
      { name: "metric_status", kind: "ENUM", aliases: ["phase", "run_state", "disposition"], enumValues: ["converged", "diverged", "pruned", "queued", "training"] },
      { name: "researcher", kind: "TEXT", aliases: ["scientist", "runner", "experimenter"], enumValues: [] },
      { name: "perplexity", kind: "NUMBER", aliases: ["ppl", "loss", "score"], enumValues: [] },
      { name: "launched", kind: "DATE", aliases: ["started_on", "queued_at", "scheduled"], enumValues: [] },
      { name: "quantized", kind: "BOOL", aliases: ["int6", "compressed", "lowbit"], enumValues: [] },
    ],
  },
  {
    name: "contacts",
    description: "CRM customer directory",
    split: "transfer",
    fields: [
      { name: "segment", kind: "ENUM", aliases: ["cohort", "plan", "vertical"], enumValues: ["enterprise", "midmarket", "startup", "agency", "academic"] },
      { name: "lifecycle", kind: "ENUM", aliases: ["funnel", "relationship", "standing"], enumValues: ["prospect", "qualified", "customer", "churned", "partner"] },
      { name: "account_exec", kind: "TEXT", aliases: ["rep", "salesperson", "advisor"], enumValues: [] },
      { name: "arr_usd", kind: "NUMBER", aliases: ["revenue", "contract_value", "spend", "budget"], enumValues: [] },
      { name: "contacted", kind: "DATE", aliases: ["reached", "emailed", "called"], enumValues: [] },
      { name: "subscribed", kind: "BOOL", aliases: ["opted_in", "active_member", "newsletter"], enumValues: [] },
    ],
  },
  {
    name: "client_work",
    description: "Agency client requests & revision chaos",
    split: "transfer",
    fields: [
      { name: "vibe", kind: "ENUM", aliases: ["mood", "tone", "energy", "attitude"], enumValues: ["chill", "passive_aggressive", "panicking", "delusional", "ghosting"] },
      { name: "ask", kind: "ENUM", aliases: ["deliverable", "scope", "request", "thing"], enumValues: ["logo_bigger", "pop_more", "ai_pivot", "weekend_rush", "quick_tweak"] },
      { name: "invoice_state", kind: "ENUM", aliases: ["payment", "bill", "retainer"], enumValues: ["paid", "overdue", "ghosted", "disputed", "net90"] },
      { name: "revisions", kind: "NUMBER", aliases: ["rounds", "iterations", "versions", "rewrites"], enumValues: [] },
      { name: "founder", kind: "TEXT", aliases: ["client", "ceo", "stakeholder"], enumValues: [] },
      { name: "due", kind: "DATE", aliases: ["deadline", "launch_date", "promised"], enumValues: [] },
      { name: "deposit_cleared", kind: "BOOL", aliases: ["escrow", "upfront_paid", "funded"], enumValues: [] },
    ],
  },
];

export const ALL_SCHEMAS: Record<string, SchemaDef> = Object.fromEntries(
  [...TRAIN_SCHEMAS, ...TRANSFER_SCHEMAS].map((s) => [s.name, s]),
);

export function assertDisjointSchemas(
  trainSchemas: readonly SchemaDef[] = TRAIN_SCHEMAS,
  transferSchemas: readonly SchemaDef[] = TRANSFER_SCHEMAS,
): void {
  const collect = (schemas: readonly SchemaDef[]): Set<string> => {
    const terms = new Set<string>();
    for (const s of schemas) {
      for (const f of s.fields) {
        terms.add(f.name.toLowerCase());
        for (const a of f.aliases) terms.add(a.toLowerCase());
        for (const v of f.enumValues) terms.add(v.toLowerCase());
      }
    }
    return terms;
  };
  const trainTerms = collect(trainSchemas);
  const transferTerms = collect(transferSchemas);
  const overlap: string[] = [];
  for (const t of transferTerms) {
    if (trainTerms.has(t)) overlap.push(t);
  }
  if (overlap.length > 0) {
    throw new Error(`Vocabulary overlap between train and transfer schemas: ${overlap.join(", ")}`);
  }
}

assertDisjointSchemas();
