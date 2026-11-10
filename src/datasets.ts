import { FilterClause } from "./engine/schema";
import { isIsoDate } from "./engine/autoSchema";

export interface DatasetRow {
  id: string;
  title: string;
  subtitle: string;
  fields: Record<string, string | number | boolean>;
  badges: Array<{ label: string; tone: "warm" | "blue" | "green" | "red" | "neutral" }>;
  rightMeta: string;
  rightSub: string;
}

export interface DatasetPreset {
  schemaName: string;
  tabLabel: string;
  tagline: string;
  placeholder: string;
  sampleQueries: Array<{ label: string; query: string }>;
  defaultShortcuts: Array<{
    id: string;
    name: string;
    examples: string[];
    clauses: FilterClause[];
    description: string;
  }>;
  rows: DatasetRow[];
}

export const DATASET_PRESETS: Record<string, DatasetPreset> = {
  client_work: {
    schemaName: "client_work",
    tabLabel: "Agency Client Chaos",
    tagline: "Filter passive-aggressive scope creep, overdue retainers, and 'make the logo bigger' emergencies.",
    placeholder: "Try: panicking vibe and revisions over 3 and not paid...",
    sampleQueries: [
      {
        label: "Delusional AI pivots with >4 revisions not paid",
        query: "vibe is delusional and ask is ai_pivot and revisions over 4 and not paid",
      },
      {
        label: "Passive-aggressive 'make logo bigger' requests",
        query: "vibe is passive_aggressive and ask is logo_bigger and revisions over 2",
      },
      {
        label: "Slang shortcut: 'make it pop before friday'",
        query: "client wants to make it pop more asap",
      },
      {
        label: "Hybrid Slang + Math: 'make it pop and revisions over 4 not paid'",
        query: "make it pop and revisions over 4 not paid",
      },
      {
        label: "Rejection demo: unknown word ('banana')",
        query: "overdue invoice_state with weekend_rush banana",
      },
    ],
    defaultShortcuts: [
      {
        id: "make_it_pop",
        name: "\"make it pop\"",
        examples: [
          "make it pop before friday",
          "client wants to make it pop more asap",
          "apple style pop",
          "needs more pop",
        ],
        clauses: [
          { field: "ask", cmp: "eq", value: "pop_more", negated: false },
          { field: "vibe", cmp: "eq", value: "delusional", negated: false },
        ],
        description: "ask = pop_more & vibe = delusional",
      },
      {
        id: "red_alert_creep",
        name: "\"scope creep nightmare\"",
        examples: [
          "scope creep nightmare",
          "unpaid revision hell",
          "ghosted invoice still asking for changes",
        ],
        clauses: [
          { field: "invoice_state", cmp: "eq", value: "overdue", negated: false },
          { field: "revisions", cmp: "gt", value: "5", negated: false },
        ],
        description: "invoice_state = overdue & revisions > 5",
      },
    ],
    rows: [
      {
        id: "CL-101",
        title: "\"Can we make the logo 40% bigger and also add a ChatGPT wrapper by Monday?\"",
        subtitle: "Founder: Chad (Stealth Pet-Tech) · Due: 2026-11-09",
        fields: {
          vibe: "delusional",
          ask: "ai_pivot",
          invoice_state: "overdue",
          revisions: 9,
          founder: "chad",
          due: "2026-11-09",
          deposit_cleared: false,
        },
        badges: [
          { label: "Vibe: Delusional", tone: "warm" },
          { label: "Ask: AI Pivot", tone: "blue" },
          { label: "Invoice: Overdue", tone: "red" },
        ],
        rightMeta: "9 revisions",
        rightSub: "Deposit: Unpaid",
      },
      {
        id: "CL-102",
        title: "\"Just a quick 5-minute tweak: my wife thinks the beige doesn't 'pop' enough.\"",
        subtitle: "Founder: Derek (Artisanal Cold Brew) · Due: 2026-11-10",
        fields: {
          vibe: "delusional",
          ask: "pop_more",
          invoice_state: "overdue",
          revisions: 7,
          founder: "derek",
          due: "2026-11-10",
          deposit_cleared: true,
        },
        badges: [
          { label: "Vibe: Delusional", tone: "warm" },
          { label: "Ask: Pop More", tone: "blue" },
          { label: "Invoice: Overdue", tone: "red" },
        ],
        rightMeta: "7 revisions",
        rightSub: "Deposit: Cleared",
      },
      {
        id: "CL-103",
        title: "\"Per my last email, the header logo still looks microscopic on my iPad Mini.\"",
        subtitle: "Founder: Brenda (B2B SaaS Compliance) · Due: 2026-11-12",
        fields: {
          vibe: "passive_aggressive",
          ask: "logo_bigger",
          invoice_state: "paid",
          revisions: 5,
          founder: "brenda",
          due: "2026-11-12",
          deposit_cleared: true,
        },
        badges: [
          { label: "Vibe: Passive-Aggressive", tone: "warm" },
          { label: "Ask: Logo Bigger", tone: "blue" },
          { label: "Invoice: Paid", tone: "green" },
        ],
        rightMeta: "5 revisions",
        rightSub: "Deposit: Cleared",
      },
      {
        id: "CL-104",
        title: "\"URGENT: Our YC interview is in 3 hours, need a complete WebGPU pivot right now\"",
        subtitle: "Founder: Toby (Crypto-for-Dogs) · Due: 2026-11-07",
        fields: {
          vibe: "panicking",
          ask: "ai_pivot",
          invoice_state: "ghosted",
          revisions: 11,
          founder: "toby",
          due: "2026-11-07",
          deposit_cleared: false,
        },
        badges: [
          { label: "Vibe: Panicking", tone: "red" },
          { label: "Ask: AI Pivot", tone: "blue" },
          { label: "Invoice: Ghosted", tone: "red" },
        ],
        rightMeta: "11 revisions",
        rightSub: "Deposit: Unpaid",
      },
      {
        id: "CL-105",
        title: "\"Hey! Sending over the Saturday night Figma comments—should only take an hour!\"",
        subtitle: "Founder: Gavin (Luxury Pickleball) · Due: 2026-11-08",
        fields: {
          vibe: "passive_aggressive",
          ask: "weekend_rush",
          invoice_state: "overdue",
          revisions: 6,
          founder: "gavin",
          due: "2026-11-08",
          deposit_cleared: false,
        },
        badges: [
          { label: "Vibe: Passive-Aggressive", tone: "warm" },
          { label: "Ask: Weekend Rush", tone: "blue" },
          { label: "Invoice: Overdue", tone: "red" },
        ],
        rightMeta: "6 revisions",
        rightSub: "Deposit: Unpaid",
      },
      {
        id: "CL-106",
        title: "\"Looks awesome, approved as-is! Wire transfer for the milestone just went out.\"",
        subtitle: "Founder: Maya (Developer Tools) · Due: 2026-11-15",
        fields: {
          vibe: "chill",
          ask: "quick_tweak",
          invoice_state: "paid",
          revisions: 1,
          founder: "maya",
          due: "2026-11-15",
          deposit_cleared: true,
        },
        badges: [
          { label: "Vibe: Chill", tone: "green" },
          { label: "Ask: Quick Tweak", tone: "neutral" },
          { label: "Invoice: Paid", tone: "green" },
        ],
        rightMeta: "1 revision",
        rightSub: "Deposit: Cleared",
      },
      {
        id: "CL-107",
        title: "\"Can we make the wordmark 2x wider so it fills the whole navbar?\"",
        subtitle: "Founder: Arthur (LegalTech) · Due: 2026-11-11",
        fields: {
          vibe: "passive_aggressive",
          ask: "logo_bigger",
          invoice_state: "overdue",
          revisions: 4,
          founder: "arthur",
          due: "2026-11-11",
          deposit_cleared: true,
        },
        badges: [
          { label: "Vibe: Passive-Aggressive", tone: "warm" },
          { label: "Ask: Logo Bigger", tone: "blue" },
          { label: "Invoice: Overdue", tone: "red" },
        ],
        rightMeta: "4 revisions",
        rightSub: "Deposit: Cleared",
      },
    ],
  },
  incidents: {
    schemaName: "incidents",
    tabLabel: "Production Incidents",
    tagline: "Zero-shot transfer onto an unseen DevOps incident schema (0 shared words with training).",
    placeholder: "Try: critical on auth where duration_min over 15 and not mitigated...",
    sampleQueries: [
      {
        label: "Critical auth outages > 15m not mitigated",
        query: "show me critical on auth where duration_min over 15 and not mitigated",
      },
      {
        label: "Billing sev1 incidents",
        query: "service is billing and severity is sev1",
      },
      {
        label: "Slang shortcut: 'p0 fires'",
        query: "current p0 fires right now",
      },
    ],
    defaultShortcuts: [
      {
        id: "p0_fires",
        name: "\"p0 fires\"",
        examples: ["p0 fires", "current p0 fires right now", "active p0 fire"],
        clauses: [
          { field: "severity", cmp: "eq", value: "critical", negated: false },
          { field: "mitigated", cmp: "eq", value: "true", negated: true },
        ],
        description: "severity = critical & NOT mitigated",
      },
    ],
    rows: [
      {
        id: "INC-4109",
        title: "OAuth token rotation lock timeout locking out mobile sessions",
        subtitle: "Commander: Marcus · Triggered: 2026-11-07",
        fields: {
          severity: "critical",
          service: "auth",
          commander: "marcus",
          duration_min: 42,
          detected: "2026-11-07",
          mitigated: false,
        },
        badges: [
          { label: "Severity: Critical", tone: "red" },
          { label: "Service: Auth", tone: "blue" },
        ],
        rightMeta: "42 mins",
        rightSub: "Unmitigated",
      },
      {
        id: "INC-4098",
        title: "Stripe webhook retry storm delaying invoice receipts",
        subtitle: "Commander: Elena · Triggered: 2026-11-07",
        fields: {
          severity: "sev1",
          service: "billing",
          commander: "elena",
          duration_min: 27,
          detected: "2026-11-07",
          mitigated: false,
        },
        badges: [
          { label: "Severity: Sev1", tone: "warm" },
          { label: "Service: Billing", tone: "blue" },
        ],
        rightMeta: "27 mins",
        rightSub: "Unmitigated",
      },
      {
        id: "INC-4082",
        title: "SAML assertion verification spike on EU gateway",
        subtitle: "Commander: Priya · Triggered: 2026-11-06",
        fields: {
          severity: "critical",
          service: "auth",
          commander: "priya",
          duration_min: 64,
          detected: "2026-11-06",
          mitigated: false,
        },
        badges: [
          { label: "Severity: Critical", tone: "red" },
          { label: "Service: Auth", tone: "blue" },
        ],
        rightMeta: "64 mins",
        rightSub: "Unmitigated",
      },
      {
        id: "INC-4071",
        title: "Search index shard rebalance completed cleanly",
        subtitle: "Commander: Sam · Triggered: 2026-11-05",
        fields: {
          severity: "minor",
          service: "search",
          commander: "sam",
          duration_min: 8,
          detected: "2026-11-05",
          mitigated: true,
        },
        badges: [
          { label: "Severity: Minor", tone: "green" },
          { label: "Service: Search", tone: "neutral" },
        ],
        rightMeta: "8 mins",
        rightSub: "Mitigated",
      },
    ],
  },
  contacts: {
    schemaName: "contacts",
    tabLabel: "CRM Leads & Accounts",
    tagline: "Zero-shot transfer onto CRM leads, ARR budgets, and lifecycle stages.",
    placeholder: "Try: enterprise segment with arr_usd over 50000 and not churned...",
    sampleQueries: [
      {
        label: "Enterprise accounts > $50k ARR",
        query: "segment is enterprise and arr_usd over 50000 and subscribed",
      },
      {
        label: "Churned startup leads",
        query: "segment is startup and lifecycle is churned",
      },
    ],
    defaultShortcuts: [
      {
        id: "whales",
        name: "\"whale accounts\"",
        examples: ["whale accounts", "big enterprise whales", "high paying whales"],
        clauses: [
          { field: "segment", cmp: "eq", value: "enterprise", negated: false },
          { field: "arr_usd", cmp: "gt", value: "50000", negated: false },
        ],
        description: "segment = enterprise & arr_usd > 50000",
      },
    ],
    rows: [
      {
        id: "CRM-01",
        title: "Acme Global Corp — Multi-seat Enterprise License",
        subtitle: "Rep: Sarah · Last Contacted: 2026-11-04",
        fields: {
          segment: "enterprise",
          lifecycle: "customer",
          account_exec: "sarah",
          arr_usd: 120000,
          contacted: "2026-11-04",
          subscribed: true,
        },
        badges: [
          { label: "Enterprise", tone: "blue" },
          { label: "Customer", tone: "green" },
        ],
        rightMeta: "$120,000 ARR",
        rightSub: "Subscribed",
      },
      {
        id: "CRM-02",
        title: "HyperScale AI Labs — Annual Platform Deal",
        subtitle: "Rep: Devon · Last Contacted: 2026-11-06",
        fields: {
          segment: "enterprise",
          lifecycle: "qualified",
          account_exec: "devon",
          arr_usd: 85000,
          contacted: "2026-11-06",
          subscribed: true,
        },
        badges: [
          { label: "Enterprise", tone: "blue" },
          { label: "Qualified", tone: "warm" },
        ],
        rightMeta: "$85,000 ARR",
        rightSub: "Subscribed",
      },
      {
        id: "CRM-03",
        title: "vibe-coder.io — Seed Stage Trial",
        subtitle: "Rep: Alex · Last Contacted: 2026-10-19",
        fields: {
          segment: "startup",
          lifecycle: "churned",
          account_exec: "alex",
          arr_usd: 2400,
          contacted: "2026-10-19",
          subscribed: false,
        },
        badges: [
          { label: "Startup", tone: "neutral" },
          { label: "Churned", tone: "red" },
        ],
        rightMeta: "$2,400 ARR",
        rightSub: "Unsubscribed",
      },
    ],
  },
};

export function filterDatasetRows(rows: DatasetRow[], clauses: FilterClause[]): DatasetRow[] {
  if (clauses.length === 0) return rows;

  return rows.filter((row) =>
    clauses.every((c) => {
      const rawVal = row.fields[c.field];
      if (rawVal === undefined || rawVal === null || rawVal === "") return false;

      let matches = false;
      if (typeof rawVal === "boolean") {
        if (c.cmp !== "eq" && c.cmp !== "neq") return false;
        if (c.value !== "true" && c.value !== "false") return false;
        const target = c.value === "true";
        matches = c.cmp === "neq" ? rawVal !== target : rawVal === target;
      } else if (typeof rawVal === "number") {
        if (!["eq", "neq", "gt", "gte", "lt", "lte"].includes(c.cmp)) return false;
        const num = Number(c.value);
        if (!Number.isFinite(num)) return false;
        if (c.cmp === "gt") matches = rawVal > num;
        else if (c.cmp === "gte") matches = rawVal >= num;
        else if (c.cmp === "lt") matches = rawVal < num;
        else if (c.cmp === "lte") matches = rawVal <= num;
        else if (c.cmp === "neq") matches = rawVal !== num;
        else matches = rawVal === num;
      } else {
        const strVal = String(rawVal).toLowerCase();
        const target = c.value.toLowerCase();
        if (!["eq", "neq", "contains", "before", "after", "gt", "gte", "lt", "lte"].includes(c.cmp)) return false;
        if (["before", "after", "gt", "gte", "lt", "lte"].includes(c.cmp) && (!isIsoDate(strVal) || !isIsoDate(target))) return false;
        if (c.cmp === "contains") matches = strVal.includes(target);
        else if (c.cmp === "neq") matches = strVal !== target;
        else if (c.cmp === "after" || c.cmp === "gt") matches = strVal > target;
        else if (c.cmp === "before" || c.cmp === "lt") matches = strVal < target;
        else if (c.cmp === "gte") matches = strVal >= target;
        else if (c.cmp === "lte") matches = strVal <= target;
        else matches = strVal === target;
      }

      return c.negated ? !matches : matches;
    }),
  );
}
