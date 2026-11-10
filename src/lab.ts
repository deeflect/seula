import "./styles.css";
import { DATASET_PRESETS, DatasetPreset, DatasetRow, filterDatasetRows } from "./datasets";
import {
  ALL_SCHEMAS,
  CentroidMemory,
  FilterClause,
  SchemaDef,
  SieveModel,
  SieveWebGPUBatchEngine,
  baselineParse,
  inferSchemaFromRows,
  parseCsvOrJsonToRows,
  normalizeImportedRow,
} from "./engine";
import rawWeightsJson from "./engine/weights.json";

const model = new SieveModel(rawWeightsJson as any);
const batchEngine = new SieveWebGPUBatchEngine(model);
const centroidMemory = new CentroidMemory(96, 0.58);
const dynamicSchemas: Record<string, SchemaDef> = { ...ALL_SCHEMAS };
const dynamicPresets: Record<string, DatasetPreset> = { ...DATASET_PRESETS };

// Pre-seed default personal shortcuts across all datasets
for (const preset of Object.values(dynamicPresets)) {
  const schema = dynamicSchemas[preset.schemaName];
  if (!schema) continue;
  for (const sc of preset.defaultShortcuts) {
    centroidMemory.teach(sc.id, schema, sc.examples, sc.clauses);
  }
}

const DEFAULT_CUSTOM_CSV = `company,plan,mrr,churn_risk,tickets,paid
"VibeCode AI",pro,4200,critical,19,false
"Stealth Crypto DAO",enterprise,12500,critical,31,false
"Boring Plumbing LLC",starter,450,low,1,true
"HyperGrowth Agency",pro,3800,high,14,false
"Quiet SaaS Co",enterprise,8900,low,2,true
"WrapperGPT Inc",starter,290,critical,24,false`;

interface AppState {
  activeDatasetKey: string;
  query: string;
  showTeachDrawer: boolean;
  showCustomTableDrawer: boolean;
  customTableRaw: string;
  teachName: string;
  teachExamples: string;
  teachField: string;
  teachValue: string;
  showTechXray: boolean;
  batchBenchmarkResult: {
    ran: boolean;
    backend: string;
    totalMs: number;
    cpuComparisonMs: number;
    speedup: string;
  } | null;
  importError: string;
}

const state: AppState = {
  activeDatasetKey: "client_work",
  query: "make it pop and revisions over 4 not paid",
  showTeachDrawer: false,
  showCustomTableDrawer: false,
  customTableRaw: DEFAULT_CUSTOM_CSV,
  teachName: "scope creep",
  teachExamples: "scope creep nightmare, unpaid revision hell, client ghosted invoice",
  teachField: "invoice_state",
  teachValue: "overdue",
  showTechXray: false,
  batchBenchmarkResult: null,
  importError: "",
};

function escapeHtml(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!);
}

function formatClauseHuman(c: FilterClause): string {
  const opMap: Record<FilterClause["cmp"], string> = {
    eq: "=",
    neq: "≠",
    gt: ">",
    gte: "≥",
    lt: "<",
    lte: "≤",
    contains: "contains",
    before: "before",
    after: "after",
  };
  const prefix = c.negated ? "Exclude: " : "";
  return `${prefix}${c.field} ${opMap[c.cmp] ?? "="} ${c.value}`;
}

function classifyTokenVisual(
  role: string,
  matchedField: string | null,
  matchedEnumField: string | null,
  text: string,
): { cssClass: string; badgeText: string } {
  if (role === "FIELD") return { cssClass: "field", badgeText: matchedField ? `field:${matchedField}` : "field" };
  if (role === "VALUE") return { cssClass: "value", badgeText: matchedEnumField ? `${matchedEnumField}` : "value" };
  if (role.startsWith("OP_")) return { cssClass: "op", badgeText: role.replace("OP_", "").toLowerCase() };
  if (role === "NEGATE") return { cssClass: "negate", badgeText: "not" };
  if (role === "AND") return { cssClass: "plain", badgeText: "" };

  const stopWords = new Set(["show", "me", "where", "with", "for", "on", "the", "a", "an", "please", "items", "rows", "all", "having", "that", "are"]);
  if (stopWords.has(text.toLowerCase())) {
    return { cssClass: "plain", badgeText: "" };
  }
  return { cssClass: "ignored", badgeText: "ignored" };
}

function render(): void {
  const app = document.getElementById("app");
  if (!app) return;

  const preset = dynamicPresets[state.activeDatasetKey];
  const schema = dynamicSchemas[preset.schemaName];

  // 1. Run Compositional Slang Splicer first (catches both whole-query shortcuts AND sub-span slang inside larger queries)
  const t2 = performance.now();
  const spliceRes = centroidMemory.splice(state.query, schema);
  const centroidMs = performance.now() - t2;

  // 2. Run 21.7 KiB Neural Tagger on the residual natural-language query (or full query if no slang splice)
  const t0 = performance.now();
  const queryForNeural = spliceRes.matched ? spliceRes.residualQuery : state.query;
  const neuralRes = model.parse(queryForNeural, schema);
  const neuralMs = performance.now() - t0;

  // 3. Run Regex / Rule baseline
  const t1 = performance.now();
  const ruleClauses = baselineParse(state.query, schema);
  const ruleMs = performance.now() - t1;

  // Hybrid resolution: merge spliced slang clauses + neural clauses on the residual query
  const activeClauses: FilterClause[] = spliceRes.matched
    ? [...spliceRes.splicedClauses, ...neuralRes.clauses]
    : neuralRes.clauses;

  // Deduplicate active clauses
  const seenKey = new Set<string>();
  const dedupedClauses = activeClauses.filter((c) => {
    const k = `${c.negated ? "!" : ""}${c.field}:${c.cmp}:${c.value}`;
    if (seenKey.has(k)) return false;
    seenKey.add(k);
    return true;
  });

  const rejected = neuralRes.issues.length > 0 || (!spliceRes.matched && !neuralRes.accepted);
  const matchedRows = rejected ? [] : filterDatasetRows(preset.rows, dedupedClauses);
  const ruleMatchedRows = filterDatasetRows(preset.rows, ruleClauses);

  // Identify ignored junk words (tokens in residual query that aren't stop words and got role NONE)
  const ignoredWords = neuralRes.tokens
    .filter((t) => {
      const vis = classifyTokenVisual(t.role, t.matchedField, t.matchedEnumField, t.text);
      return vis.cssClass === "ignored";
    })
    .map((t) => t.text);

  app.innerHTML = `
    <header class="topbar">
      <div class="brand-row">
        <div class="brand-mark">
          <div class="brand-dot"></div>
          <span>Sieve</span>
        </div>
        <span class="brand-tagline">Type like a human. Filter like a database.</span>
      </div>

      <div class="tabs-pill">
        ${Object.entries(dynamicPresets)
          .map(
            ([key, p]) => `
            <button class="tab-btn ${key === state.activeDatasetKey ? "active" : ""}" data-dataset="${key}">
              ${escapeHtml(p.tabLabel)} (${p.rows.length})
            </button>
          `,
          )
          .join("")}
      </div>

      <div style="display: flex; align-items: center; gap: 10px;">
        <button id="btn-open-custom-table" class="btn-secondary" style="height: 30px; font-size: 12px;">
          + Paste any CSV / JSON table
        </button>
        <span class="status-badge">Runs 100% locally · 21.7 KiB int6 model</span>
      </div>
    </header>

    <main class="page-body">
      <section class="search-card">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
          <div style="font-size: 13px; color: var(--ink-secondary);">${escapeHtml(preset.tagline)}</div>
          <div class="mono-num" style="font-size: 12px; color: var(--ink-muted);">
            Parsed in ${(neuralMs + centroidMs).toFixed(2)} ms on CPU
          </div>
        </div>

        <div class="search-input-row">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#c2410c" stroke-width="1.75">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.3-4.3"></path>
          </svg>
          <input
            id="sieve-input"
            class="search-input"
            type="text"
            value="${escapeHtml(state.query)}"
            placeholder="${escapeHtml(preset.placeholder)}"
            autocomplete="off"
          />
          <button id="btn-clear" class="btn-secondary">Clear</button>
        </div>

        <div class="highlighted-tokens-row">
          <div class="token-stream">
            <span style="font-size: 12px; color: var(--ink-muted); margin-right: 4px;">Live word reading:</span>
            ${spliceRes.hits
              .map(
                (h) => `<span class="tok-pill centroid">
                  <span>${escapeHtml(h.phrase)}</span>
                  <span class="tok-role-sub">slang:${escapeHtml(h.idiomId)} (${(h.similarity * 100).toFixed(0)}%)</span>
                </span>`,
              )
              .join("")}
            ${neuralRes.tokens
              .map((tok) => {
                const vis = classifyTokenVisual(tok.role, tok.matchedField, tok.matchedEnumField, tok.text);
                return `<span class="tok-pill ${vis.cssClass}">
                  <span>${escapeHtml(tok.text)}</span>
                  ${
                    vis.badgeText
                      ? `<span class="tok-role-sub">${escapeHtml(vis.badgeText)}${
                          state.showTechXray ? ` · ${(tok.confidence * 100).toFixed(0)}%` : ""
                        }</span>`
                      : ""
                  }
                </span>`;
              })
              .join("")}
          </div>

          <div style="display: flex; align-items: center; gap: 8px;">
            ${
              ignoredWords.length > 0
                ? `<span style="font-size: 12px; color: var(--ink-secondary); background-color: var(--paper-raised); padding: 4px 10px; border-radius: 6px;">
                    Unapplied ${ignoredWords.map((w) => `"${escapeHtml(w)}"`).join(", ")}
                  </span>`
                : ""
            }
            <button id="btn-toggle-xray" class="sample-btn">
              ${state.showTechXray ? "Hide neural X-ray" : "Show neural X-ray"}
            </button>
          </div>
        </div>

        ${
          state.showTechXray
            ? `<div style="padding: 12px 14px; border-radius: 10px; background-color: #141413; color: #efe9de; font-family: var(--font-mono); font-size: 12px; display: flex; flex-direction: column; gap: 6px;">
                <div style="display: flex; justify-content: space-between; color: #aaa398;">
                  <span>Gated-Affine-Scan Internal State (29,613 params · H=32 · zero word-identity hash)</span>
                  <span>Historical synthetic benchmark: 99.8% exact AST</span>
                </div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap; padding-top: 4px;">
                  ${neuralRes.tokens
                    .map(
                      (t) => `<div style="padding: 6px 8px; border-radius: 6px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);">
                        <div style="color: #fff; font-weight: 500;">${escapeHtml(t.text)} → ${t.role} (${(t.confidence * 100).toFixed(1)}%)</div>
                        <div style="color: #aaa398; font-size: 11px;">fwdGate=${t.fwdGateMean.toFixed(2)} bwdGate=${t.bwdGateMean.toFixed(2)} bnd=${t.boundaryProb.toFixed(2)}</div>
                      </div>`,
                    )
                    .join("")}
                </div>
              </div>`
            : ""
        }

        <div class="pills-bar">
          <div class="pills-list">
            <span style="font-size: 13px; color: var(--ink-muted);">Active filters:</span>
            ${
              spliceRes.matched
                ? `<span class="filter-chip shortcut">
                    <span>Slang "${escapeHtml(spliceRes.bestIdiomId)}": ${escapeHtml(spliceRes.splicedClauses.map(formatClauseHuman).join(" & "))}</span>
                  </span>`
                : ""
            }
            ${
              rejected
                ? `<span role="alert" style="color: var(--tag-red);">Could not apply query: ${escapeHtml(neuralRes.issues.join("; "))}</span>`
                : dedupedClauses.length === 0
                ? `<span style="font-size: 13px; color: var(--ink-muted);">None (showing all ${preset.rows.length} items)</span>`
                : dedupedClauses
                    .map(
                      (c, idx) => `
                      <span class="filter-chip ${c.negated ? "negated" : ""}" data-remove-idx="${idx}" title="Click to remove filter">
                        <span>${escapeHtml(formatClauseHuman(c))}</span>
                        <span style="opacity: 0.65; padding-left: 4px;">×</span>
                      </span>
                    `,
                    )
                    .join("")
            }
          </div>

          <button id="btn-open-teach" class="btn-warm">
            + Teach your own slang shortcut
          </button>
        </div>

        <div class="samples-row">
          <span style="font-size: 12px; color: var(--ink-muted);">Try clicking an example:</span>
          ${preset.sampleQueries
            .map(
              (sq) => `
              <button class="sample-btn" data-sample-query="${escapeHtml(sq.query)}">
                ${escapeHtml(sq.label)}
              </button>
            `,
            )
            .join("")}
        </div>
      </section>

      ${
        state.showCustomTableDrawer
          ? `<section class="teach-drawer">
              <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                  <div style="font-size: 17px; font-weight: 600; color: var(--ink-primary);">Zero-Config Auto-Schema: Paste any CSV or JSON table</div>
                  <div style="font-size: 14px; color: var(--ink-secondary);">
                    Sieve infers supported field types and aliases from the pasted rows.
                  </div>
                </div>
                <span class="badge blue">inferSchemaFromRows()</span>
              </div>
              <textarea
                id="custom-table-textarea"
                class="teach-input"
                style="height: 120px; padding: 10px 12px; font-family: var(--font-mono); font-size: 12px; line-height: 1.5;"
              >${escapeHtml(state.customTableRaw)}</textarea>
              ${state.importError ? `<div role="alert" style="color: var(--tag-red);">${escapeHtml(state.importError)}</div>` : ""}
              <div style="display: flex; justify-content: flex-end; gap: 10px;">
                <button id="btn-cancel-custom-table" class="btn-secondary">Cancel</button>
                <button id="btn-load-custom-table" class="btn-primary-solid">Infer schema &amp; filter this table now</button>
              </div>
            </section>`
          : ""
      }

      ${
        state.showTeachDrawer
          ? `<section class="teach-drawer">
              <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                  <div style="font-size: 17px; font-weight: 600; color: var(--ink-primary);">Teach Sieve your own slang in one click</div>
                  <div style="font-size: 14px; color: var(--ink-secondary);">
                    Type a phrase or inside joke you use at work, and pick which filter it turns on. Works both standalone AND mixed inside larger sentences!
                  </div>
                </div>
                <span class="badge warm">Compositional Slang Splicer · 0 gradient steps</span>
              </div>

              <div class="teach-grid">
                <div class="teach-field">
                  <label style="font-size: 13px; font-weight: 500;">When I type this phrase (comma-separated examples):</label>
                  <input id="teach-examples" class="teach-input" type="text" value="${escapeHtml(state.teachExamples)}" />
                </div>
                <div class="teach-field" style="max-width: 220px;">
                  <label style="font-size: 13px; font-weight: 500;">Set field:</label>
                  <select id="teach-field-select" class="teach-input">
                    ${schema.fields
                      .map(
                        (f) => `<option value="${escapeHtml(f.name)}" ${f.name === state.teachField ? "selected" : ""}>${escapeHtml(f.name)} (${f.kind})</option>`,
                      )
                      .join("")}
                  </select>
                </div>
                <div class="teach-field" style="max-width: 240px;">
                  <label style="font-size: 13px; font-weight: 500;">To value:</label>
                  <input id="teach-value" class="teach-input" type="text" value="${escapeHtml(state.teachValue)}" />
                </div>
              </div>

              <div style="display: flex; justify-content: flex-end; gap: 10px;">
                <button id="btn-cancel-teach" class="btn-secondary">Cancel</button>
                <button id="btn-save-teach" class="btn-primary-solid">Save shortcut &amp; test it now</button>
              </div>
            </section>`
          : ""
      }

      <div class="main-split">
        <div class="results-col">
          <div style="display: flex; align-items: center; justify-content: space-between; padding: 0 4px;">
            <div style="display: flex; align-items: center; gap: 10px;">
              <span style="font-size: 16px; font-weight: 600;">${rejected ? "Query rejected" : `${matchedRows.length} matching items`}</span>
              <span style="font-size: 13px; color: var(--ink-muted);">${rejected ? "No filter was applied" : `filtered from ${preset.rows.length} total in ${escapeHtml(preset.tabLabel)}`}</span>
            </div>
            <span style="font-size: 13px; color: var(--ink-secondary);">Schema: ${escapeHtml(schema.name)} (${escapeHtml(schema.fields.map((f) => f.name).join(", "))})</span>
          </div>

          ${
            matchedRows.length === 0
              ? `<div class="row-card" style="justify-content: center; padding: 40px; color: var(--ink-secondary);">
                  ${rejected ? `Could not apply query: ${escapeHtml(neuralRes.issues.join("; "))}` : "No items matched all active filters. Click a filter pill above to remove it or click Clear."}
                </div>`
              : matchedRows
                  .map(
                    (row) => `
                    <div class="row-card">
                      <div style="display: flex; flex-direction: column; gap: 6px;">
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                          ${row.badges.map((b) => `<span class="badge ${b.tone}">${escapeHtml(b.label)}</span>`).join("")}
                          <span class="mono-num" style="font-size: 12px; color: var(--ink-muted);">${escapeHtml(row.id)} · ${escapeHtml(row.subtitle)}</span>
                        </div>
                        <div style="font-size: 15px; font-weight: 500; color: var(--ink-primary);">${escapeHtml(row.title)}</div>
                      </div>
                      <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px; flex-shrink: 0;">
                        <span class="badge green mono-num">${escapeHtml(row.rightMeta)}</span>
                        <span style="font-size: 12px; color: var(--ink-secondary);">${escapeHtml(row.rightSub)}</span>
                      </div>
                    </div>
                  `,
                  )
                  .join("")
          }
        </div>

        <aside class="arena-col">
          <div class="arena-panel">
            <div>
              <div style="font-size: 15px; font-weight: 600;">How this search was understood</div>
              <div style="font-size: 13px; color: var(--ink-secondary);">
                Live 3-way comparison on your current phrase:
              </div>
            </div>

            <div class="method-box">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; font-weight: 500;">1. Standard Regex / Rule Parser</span>
                <span class="mono-num" style="font-size: 12px; color: ${
                  ruleClauses.length === 0 && dedupedClauses.length > 0 ? "var(--tag-red)" : "var(--ink-secondary)"
                };">
                  ${ruleClauses.length} filters (${ruleMs.toFixed(2)} ms)
                </span>
              </div>
              <div style="font-size: 12px; color: var(--ink-secondary);">
                ${
                  ruleClauses.length === 0
                    ? "Failed to extract filters because the sentence used natural human word order instead of strict key=value syntax."
                    : `Extracted: ${escapeHtml(ruleClauses.map(formatClauseHuman).join(", "))} (${ruleMatchedRows.length} items)`
                }
              </div>
            </div>

            <div class="method-box winner">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; font-weight: 600; color: var(--accent-warm);">2. Sieve 22 KB Neural Tagger</span>
                <span class="mono-num" style="font-size: 12px; color: var(--accent-warm);">
                  ${neuralRes.clauses.length} filters (${neuralMs.toFixed(2)} ms)
                </span>
              </div>
              <div style="font-size: 12px; color: var(--ink-primary);">
                ${
                  neuralRes.clauses.length === 0
                    ? neuralRes.issues.length ? `Could not apply: ${escapeHtml(neuralRes.issues.join("; "))}` : "No residual query."
                    : `Extracted: ${escapeHtml(neuralRes.clauses.map(formatClauseHuman).join(" · "))}`
                }
              </div>
              <div class="mono-num" style="font-size: 11px; color: var(--ink-secondary); padding-top: 2px;">
                Historical synthetic benchmark (500 queries): 99.8% vs 2.4% for a strict regex reference
              </div>
            </div>

            <div class="method-box ${spliceRes.matched ? "winner" : ""}">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; font-weight: 500;">3. Compositional Slang Memory</span>
                <span class="mono-num" style="font-size: 12px; color: var(--ink-secondary);">
                  sim ${(spliceRes.bestSimilarity * 100).toFixed(0)}% (${centroidMs.toFixed(2)} ms)
                </span>
              </div>
              <div style="font-size: 12px; color: var(--ink-secondary);">
                ${
                  spliceRes.matched
                    ? `Spliced "${escapeHtml(spliceRes.bestIdiomId)}" (${spliceRes.mode}) → ${escapeHtml(spliceRes.splicedClauses.map(formatClauseHuman).join(" & "))}${
                        spliceRes.residualQuery ? ` + passed "${escapeHtml(spliceRes.residualQuery)}" to Neural Tagger` : ""
                      }`
                    : `Abstained cleanly (similarity ${(spliceRes.bestSimilarity * 100).toFixed(0)}% is below threshold), handing off 100% to the Neural Tagger.`
                }
              </div>
            </div>

            <div style="padding-top: 8px; border-top: 1px solid var(--paper-border); display: flex; flex-direction: column; gap: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 13px; font-weight: 500;">Batch 1,024-query stress test</span>
                <button id="btn-run-batch" class="btn-secondary" style="height: 28px; font-size: 12px;">
                  Run 1,024 queries
                </button>
              </div>
              ${
                state.batchBenchmarkResult
                  ? `<div class="mono-num" style="font-size: 12px; color: var(--tag-green); background-color: var(--tag-green-bg); padding: 8px 10px; border-radius: 6px;">
                      Batch comparison: 1,024 queries; GPU ${state.batchBenchmarkResult.totalMs.toFixed(1)} ms, CPU ${state.batchBenchmarkResult.cpuComparisonMs.toFixed(1)} ms (${state.batchBenchmarkResult.backend.toUpperCase()})
                    </div>`
                  : `<div style="font-size: 12px; color: var(--ink-muted);">
                      Single query stays on CPU (~0.4 ms). Click to run 1,024 queries in a batch via WebGPU / typed arrays.
                    </div>`
              }
            </div>
          </div>
        </aside>
      </div>
    </main>
  `;

  // Wire event listeners
  const inputEl = document.getElementById("sieve-input") as HTMLInputElement | null;
  if (inputEl) {
    inputEl.addEventListener("input", (e) => {
      state.query = (e.target as HTMLInputElement).value;
      const cursorPos = inputEl.selectionStart ?? state.query.length;
      render();
      const nextInput = document.getElementById("sieve-input") as HTMLInputElement | null;
      if (nextInput) {
        nextInput.focus();
        nextInput.setSelectionRange(cursorPos, cursorPos);
      }
    });
  }

  document.getElementById("btn-clear")?.addEventListener("click", () => {
    state.query = "";
    render();
  });

  document.getElementById("btn-toggle-xray")?.addEventListener("click", () => {
    state.showTechXray = !state.showTechXray;
    render();
  });

  document.getElementById("btn-open-teach")?.addEventListener("click", () => {
    state.showTeachDrawer = !state.showTeachDrawer;
    state.showCustomTableDrawer = false;
    render();
  });

  document.getElementById("btn-cancel-teach")?.addEventListener("click", () => {
    state.showTeachDrawer = false;
    render();
  });

  document.getElementById("btn-open-custom-table")?.addEventListener("click", () => {
    state.showCustomTableDrawer = !state.showCustomTableDrawer;
    state.showTeachDrawer = false;
    render();
  });

  document.getElementById("btn-cancel-custom-table")?.addEventListener("click", () => {
    state.showCustomTableDrawer = false;
    render();
  });

  document.getElementById("btn-load-custom-table")?.addEventListener("click", () => {
    const rawEl = document.getElementById("custom-table-textarea") as HTMLTextAreaElement | null;
    if (!rawEl) return;
    state.customTableRaw = rawEl.value;
    let rawRows: Array<Record<string, unknown>>;
    let inferredSchema: SchemaDef;
    try {
      rawRows = parseCsvOrJsonToRows(rawEl.value);
      if (rawRows.length === 0) throw new Error("Paste at least one row");
      inferredSchema = inferSchemaFromRows("custom_table", rawRows);
      if (inferredSchema.fields.length === 0) throw new Error("No supported columns found");
      state.importError = "";
    } catch (error) {
      state.importError = error instanceof Error ? error.message : "Could not read table";
      render();
      return;
    }
    dynamicSchemas["custom_table"] = inferredSchema;

    const datasetRows: DatasetRow[] = rawRows.map((r, i) => {
      const keys = Object.keys(r);
      const titleKey = keys[0] ?? "id";
      const fields = normalizeImportedRow(r, inferredSchema);
      const badges: DatasetRow["badges"] = [];
      for (const f of inferredSchema.fields) {
        const val = fields[f.name];
        if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
          badges.push({
            label: `${f.name}: ${String(val)}`,
            tone: f.kind === "BOOL" ? (val ? "green" : "red") : f.kind === "NUMBER" ? "blue" : "warm",
          });
        }
      }
      return {
        id: `#ROW-${i + 1}`,
        title: String(r[titleKey] ?? `Row ${i + 1}`),
        subtitle: inferredSchema.fields.map((f) => `${f.name}=${String(fields[f.name] ?? "")}`).join(" · "),
        fields,
        badges: badges.slice(0, 4),
        rightMeta: String(fields[inferredSchema.fields.find((f) => f.kind === "NUMBER")?.name ?? ""] ?? r[titleKey] ?? ""),
        rightSub: "auto-inferred row",
      };
    });

    const exampleField = inferredSchema.fields.find((f) => f.kind === "ENUM" && f.enumValues.length) ?? inferredSchema.fields.find((f) => f.kind === "BOOL") ?? inferredSchema.fields.find((f) => f.kind === "NUMBER") ?? inferredSchema.fields[0];
    const exampleValue = datasetRows.find((row) => exampleField.name in row.fields)?.fields[exampleField.name];
    const exampleQuery = `${exampleField.name} is ${typeof exampleValue === "string" && exampleField.kind === "TEXT" ? JSON.stringify(exampleValue) : String(exampleValue)}`;

    dynamicPresets["custom_table"] = {
      schemaName: "custom_table",
      tabLabel: "Custom Pasted Table",
      tagline: `Auto-inferred ${inferredSchema.fields.length} columns (${inferredSchema.fields.map((f) => `${f.name}:${f.kind}`).join(", ")}).`,
      placeholder: "Query your pasted table in plain English...",
      sampleQueries: [{ label: exampleQuery, query: exampleQuery }],
      defaultShortcuts: [],
      rows: datasetRows,
    };

    state.activeDatasetKey = "custom_table";
    state.query = exampleQuery;
    state.showCustomTableDrawer = false;
    if (inferredSchema.fields.length > 0) {
      state.teachField = inferredSchema.fields[0].name;
      state.teachValue = inferredSchema.fields[0].enumValues[0] ?? "true";
    }
    render();
  });

  document.getElementById("btn-save-teach")?.addEventListener("click", () => {
    const exEl = document.getElementById("teach-examples") as HTMLInputElement | null;
    const fEl = document.getElementById("teach-field-select") as HTMLSelectElement | null;
    const vEl = document.getElementById("teach-value") as HTMLInputElement | null;
    if (exEl && fEl && vEl) {
      const examples = exEl.value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const fieldName = fEl.value;
      const val = vEl.value.trim();
      if (examples.length > 0 && val) {
        const shortcutId = examples[0].slice(0, 24);
        centroidMemory.teach(shortcutId, schema, examples, [
          { field: fieldName, cmp: "eq", value: val, negated: false },
        ]);
        state.query = examples[0];
        state.showTeachDrawer = false;
        render();
      }
    }
  });

  document.getElementById("btn-run-batch")?.addEventListener("click", async () => {
    const queries = Array.from({ length: 1024 }, (_, i) => preset.sampleQueries[i % preset.sampleQueries.length].query);
    const res = await batchEngine.evaluateBatch(queries, schema);
    state.batchBenchmarkResult = {
      ran: true,
      backend: res.backend,
      totalMs: res.totalMs,
      cpuComparisonMs: res.cpuComparisonMs,
      speedup: (res.cpuComparisonMs / Math.max(res.totalMs, 0.1)).toFixed(1) + "x",
    };
    render();
  });

  document.querySelectorAll("[data-dataset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-dataset");
      if (key && dynamicPresets[key]) {
        state.activeDatasetKey = key;
        state.query = dynamicPresets[key].sampleQueries[0].query;
        const firstField = dynamicSchemas[dynamicPresets[key].schemaName].fields[0];
        if (firstField) {
          state.teachField = firstField.name;
          state.teachValue = firstField.enumValues[0] ?? "true";
        }
        render();
      }
    });
  });

  document.querySelectorAll("[data-sample-query]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-sample-query");
      if (q !== null) {
        state.query = q;
        render();
      }
    });
  });

  document.querySelectorAll("[data-remove-idx]").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.query = "";
      render();
    });
  });
}

render();
