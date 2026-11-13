# How others turn text into table filters (checked 2026-11-07)

Read from public docs and papers on the date above. Features change fast; "not stated" means the docs we found didn't say.

## Products

| Product | How | Shows the parsed filter? | Offline | Notes |
|---|---|---|---|---|
| Linear "Filter with AI" | cloud LLM (inferred) | applies normal filters | no | handles "due next week". [docs](https://linear.app/docs/filters) |
| Langfuse "Ask AI" filter bar | cloud LLM (Bedrock) | **editable filter pills**, refine/add/remove | no | drops columns the model invents; off by default. [changelog](https://langfuse.com/changelog/2026-06-26-ask-ai-filter-search-bar) |
| NocoDB "Filter with AI" | cloud LLM, credits | rows in the filter panel | self-host needs an AI key | relative dates, AND/OR; Add / Replace / Clear modes. [docs](https://nocodb.com/docs/product-docs/noco-ai/create-filter) |
| AG Grid AI Toolkit | your LLM (example: gpt-5-mini) | JSON → grid state | no | Enterprise; example UI warns "up to 30s". [docs](https://www.ag-grid.com/javascript-data-grid/ai-toolkit/) |
| HubSpot Breeze (lists) | cloud LLM | review before saving | no | segments only. [docs](https://knowledge.hubspot.com/lists/use-ai-assistants-in-lists) |
| Kibana ES\|QL, Stripe Sigma, Jira, Grafana, Datadog | cloud LLM | full query language, editable | no | ask for a query, not a filter box. [Kibana](https://www.elastic.co/docs/explore-analyze/query-filter/languages/esql-kibana), [Stripe](https://docs.stripe.com/stripe-data/write-queries) |
| openstatus data-table (OSS) | any LLM via Vercel AI SDK | streams JSON from a Zod schema of the columns | no | typed `host:api` skips the LLM. [docs](https://data-table.openstatus.dev/docs/ai-filters) |
| Algolia Query Categorization | model trained on clicks, no LLM | category filter + boost | n/a | one fixed taxonomy, not any table. [blog](https://www.algolia.com/blog/product/new-ai-capabilities-available-to-algolia-customers-query-categorization/) |

**Pattern:** every product filter box we found calls a cloud LLM per query, most are paid tiers, none run offline, and none publish accuracy numbers.

## Models

| Approach | Size | Notes |
|---|---|---|
| SQLova, X-SQL, HydraNet, IE-SQL, SeaD (WikiSQL) | BERT/RoBERTa-large class, 100M+ | full SQL incl. SELECT, 86–93% execution accuracy with execution-guided decoding. Not the same metric as ours. [leaderboard](https://github.com/salesforce/wikiSQL), [SQLova](https://arxiv.org/abs/1902.01069) |
| SQLNet / TypeSQL | ~10M, no pretrained LM | closest relatives: sketch slots + table-content features |
| Flan-T5-small query → filter JSON | ~77M | 99.4% exact on its own synthetic, fixed-schema data (price, rating…); not any table. [paper](https://arxiv.org/abs/2601.16492) |
| Qwen2.5-0.5B LoRA, schema in prompt | 0.5B | 20.8% exact match on its own set. [card](https://huggingface.co/Ionio-ai/Qwen2.5-0.5B-Instruct-Ecommerce-Extraction-LoRA) |
| FunctionGemma 270M, Hammer 0.5B, NuExtract-tiny 0.5B, GLiNER 50–90M | 50M–0.5B | general function calling / extraction; no filter benchmark numbers found |
| Small LLMs, 3-shot (measured by us) | 0.5–1.5B | 24–66% on FilterBench test, 0.5–1.2 s |
| **Seula** | **0.18M, 182 KB** | Historical v4c FilterBench: 92.8%, 0.39 ms with execution-guided decoding. Current pipeline unmeasured; confidence is heuristic. |

No public NL→filter benchmark turned up; FilterBench seems to be the only one.

## Ideas, ranked (status)

1. **Execution-guided decoding** (SQLova): tested historically, then disabled by default because it can change a valid empty-result filter. Historical test 85.9 → 88.7, cascade LLM calls 16% → 9%.
9b. **Second writer** (Mistral alongside Qwen): **done.** Neutral dev 87.8 → 89.6, test 90.1.
2. **Editable chips + "did you mean"** (Langfuse, HubSpot): show the parse as chips; at low confidence show it greyed out with the next-best alternative. `?live=1` page.
3. **Typed syntax bypass** (openstatus): `col:value`, `col>5` parsed by rules, model skipped.
4. **(Out of scope: the product has no LLM fallback.) Cache and mine LLM fallbacks** (Instacart, Etsy): cache by (table schema, normalized query); keep fallback answers as training labels for the next model.
5. **Condition-count head** (SQLova where-number): targets the ~13 dev errors with the wrong number of conditions.
6. **Learned calibrator** ([Kamath et al. 2020](https://aclanthology.org/2020.acl-main.503)): a possible future model over decision probabilities and margins; the current confidence is not calibrated.
7. **Seed ensemble**: 3 models (550 KB); disagreement = abstain.
8. **Relative dates**: a small rule layer ("last 30 days", "this quarter") before the tagger, or chrono-node on the value span.
9. **Writer diversity in generated data** (Spider-Syn / Spider-Realistic idea): queries that paraphrase or omit column names, a second writer model, tables with the same value in two columns. Targets wrong-column errors (16 on dev).
10. **`-oregon` / `-18` as NOT**: dev misses where a leading dash means "not".

## Baseline we still owe

A **fine-tuned** Qwen2.5-0.5B with the schema in the prompt and grammar-constrained output (llama.cpp GBNF / XGrammar, column and value as enums). This is the fair "small LLM" comparison. Expect it to be ~1000× slower and ~1500× bigger; accuracy unknown. About an hour of GPU time.
