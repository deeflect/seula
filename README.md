# Seula

**A 180 KB model that turns what people type into a table's search box into exact filters, in under a millisecond, in the browser.**

```
"overdue and balance at least 1k"   →   Status = overdue   AND   Balance >= 1000
"gdp per cap over 30k, not euro"    →   GDP per cap. (PPP) > 30000   AND   Currency != euro
"population over 10 million"        →   Population > 10000000
"directed by marcos siega"          →   Directed by = marcos siega
"overdue or paid, top 3 by balance" →   Status = overdue OR paid,   sorted by Balance,   first 3
```

It works on tables it has never seen (just pass the header and rows; no retraining). It runs alone, with no LLM, no server and no network: 92.8% rows correct on unseen tables, against 93.2% for Claude Haiku, which needs a network call for every query. It also says when it is unsure, so your UI can show the parsed filters and let the user fix them. The shipped weights are trained only on queries written by open-weight models (Qwen3, Mistral), so there are no API terms to inherit.

![Comparison on FilterBench test](docs/seula-comparison.svg)

## Why use it

| | Seula | Hand-written rules | Small local LLM (Qwen 0.5–1.5B) | Cloud LLM (Claude Haiku) |
|---|---|---|---|---|
| Rows correct (FilterBench test) | 92.8% | 77.4% | 24% / 45% / 66% (0.5B / 0.6B / 1.5B) | 93.2% |
| Right when it says it's sure | 98.2% on the 83% of queries with confidence 0.8 or more | no confidence signal | no useful signal | no confidence signal |
| Time per query | **0.39 ms**, CPU, one thread | 0.03 ms | 0.5 to 1.2 s on a laptop GPU | 0.9 to 4 s per query as we ran it, through a CLI; not measured over the raw API |
| Size | **182 KB** int8 | none | 1–3 GB | API |
| Runs offline / in the browser | yes, no dependencies | yes | heavy | no |
| Per-keystroke filtering | yes | yes | no | no |
| Cost per query | 0 | 0 | 0 | paid |

**Use the confidence.** Every parse comes with a confidence. When it is high, apply the filter as the user types; when it is low, show the parsed filters as chips the user can confirm or edit. On test, 83% of queries get a confidence of 0.8 or more, and 98.2% of those are right. Seula is right on 45 queries where Haiku is wrong, and Haiku is right on 49 where Seula is wrong. Seula reads the whole table and snaps values to the real cell (`éva`, `850 watts`, `–3` with an en dash); Haiku only saw 3 example rows.

## How others do it

- **Products** (Linear, Langfuse, NocoDB, AG Grid, HubSpot, Kibana, Stripe Sigma): a cloud LLM call per query, usually on a paid tier, none offline, and none publish accuracy. AG Grid's own example warns "up to 30s".
- **Research parsers** (SQLova, HydraNet, SeaD on WikiSQL): 100M+ parameter BERT-class models for full SQL. Seula borrows their execution-guided decoding.
- **Small LLMs**: 0.5–1.5B models, 3-shot, get 24–66% here at 0.5–1.2 s. A fine-tuned 0.5B with grammar-constrained output is the fair comparison we haven't run yet.

Seula's niche: any table, in the browser, per keystroke, with a confidence you can act on. Details, sources and the idea backlog: [docs/landscape.md](docs/landscape.md).

## How it works

1. **Read the query and the table together.** Each word gets a hashed word + letter-trigram embedding, plus link features: does it match a column name (exactly, fuzzily, by prefix), is it inside a cell of that column, is it a number, is the column numeric.
2. **Tag the values.** A small BiGRU marks which words are condition values (`overdue`, `1k`).
3. **Decide each condition.** For each value: pick the operator (`= != contains > >= < <=`) from nearby words, and the column using the links plus the header words just before and after it.
4. **Clean up.** Columns are assigned jointly (two conditions don't land on one column, except a lower and upper bound). Values joined by "or" on one column become an any-of list (`overdue or pending`), and `not a or b` a none-of list. Numbers are normalized (`1k`, `2.5m`, `10 million`), text values snap to the real cell (typos, dashes, quotes). If the filter returns no rows, the next most likely column or operator is tried (execution-guided decoding, as in SQLova); if nothing returns rows, confidence drops to 0.
5. **Confidence** is the product of every decision's probability. Low confidence → show the guess for the user to confirm.
6. **Dates, yes/no columns, sorting and top-N** go through a small rule pass before the model (`src/seula/extras.ts`): `last 30 days`, `before march 2024`, `created after 2025`, `not shipped`, `top 5 by balance`, `newest first`. The model parses what is left.

184k parameters. No pretrained language model. Python training code in `bench/seula.py`; the TypeScript runtime `src/seula/model.ts` has no dependencies. It matches Python on 74 parity cases (confidence within 3e-6) and gives the same filters on every dev and test query when both run the int8 weights.

## Use it (TypeScript)

```ts
import { load } from "seula";

const seula = await load();
const invoices = seula.table([
  { client: "Northwind", status: "overdue", balance: 12400 },
  { client: "Fieldwork", status: "paid", balance: 0 },
  { client: "Acme", status: "overdue", balance: 310 },
]);
const result = invoices.parse("overdue and balance over 1k");
// result.conditions: [{ column: "status", type: "text", op: "=", value: "overdue" }, { column: "balance", type: "number", op: ">", value: 1000 }]
// result.confidence: 0.99
if (!result.conditions || result.confidence < 0.5) { /* unsure: show the guess as chips the person can confirm or remove */ }
invoices.apply(result); // the matching rows, sorted and cut if the query asked for that
```

The package is built from `src/seula/` with `npm run pkg` and is not on npm yet. `pkg/README.md` covers the rest: tables that live on a server (`seula.schema`), the TanStack Table and SQL adapters, and loading the weights.

The lower-level calls are still exported for the benchmark scripts: `filter(model, query, table)` returns `[column index, op, value]` conditions and `rowsOf` returns row indices. Several `=` or `contains` conditions on one column mean any of them; several `!=` mean none of them. `seula.model.parse(query, table)` is the model alone, without the date, sort and top-N rules.

`npx tsx spike/readme_examples.ts` prints what the example queries in this README produce. Try it live: `npm install && npm run dev`, then open the page and type into any of the five tables. `npm run deploy` builds it and puts it on Cloudflare (`wrangler.jsonc`, with `worker/capture.js` storing the yes/no verdicts).

## Train it

Training data is the trick: an LLM writes realistic queries for conditions sampled from real tables, so the labels are known by construction.

```bash
# 1. Generate queries with any OpenAI-compatible endpoint (OpenRouter, llama-server, vLLM...).
#    WikiSQL tables come from bench/prepare.py. About $1 on OpenRouter for all four files.
export LLM_URL=https://openrouter.ai/api LLM_KEY=$OPENROUTER_API_KEY
Q=qwen/qwen3-235b-a22b-2507; M=mistralai/mistral-small-3.2-24b-instruct
python3 bench/filterbench.py gen train 7500 $Q bench/out/fb.qtrain.json
python3 bench/filterbench.py gen train 2400 $Q bench/out/fb.qtrain2.json --seed 4 --ranges
python3 bench/filterbench.py gen train 6000 $M bench/out/fb.mtrain.json --seed 5
python3 bench/filterbench.py gen train 2000 $M bench/out/fb.mtrain2.json --seed 6 --ranges
#    Template-written queries add what LLM writers rarely produce: or-lists, `col:value`, `16k+`. No LLM needed.
python3 bench/synth.py 4200 bench/out/fb.synth2.json
# 2. Train (~20 min on a laptop CPU)
cd bench && FB_TRAIN=fb.qtrain.json,fb.qtrain2.json,fb.mtrain.json,fb.mtrain2.json,fb.synth2.json OUT=seula.pt uv run --with torch python3 seula.py train
# 3. Export int8 weights for the browser + a parity fixture
OUT=seula.pt uv run --with torch --with numpy python3 export_seula.py && cd ..
npm test    # checks the TS runtime against Python
```

How much generated data you need (FilterBench dev, rows correct):

| Training data | Rows correct |
|---|---:|
| WikiSQL only (no generated queries) | 46.4% |
| + 2,000 generated queries | 85.3% |
| + 7,485 generated queries | 88.9% |
| + 2,205 queries with ranges (v3c) | 90.0% |
| + better value snapping, `feb 25` dates (decoding only, same weights) | 92.0% |
| + zero-rows check (execution-guided decoding, same weights) | 92.7% |

Open-weight writers work as well as Claude. Same code; the Qwen set uses the same sampled conditions as the Haiku set, and the Mistral set adds new ones, so the shipped set is also larger (part of its gain may be quantity, not the second writer). Scored on dev queries written by Haiku and, as a neutral check, by DeepSeek V3.2 (in no training set):

| Training queries written by | Haiku-written dev | DeepSeek-written dev | Test |
|---|---:|---:|---:|
| Claude Haiku (9.7k) | 92.7% | 88.7% | 88.7% |
| Qwen3-235B (9.9k) | 86.2% | 87.8% | — |
| Qwen3-235B + Mistral Small 3.2 (17.8k) | 90.9% | 89.6% | 90.1% |
| **Same + 4.2k template queries, a label fix and decoding fixes, shipped** | **95.3%** | **93.6%** | **92.5%** |

The last row is the model alone; with the rule pass in front (what `filter()` runs) test is 92.8%. Most of its dev gain comes from the label fix (`over 5` was sometimes labelled `>=`) and decoding, not the template queries: without them dev is 95.8%. They are there for or-lists, which went from 65% to 93% on a 300-query template probe.

Qwen3-4B was tried as a local writer and dropped: it copies the conditions almost word for word (`date contains january`).

**Most weaknesses are fixed with data.** `between 3 and 6` failed because the generator never sampled two bounds on one column. Adding `--ranges` to the generator and 2,205 more queries took a held-out range-only probe (49 queries) from 40.8% to 71.4%, with no code changes. Test moved 84.4% → 85.1%; the decoding fixes took it to 88.7%, the two-writer open-model data to 90.1%, and the current model to 92.5%.

## Benchmarks

- **WikiSQL** (15,747 test questions, ops `= > <`): Seula v2 93.6% correct (rules 88.6%), 83.5% of questions answered at 99% precision. On a 1k sample, 3-shot: Qwen2.5-0.5B 5.1%, Qwen2.5-1.5B 45.5% (Seula 94.8% on the same sample).
- **FilterBench** (ours, `bench/filterbench.py`): filter-box queries with all seven operators over real WikiSQL tables. Conditions are sampled from the table, an LLM writes the query, gold rows are known. Train/dev queries by Haiku, test queries (870) by Sonnet on test tables never seen in training. Scored by returned rows.
- **App-style probe** (80 queries we wrote by hand in the style of SaaS filter boxes, `bench/out/fb.app.json`): 92.5%, up from 72.5% for the previous model. We tuned against it, so it only works as a regression check.

Full numbers: `python3 bench/filterbench.py score bench/out/fb.test.json bench/out/preds/*.fb-test.jsonl`.

## Limits (read before trusting the numbers)

- FilterBench queries are LLM-written, not collected from real users. Training queries come from Qwen3 and Mistral; dev (Haiku) and test (Sonnet) use other writers. The test set was scored once per shipped model (six models in all); every choice was made on dev. For the current model the full pipeline was scored five times: 92.0% first, then 92.8% after fixing date-rule mistakes found on training data, twice more after small rule changes that altered no test prediction, and once after a decoding fix chosen on dev that changed two predictions and left the score at 92.8%. We did not read test queries in between.
- A scorer bug counted a filter as wrong when one column had both a number and a text value. Fixing it moved only the current model (by 0.2 points); the other numbers here were rescored and did not change, apart from the any-of rule for lists (Haiku 93.6% → 93.2%, rules 77.7% → 77.4%, previous Seula 90.1% → 90.0%).
- Haiku saw 3 example rows per table, not the whole table. Its latency includes `claude -p` CLI startup.
- Small LLMs were prompted 3-shot, not fine-tuned. A fine-tuned 0.5B model would likely do much better; it would still be ~1000× slower and ~5000× bigger.
- AND between columns. OR works only as a list of values on one column (`overdue or pending`); there is no OR across columns and no grouping. Dates, sorting and top-N come from rules, covered by 121 unit checks but not by the benchmark. Implied conditions ("home games" → H/A = H) usually fail, with low confidence.
- The zero-rows check assumes the user wants a non-empty result; benchmark answers always have rows, so this helps on the benchmark by construction (+2.8 points on test for the previous model). A real query that matches nothing gets a nearby filter at lower confidence, or confidence 0. Show users the parsed filters.
- Confident mistakes still happen, though fewer: `richest countries` used to be `State = richest` at 0.72; it now returns no filter.
- Ranges still slip: `laps between 90 and 97` → `Laps > 90 AND Laps <= 97` (0.75).
- Or-lists need values that are in the table: `overdue or pending` on a table with no `pending` cell comes back as `Status != overdue` at 0.68.

The earlier rule-guarded prototype and its invoice demo are documented in [docs/v1-README.md](docs/v1-README.md).
