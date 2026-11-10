# Seula

**A filter box that reads plain English.**

Seula is a 182 KB model that turns what someone types above a table into exact filters. It runs in the browser or in Node with no API calls, takes under a millisecond per query, and works on tables it has never seen: you pass the rows, there is no retraining.

**[Try the live demo](https://seula.deeflect.com/demo)** on five sample tables or your own CSV. Site: [seula.deeflect.com](https://seula.deeflect.com).

```
overdue and balance at least 1k     →  Status = overdue  AND  Balance >= 1000
gdp per cap over 30k, not euro      →  GDP per cap. (PPP) > 30000  AND  Currency != euro
population over 10 million          →  Population > 10000000
directed by marcos siega            →  Directed by = marcos siega
overdue or paid, top 3 by balance   →  Status = overdue OR paid,  sorted by Balance,  first 3
```

Every parse comes with a confidence, so your UI can apply a sure filter as the person types and ask about an unsure one.

## Quick start

```sh
npm install seula
```

Until the first release is on npm, build the tarball from this repo: `npm install && npm run pkg && npm pack ./pkg`, then install that file in your project.

```ts
import { load } from "seula";

const seula = await load();
const invoices = seula.table([
  { client: "Northwind", status: "overdue", balance: 12400 },
  { client: "Fieldwork", status: "paid", balance: 0 },
  { client: "Acme", status: "overdue", balance: 310 },
]);

const result = invoices.parse("overdue and balance over 1k");
// result.conditions: [
//   { column: "status", type: "text", op: "=", value: "overdue" },
//   { column: "balance", type: "number", op: ">", value: 1000 },
// ]
// result.confidence: 0.99

invoices.apply(result); // the matching rows, sorted and cut if the query asked for that
```

Build the table once per dataset and call `parse` on every keystroke. A workable rule for the confidence: at 0.5 or more, apply the filter; below that, show the conditions as chips the person can confirm or remove. `conditions` is `null` when nothing in the query matched the table.

| You have | Use |
|---|---|
| Rows in memory | `seula.table(rows)` |
| A table paged on the server | `seula.schema({ column: sampleValues })`, then send `result.conditions` to your API |
| TanStack Table | `toTanStack(result)` from `seula/tanstack` |
| A SQL database | `toSql(result)` from `seula/sql`, values always as parameters |

The package guide, [pkg/README.md](pkg/README.md), covers each of these and how the weights load.

## What it reads

| Typed | Parsed as |
|---|---|
| `balance over 1k`, `at least 16k`, `under 5` | number comparisons |
| `days late between 10 and 40` | two comparisons on one column |
| `overdue or paid` | either value |
| `not euro`, `clients except fieldwork` | exclusion |
| `directed by marcos siega` | a value found in the table |
| `last 30 days`, `before march 2024` | date comparisons, for columns that hold dates |
| `top 3 by balance`, `newest first` | sort and limit |

Values are matched against what is in the table, and the result holds the value as the table spells it (`éva`, `850 watts`, `–3` with an en dash). Typo repair works for near misses on longer words and does not catch everything.

## How it compares

![Comparison on FilterBench test](docs/seula-comparison.svg)

FilterBench test: 870 queries on tables the model never saw in training, scored by the rows returned.

| | Seula | Hand-written rules | Small local LLM (Qwen 0.5 to 1.5B) | Cloud LLM (Claude Haiku) |
|---|---|---|---|---|
| Rows correct | 92.8% | 77.4% | 24% / 45% / 66% (0.5B / 0.6B / 1.5B) | 93.2% |
| Right when it says it is sure | 98.2% on the 83% of queries with confidence 0.8 or more | no confidence | no useful confidence | no confidence |
| Time per query | 0.39 ms, CPU, one thread | 0.03 ms | 0.5 to 1.2 s on a laptop GPU | 0.9 to 4 s as we ran it, through a CLI |
| Size | 182 KB | none | 1 to 3 GB | API |
| Runs offline and in the browser | yes, no dependencies | yes | heavy | no |
| Cost per query | 0 | 0 | 0 | paid |

Read these with the caveats in mind. The test queries were written by a language model, not collected from real users. Haiku saw three example rows per table while Seula read the whole table, and Haiku's latency includes CLI startup. The small LLMs were prompted with three examples and not fine-tuned. Seula is right on 45 queries where Haiku is wrong, and Haiku is right on 49 where Seula is wrong. Parse time grows with the table: about 28 ms at 50,000 rows.

How the sets were built, every number and the full list of caveats: [docs/benchmarks.md](docs/benchmarks.md). Products and research that solve the same problem another way: [docs/landscape.md](docs/landscape.md).

## How it works

1. **Read the query and the table together.** Each word gets a hashed word and letter-trigram embedding, plus link features: does it match a column name (exactly, fuzzily, by prefix), is it inside a cell of that column, is it a number, is the column numeric.
2. **Tag the values.** A small BiGRU marks which words are condition values (`overdue`, `1k`).
3. **Decide each condition.** For each value, pick the operator (`= != contains > >= < <=`) from nearby words, and the column from the links plus the header words just before and after it.
4. **Clean up.** Columns are assigned jointly, so two conditions do not land on one column unless they are a lower and an upper bound. Values joined by "or" on one column become an any-of list. Numbers are normalized (`1k`, `2.5m`, `10 million`) and text values snap to the real cell. If the filter returns no rows, the next most likely column or operator is tried (execution-guided decoding, as in SQLova). If nothing returns rows, confidence drops to 0.
5. **Score it.** Confidence is the product of every decision's probability.
6. **Rules for the rest.** Dates, yes/no columns, sorting and top-N go through a small rule pass before the model (`src/seula/extras.ts`): `last 30 days`, `created after 2025`, `not shipped`, `top 5 by balance`, `newest first`.

The model has 184k parameters and uses no pretrained language model. The shipped weights are trained only on queries written by open-weight models (Qwen3, Mistral). Training code is in `bench/seula.py`. The TypeScript runtime in `src/seula/model.ts` has no dependencies and matches Python on 74 parity cases, with confidence within 3e-6.

## Limits

- English only.
- One table at a time, no joins.
- It filters, sorts and cuts. It does not total or average.
- AND between columns. OR works as a list of values on one column (`overdue or pending`). There is no OR across columns and no grouping, so `overdue and (balance over 1k or days late over 30)` is out of reach.
- Implied conditions (`home games` meaning `H/A = H`) usually fail, with low confidence.
- A query that matches nothing gets a nearby filter at lower confidence, or confidence 0. Show people the parsed filters.
- Ranges still slip sometimes: `laps between 90 and 97` came back as `Laps > 90 AND Laps <= 97` at 0.75.
- Or-lists need values that are in the table: `overdue or pending` on a table with no `pending` cell comes back as `Status != overdue` at 0.68.

## Train your own

An LLM writes realistic queries for conditions sampled from real tables, so the labels are known by construction. Generating the data costs about $1 on OpenRouter and training takes about 20 minutes on a laptop CPU. Commands and what each batch of data added: [docs/training.md](docs/training.md).

The dev and test query files are not in this repository yet. They were written with Claude and will be regenerated with open-weight models before the benchmark data is published. The generator and the scorer are here.

## Develop

```sh
npm install
npm run dev      # the site and the demo, locally
npm test         # runtime, parity with Python, rules, API and adapters
npm run pkg      # build the npm package into pkg/dist
npm run deploy   # build and deploy to Cloudflare (wrangler.jsonc)
```

| Path | What is there |
|---|---|
| `src/seula/` | the runtime, the object API, the TanStack and SQL adapters, the weights |
| `pkg/` | the npm package manifest and its guide |
| `bench/` | training, the FilterBench generator and scorer, baselines |
| `worker/` | the Cloudflare worker that stores yes/no verdicts from the demo |
| `docs/` | benchmarks, training, landscape, and notes on the earlier prototype ([docs/v1-README.md](docs/v1-README.md)) |

The demo asks whether a parse was right. Answering sends the query, the column names and the parsed filter. The rows never leave the tab.

## Licence

Apache-2.0, code and weights. Seula is Finnish for sieve. Made by [Dee](https://deeflect.com). If you want this in your product, get in touch at [dee.agency](https://dee.agency).
