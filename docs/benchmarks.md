# Benchmarks and how to read them

The FilterBench and speed figures below are historical v4c measurements, including execution-guided decoding that tried alternative columns or operators when no rows matched. The current runtime disables that behavior by default, adds status-based abstention, expressions, concepts, and count/sum. Its held-out test accuracy, confidence calibration, and accuracy on the new features have not been measured. The comparison chart in the README also shows the historical v4c run.

On 2026-11-12 we ran a **development regression check** over the same 450 FilterBench dev queries, comparing returned rows to their annotations. The current runtime marked 380/450 (84.44%) `ok`; 373/380 (98.16%) of those returned the annotated rows. The historical pipeline accepted 397/450 (88.22%) at confidence ≥0.8, with 98.24% correct among accepted queries. We used these dev queries to find and fix regressions, so this is not independent validation or an estimate of real-user accuracy. The new expressions, concepts, and aggregates were not evaluated by this set. In a separate warm local timing check on this machine, six queries over a synthetic 1,000-row table were parsed 30 times each: median 0.249 ms, p95 1.438 ms. These timings do not predict speed on other hardware.

## The sets

- **FilterBench** (ours, `bench/filterbench.py`): filter-box queries with all seven operators over real WikiSQL tables. Conditions are sampled from the table, an LLM writes the query, gold rows are known. Train/dev queries by Haiku, test queries (870) by Sonnet on test tables never seen in training. Scored by returned rows.
- **WikiSQL** (15,747 test questions, ops `= > <`): Seula v2 93.6% correct (rules 88.6%), 83.5% of questions answered at 99% precision. On a 1k sample, 3-shot: Qwen2.5-0.5B 5.1%, Qwen2.5-1.5B 45.5% (Seula 94.8% on the same sample).
- **App-style probe** (80 queries we wrote by hand in the style of SaaS filter boxes): 92.5%, up from 72.5% for the previous model. We tuned against it, so it only works as a regression check.

The dev and test query files are not in this repository yet. They were written with Claude and are used only for scoring; they will be regenerated with open-weight models before the benchmark data is published. The generator and scorer are here, so you can build your own sets:

```bash
python3 bench/filterbench.py score bench/out/fb.test.json bench/out/preds/*.fb-test.jsonl
```

## Limits (read before trusting the numbers)

- FilterBench queries are LLM-written, not collected from real users. Training queries come from Qwen3 and Mistral; dev (Haiku) and test (Sonnet) use other writers. The test set was scored once per historical shipped model (six models in all); every choice was made on dev. For v4c the full pipeline was scored five times: 92.0% first, then 92.8% after fixing date-rule mistakes found on training data, twice more after small rule changes that altered no test prediction, and once after a decoding fix chosen on dev that changed two predictions and left the score at 92.8%. We did not read test queries in between.
- A scorer bug counted a filter as wrong when one column had both a number and a text value. Fixing it moved only the current model (by 0.2 points); the other numbers here were rescored and did not change, apart from the any-of rule for lists (Haiku 93.6% → 93.2%, rules 77.7% → 77.4%, previous Seula 90.1% → 90.0%).
- Haiku saw 3 example rows per table, not the whole table. Its latency includes `claude -p` CLI startup.
- Small LLMs were prompted 3-shot, not fine-tuned. A fine-tuned 0.5B model would likely do much better; it would still be ~1000× slower and ~5000× bigger.
- The historical benchmark covered AND between columns and OR lists on one column. It did not cover the new cross-column OR, NOT, grouping, concepts, aggregates, dates, sorting, or top-N. Implied conditions ("home games" → H/A = H) usually fail.
- The historical zero-rows check assumed the user wanted a non-empty result; benchmark answers always had rows, so this helped by construction (+2.8 points on test for the previous model). The current runtime preserves empty results and will not rewrite a filter to find rows.
- Confident mistakes still happen, though fewer: `richest countries` used to be `State = richest` at 0.72; it now returns no filter.
- Ranges still slip: `laps between 90 and 97` → `Laps > 90 AND Laps <= 97` (0.75).
- In the historical run, an OR list with an absent value could be misread: `overdue or pending` without a `pending` cell became `Status != overdue` at 0.68. The new parser uses sibling context for this case; current aggregate accuracy is unmeasured.
