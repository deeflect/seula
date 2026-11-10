# Benchmarks and how to read them

## The sets

- **FilterBench** (ours, `bench/filterbench.py`): filter-box queries with all seven operators over real WikiSQL tables. Conditions are sampled from the table, an LLM writes the query, gold rows are known. Train/dev queries by Haiku, test queries (870) by Sonnet on test tables never seen in training. Scored by returned rows.
- **WikiSQL** (15,747 test questions, ops `= > <`): Seula v2 93.6% correct (rules 88.6%), 83.5% of questions answered at 99% precision. On a 1k sample, 3-shot: Qwen2.5-0.5B 5.1%, Qwen2.5-1.5B 45.5% (Seula 94.8% on the same sample).
- **App-style probe** (80 queries we wrote by hand in the style of SaaS filter boxes): 92.5%, up from 72.5% for the previous model. We tuned against it, so it only works as a regression check.

The dev and test query files are not in this repository yet. They were written with Claude and are used only for scoring; they will be regenerated with open-weight models before the benchmark data is published. The generator and scorer are here, so you can build your own sets:

```bash
python3 bench/filterbench.py score bench/out/fb.test.json bench/out/preds/*.fb-test.jsonl
```

## Limits (read before trusting the numbers)

- FilterBench queries are LLM-written, not collected from real users. Training queries come from Qwen3 and Mistral; dev (Haiku) and test (Sonnet) use other writers. The test set was scored once per shipped model (six models in all); every choice was made on dev. For the current model the full pipeline was scored five times: 92.0% first, then 92.8% after fixing date-rule mistakes found on training data, twice more after small rule changes that altered no test prediction, and once after a decoding fix chosen on dev that changed two predictions and left the score at 92.8%. We did not read test queries in between.
- A scorer bug counted a filter as wrong when one column had both a number and a text value. Fixing it moved only the current model (by 0.2 points); the other numbers here were rescored and did not change, apart from the any-of rule for lists (Haiku 93.6% → 93.2%, rules 77.7% → 77.4%, previous Seula 90.1% → 90.0%).
- Haiku saw 3 example rows per table, not the whole table. Its latency includes `claude -p` CLI startup.
- Small LLMs were prompted 3-shot, not fine-tuned. A fine-tuned 0.5B model would likely do much better; it would still be ~1000× slower and ~5000× bigger.
- AND between columns. OR works only as a list of values on one column (`overdue or pending`); there is no OR across columns and no grouping. Dates, sorting and top-N come from rules, covered by unit checks but not by the benchmark. Implied conditions ("home games" → H/A = H) usually fail, with low confidence.
- The zero-rows check assumes the user wants a non-empty result; benchmark answers always have rows, so this helps on the benchmark by construction (+2.8 points on test for the previous model). A real query that matches nothing gets a nearby filter at lower confidence, or confidence 0. Show users the parsed filters.
- Confident mistakes still happen, though fewer: `richest countries` used to be `State = richest` at 0.72; it now returns no filter.
- Ranges still slip: `laps between 90 and 97` → `Laps > 90 AND Laps <= 97` (0.75).
- Or-lists need values that are in the table: `overdue or pending` on a table with no `pending` cell comes back as `Status != overdue` at 0.68.
