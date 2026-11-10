# Seula — decision

**Status:** shipped (open source, local)
**Date:** 2026-11-09 (numbers refreshed for model v4c; the call is from 2026-11-07) · v1 invoice-demo decision kept in `docs/v1-DECISION.md`

## The call
Open-source Seula (184k params, 182 KB int8, 0.39 ms CPU, Apache-2.0, trained only on open-model-written queries) as a filter-box parser for any table, standalone (no LLM, no server, no fallback), sold on one claim we can show: **within half a point of a cloud LLM on our benchmark, in under a millisecond, offline, with a confidence the UI can act on**.

## Evidence this rests on (FilterBench test, 870 Sonnet-written queries, unseen tables)
- Rows correct: Seula 92.8% (model alone 92.5%; previous model 90.0%), rules 77.4%, Haiku 93.2%, Qwen2.5-1.5B 65.7%, Qwen3-0.6B 45.1%, Qwen2.5-0.5B 24.4% (3-shot).
- Confidence: 83% of queries get 0.8 or more and 98.2% of those are right. Seula is right on 45 queries Haiku misses; Haiku on 49 Seula misses. Execution-guided decoding (zero rows → next-best column/op) gave +2.8 test on the previous model; benchmark answers are never empty, so real-query gain may be smaller.
- Data is the lever: 0 → 2k → 7.5k generated queries = 46 → 85 → 89% dev; +2.2k range queries raised range-only from 41% to 71% with no code change. Writers: Qwen3-235B + Mistral Small (Apache-2.0) beat Haiku-written data on a neutral DeepSeek-written dev (89.6 vs 88.7).
- WikiSQL: 93.6% vs rules 88.6%.
- v4c (2026-11-09): label fix (`over 5` no longer `>=`), 4.2k template queries for or-lists (65% → 93% on a template probe), decoding fixes, a rule pass for dates, sort and top-N. Dev 90.9 → 95.1, neutral dev 89.6 → 93.6, hand-written app probe 72.5 → 92.5.
- Python↔TS parity: 74/74, and identical filters on all dev and test queries with the int8 weights. Landscape (products all cloud-LLM per query, no published accuracy): docs/landscape.md.

## Rejected
- An LLM fallback (cascade): it tested well (95.2% with 18% of calls) but the product is local-only, so it is out of scope.
- "Breakthrough" / "beats LLMs" framing: alone it is 0.4 pts behind Haiku, on LLM-written queries and with Haiku seeing only 3 rows. The win is latency, size, offline use and the confidence signal.
- Rules∩model agreement as the confidence signal: 97.8% prec @ 88.5% cov, worse than model confidence.
- Tiny prompted LLMs as the local tier: 24–66%, at 1000× the latency.
- More architecture work before real queries: the failures left are data gaps.

## Next build step
Collect ~200 real filter queries from real people on their own tables (the `?live=1` page), score rules versus Seula on them, and record the 30–60 s demo video from that page.

## Before publishing
Done: training data regenerated with Qwen3 + Mistral (Apache-2.0), LICENSE Apache-2.0 + NOTICE, renamed Sieve → Seula (Finnish for "sieve"; npm and PyPI free). Open: dev/test are still Claude-written (scoring only), regenerate before publishing them. Plan, prior art, input licenses: docs/commercial.md.

## Kill criterion
On real-user queries, Seula at 95% precision covers no more queries than rules do. Then the generated-data story doesn't transfer, so stop.
