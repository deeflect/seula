# Scope and possible uses

> Stale: written for v1 (Sieve, the invoice demo). Current state and numbers are in the README and HANDOFF.md.

Sieve is a small research demo for applying supported table filters locally. The safest current use is an interactive UI that shows the parsed clauses and any rejection issues before applying them. It can be tested with explicit schemas or with inferred CSV/JSON schemas after reviewing the inferred field types and aliases.

The [agent-authored challenge](experiment-results.md) found 22 exact guarded ASTs among 90 valid queries initially, and 26 after a compiler fix informed by that run. All 10 invalid queries were rejected in both runs. That is too little evidence for dependable natural-language filtering in legal, medical, financial, or arbitrary table workflows. The earlier same-generator synthetic transfer benchmark and strict regex comparison do not establish superiority over a practical rules parser.

Potential product work includes a reviewed table filter UI, a local prefilter for structured queries, or a shortcut library for team vocabulary. Each needs its own representative user queries, error and rejection review, and workload measurement. There is no measured API cost saving, privacy guarantee, universal HTML table support, frame-rate guarantee, or zero-hallucination result here.

The [vocabulary experiment](vocabulary-results.json) isolates a key limit: explicit supervised comparator mapping distinguishes opposite words when they have identical neural features; output-head editing does not. Any lexical adapter should be evaluated separately before shipping. Current production parsing rejects those unknown words.

For implementation details and prior art, see [How Sieve works](HOW-IT-WORKS-HUMANS.md).
