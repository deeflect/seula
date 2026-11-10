# Frozen challenge results

The fixture at `spike/challenge_cases.ts` was frozen before scoring: SHA-256 `721878c3294c6c35d69aad0fa7a8468827a1b04f8334e86adeb673a97f00b55a`. It contains 90 valid queries, 10 expected rejects, and five synthetic rows for each of three existing schemas. All cases and labels were independently agent-authored; they are not user-collected and do not estimate general accuracy. Model SHA-256: `74dc40c3819dbcfb6dbd2c974e8aaa711b53be4911538dc8cbe2db86a598d32f`.

Run `npm run experiment:challenge`. The script uses `featurizeQuery` and `compileTokens` for each arm. Raw mode uses neural roles and boundaries without validation. Guarded mode is production `model.parse`. Rules mode assigns transparent roles from schema matches, grammar lexicons and value shapes, then uses the same compiler and validator. It is a small heuristic, not a tuned competitive parser. The first runner pass accidentally applied validation to raw mode; the runner was corrected without changing the fixture. Scoring then normalized equivalent negated numeric comparisons. The [initial report](experiment-results-initial.json) records the first core after those scoring corrections. A compiler defect exposed by that run was repaired, and the [current report](experiment-results.json) is a **development recheck**, not a new held-out evaluation. Neither labels nor query text changed after scoring.

| Metric | Raw initial → current | Guarded initial → current | Rules initial → current |
| --- | ---: | ---: | ---: |
| Valid accepted / 90 | 82 → 82 | 25 → 26 | 27 → 27 |
| Accepted valid, exact AST | 40 → 44 | 22 → 26 | 26 → 26 |
| Accepted valid, wrong clauses | 42 → 38 | 3 → 0 | 1 → 1 |
| Valid rejected | 8 → 8 | 65 → 64 | 63 → 63 |
| Accepted valid, correct row set | 44 → 49 | 22 → 26 | 26 → 26 |
| Invalid rejected / 10 | 6 → 6 | 10 → 10 | 9 → 9 |

AST comparison ignores clause order, casefolds text, and normalizes equivalent Boolean `eq:false`, `neq`, and negated numeric/date order forms. A nonempty query producing no clause counts as rejected. Correct row sets compare production filtering against an independently implemented evaluator on the 15 synthetic rows; a wrong AST can still select the same five rows by chance. Valid difficult wording stays in the denominator when rejected. The per-case [JSON artifact](experiment-results.json) contains expected and actual clauses, issues, row IDs, and all scores.

The frozen model's raw roles frequently compile some clause, but many are wrong; the guard rejects many valid phrases while removing unsafe accepts on this small invalid set. The rules arm matches the guarded model's current exact AST coverage here, so this challenge does not show that the neural model is necessary or superior. It is not a broad benchmark of either method.

`npm run experiment:vocabulary` reproduces [the comparator experiment](vocabulary-results.json). The original surgery uses four calibration phrases and fixes 4/4 raw positive held-out contexts, but production validation rejects all four. It fails two opposite controls. In a matched-supervision comparison, both methods are supplied `topping => gt` and `beneath => lt` at value 5, then tested at 7. Both queries have identical full feature matrices. Sequential head edits output `lt` for both; the explicit dictionary maps each correctly. This demonstrates a representation limit and dictionary supervision, not discovery of unseen word meanings. Four extra known-query retention examples were unchanged after the original edit; neither those nor the five parity traces prove no forgetting.
