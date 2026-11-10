# Sieve

Sieve is a browser demo of a small trained role tagger plus schema matching, deterministic compilation, validation, and row filtering. It turns supported **AND-only** filter phrases into clauses for a known table schema. It is a research prototype, not a general natural-language query interface.

The model has 29,613 parameters. Its int6 packed tensors total about 21.7 KiB; the JSON file, TypeScript runtime, browser bundle, and app are larger. Training used generated examples on four schemas. Earlier transfer scores used examples from the same generator on disjoint schemas, so they do not estimate real user reliability.

## Current evidence

The independently authored, frozen [challenge](spike/challenge_cases.ts) contains 90 valid phrases, 10 expected rejects, and 15 synthetic rows. Cases are agent-authored, not human-collected. The table is a **development recheck** after a compiler fix prompted by the first score; the [initial report](docs/experiment-results-initial.json), [current per-case results](docs/experiment-results.json), and [method](docs/experiment-results.md) are available.

| Method | Valid accepted / 90 | Exact accepted AST / 90 | Accepted with wrong clauses | Correct row sets / 90 | Invalid rejected / 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw neural roles + compiler | 82 | 44 | 38 | 49 | 6 |
| Guarded `model.parse` | 26 | 26 | 0 | 26 | 10 |
| Heuristic roles + same compiler/validator | 27 | 26 | 1 | 26 | 9 |

Rejection protects against some wrong results but leaves many valid phrases unsupported. Guarded valid exact coverage is 26/90 in this development recheck; the initial frozen-core score was 22/90. A correct row set can occur by coincidence even with a wrong AST. The rules baseline is a simple grammar/schema heuristic, not an optimized parser; the older `baselineParse` strict regex is a weaker control.

A separate [vocabulary experiment](docs/vocabulary-results.json) compares the existing output-head edit with an explicit, supervised word dictionary. The edit changes raw roles for four positive examples, but guarded parsing still rejects their unknown words. `topping` and `beneath` produce identical input features; even after both meanings are supplied as calibration, the head edit predicts the same comparator for both. A dictionary can distinguish them by reading their words. This does not imply the model learned unseen meanings.

## Use the API

```ts
import { SieveModel, inferSchemaFromRows, normalizeImportedRow, parseCsvOrJsonToRows } from "./src/engine";
import { filterDatasetRows } from "./src/datasets";

const rawRows = parseCsvOrJsonToRows("Name,Score\nAna,4\nZoë,7");
const schema = inferSchemaFromRows("people", rawRows);
const rows = rawRows.map((row, i) => ({
  id: String(i), title: "", subtitle: "", badges: [], rightMeta: "", rightSub: "",
  fields: normalizeImportedRow(row, schema),
}));
const result = new SieveModel().parse("score > 5", schema);
const matches = result.accepted ? filterDatasetRows(rows, result.clauses) : [];
if (!result.accepted) console.log(result.issues);
```

Always check `accepted` and `issues`; rejected parses have empty clauses. Imported row values should be normalized before filtering. Review inferred field types and aliases for your domain. `OR`, grouping, unsupported syntax, ambiguous words, and queries over 24 tokens are rejected or unsupported. Filtering clauses use conjunction; missing field values never match, including negated clauses. The browser demo is in `src/main.ts`.

### Teach numeric operator words explicitly

```ts
import { parseWithOperatorAliases } from "./src/engine";

const adapted = parseWithOperatorAliases(new SieveModel(), "score topping 5", schema, { topping: "gt" });
// adapted.normalizedQuery: "score > 5"; adapted.result.clauses only if adapted.result.accepted
```

The caller supplies each meaning (`eq`, `neq`, `gt`, `gte`, `lt`, or `lte`). This opt-in dictionary only rewrites unquoted phrases beside a known numeric field and a compatible numeric value, in field–operator–value, value–operator–field, or operator–value–field order. It returns the original query, normalized query, and source ranges for rewrites; `result.tokens` offsets refer to the normalized query. The guarded parser still decides acceptance. This is explicit vocabulary mapping, not model learning, and it does not add `OR` or grouping support.

## Run

```bash
npm install
npm test
npm run experiment:challenge
npm run experiment:vocabulary
npx tsc --noEmit
npm run build
```

See [how it works](docs/HOW-IT-WORKS-HUMANS.md) and [scope and possible uses](docs/BREAKTHROUGH-USAGE.md).
