# Seula

A plain-English filter box for any table, with no LLM.

Seula is a 182 KB model that runs in the browser or in Node. Someone types `overdue and balance over 1k, top 5 by balance` and gets structured filters. The query and the data stay on the device.

```sh
npm install seula
```

Until the first npm release, build a tarball from the repository with `npm install && npm run pkg && npm pack ./pkg`, then install that file.

## Filter rows you already have

```ts
import { load } from "seula";

const seula = await load();
const invoices = seula.table([
  { client: "Northwind", status: "overdue", balance: 12400 },
  { client: "Fieldwork", status: "paid", balance: 0 },
  { client: "Acme", status: "overdue", balance: 310 },
]);

const result = invoices.parse("overdue and balance over 1k");
// {
//   conditions: [
//     { column: "status", type: "text", op: "=", value: "overdue" },
//     { column: "balance", type: "number", op: ">", value: 1000 },
//   ],
//   status: "ok",
//   confidence: 0.99, // heuristic score
// }

if (result.status === "ok") invoices.apply(result);
```

`table()` reads the column names from the first row. Build it once per dataset and call `parse` on every keystroke.

## Check the status

`status` is `ok`, `incomplete`, `unsupported`, or `ambiguous`. Non-`ok` results carry a short `message`. Apply `ok` results only. An ambiguous result may contain a candidate `conditions` or `expression`; after the person confirms it, call `apply({ ...result, confirmAmbiguous: true })`. Incomplete and unsupported results cannot be forced. `confidence` is a heuristic score, not a calibrated probability.

For a simple result, `conditions` is an editable list; removing a chip can remove its condition. Grouped OR/NOT results use `expression`. Keep that tree intact or remove it as a unit, since flattening it changes the logic.

## What it reads

| Typed | Parsed as |
|---|---|
| `balance over 1k`, `at least 16k`, `under 5` | number comparisons |
| `days late between 10 and 40` | two comparisons on one column |
| `overdue or paid` | either value |
| `not euro`, `clients except fieldwork` | exclusion |
| `directed by marcos siega` | a value found in the table |
| `last 30 days`, `before march 2024` | date comparisons, for columns that hold dates |
| `top 3 by balance`, `newest first` | `sort` and `limit` |
| `overdue and (balance over 1k or days late over 30)` | grouped `expression` |
| `count overdue`, `total overdue balance` | `aggregate` descriptor |

Count and sum use all matching rows: `invoices.aggregate(invoices.parse("total overdue balance"))`. `apply` still returns matching rows. Define a table-scoped concept with `seula.table(rows, { concepts: { "at risk": [{ column: "daysLate", type: "number", op: ">", value: 10 }] } })`; `at risk except Acme` combines that exact definition with a normal exclusion. Concepts are declared filters, not learned meanings.

Values are matched against what is in the table, and the result holds the value as the table spells it. Typing `feildwork` returns `Fieldwork` when that is a client in the table. Typo repair works for near misses on longer words and does not catch everything.

## Tables that live on a server

When the browser holds one page of a large table, give Seula the column names and a sample of the values in each column. Include the smallest and largest value for number columns.

```ts
const orders = seula.schema({
  status: ["open", "shipped", "refunded"],
  country: ["Germany", "France", "Japan"],
  total: [0, 25000],
  placed: ["2024-01-03", "2026-11-09"],
});

const result = orders.parse("shipped to germany, total over 500");
```

Check `result.status` and send the full result to your API, including `expression` and `aggregate`, or turn it into SQL (below). Missing sample values are uncertain unless the column is explicit or an OR sibling grounds them. A larger sample gives more evidence. Historical v4c development measurements reported 94.9% with sampled values and 95.1% with full rows; the current sampled-schema behavior has not been remeasured.

## TanStack Table

```ts
import { toTanStack, seulaFilterFn } from "seula/tanstack";

// in your column definitions: filterFn: seulaFilterFn
if (result.status === "ok") {
  const { columnFilters, sorting, limit, aggregate } = toTanStack(result);
  table.setColumnFilters(columnFilters);
  table.setSorting(sorting);
}
```

Column ids must equal the keys of the row objects, which is what `accessorKey` gives you. `limit` is returned as a number for you to apply (a page size, or a slice).

## SQL

```ts
import { toSql } from "seula/sql";

if (result.status === "ok") {
  const { select, where, orderBy, limit, params } = toSql(result);
  db.prepare(`SELECT ${select} FROM orders WHERE ${where}`).all(...params);
}
```

Values are always parameters. Column names come from your own schema and are quoted. For Postgres, pass `{ placeholder: (i) => "$" + i }`. For MySQL, pass ``{ quote: (name) => "`" + name + "`" }``. Date conditions compare against ISO dates (`2026-11-09`), so the column needs to be a date type or ISO text.

## Conditions

```ts
type Condition = {
  column: string;
  type: "text" | "number" | "date";
  op: "=" | "!=" | ">" | ">=" | "<" | "<=" | "contains" | "in" | "not in" | "contains any";
  value: string | number | (string | number)[]; // an array for the last three ops
};
```

For a legacy simple `conditions` list, all conditions must hold, with one exception: on a single column, `=`, `in`, `contains`, or `contains any` are alternatives. `passes(getValue, conditions)` applies this rule. For a complex `expression`, use `passesExpression`:

```ts
import { passes, passesExpression } from "seula";
rows.filter((row) => passes((column) => row[column], result.conditions ?? []));
rows.filter((row) => result.expression && passesExpression((column) => row[column], result.expression));
```

Text is compared without case or punctuation. `contains` matches whole words.

## Loading the model

`load()` finds `weights.bin` next to the package code. In Node it reads the file from disk. In the browser it fetches it, and Vite, webpack 5 and other bundlers that understand `new URL("./weights.bin", import.meta.url)` copy the file for you. If your setup does not, host the file yourself and pass its URL or its bytes:

```ts
await load("/static/seula-weights.bin");
await load(bytes); // ArrayBuffer or Uint8Array
```

The file is exported as `seula/weights.bin`.

## Limits

- English queries, one table at a time. No joins, averages, or grouped aggregation. Count and sum are supported.
- Historical v4c scoring on 870 held-out, model-written queries found 92.8% exact rows. That pipeline used execution-guided rewriting, which is disabled by default now. Current held-out test and new-feature accuracy have not been measured. A dated development regression check and local timing are in the repository's `docs/benchmarks.md`.
- Relative dates use the device clock. Pass `parse(query, { today })` to fix the date.
- A column is treated as numeric when at least 80% of its values parse as numbers.

## Licence

Apache-2.0, for the code and the model weights.
