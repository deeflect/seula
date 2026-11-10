# Seula

A plain-English filter box for any table, with no LLM.

Seula is a 182 KB model that runs in the browser or in Node. Someone types `overdue and balance over 1k, top 5 by balance` and gets structured filters back fast enough to run on every keystroke: under a millisecond on a table of a few hundred rows, about 28 ms on 50,000. The query and the data stay on the device.

```sh
npm install seula
```

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
//   confidence: 0.99,
// }

invoices.apply(result); // [{ client: "Northwind", status: "overdue", balance: 12400 }]
```

`table()` reads the column names from the first row. Build it once per dataset and call `parse` on every keystroke.

## Use the confidence

Every parse comes with a `confidence` between 0 and 1. A workable rule: at 0.5 or more, apply the filter as the person types. Below that, show the conditions as chips and let them confirm or remove one. `conditions` is `null` when nothing in the query matched the table.

`apply` accepts a result you have edited, so removing a chip is `result.conditions.splice(i, 1)` followed by `apply(result)`.

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

Send `result.conditions` to your API, or turn them into SQL (below). A text value that is missing from the sample still parses, but it cannot be corrected against the real values, so a larger sample gives better matches. A value outside the sample comes back in lower case, which matters if your database compares text case-sensitively. On our development set, parsing from column values alone selected the right rows for 94.9% of queries, against 95.1% with the full rows.

## TanStack Table

```ts
import { toTanStack, seulaFilterFn } from "seula/tanstack";

// in your column definitions: filterFn: seulaFilterFn
const { columnFilters, sorting, limit } = toTanStack(result);
table.setColumnFilters(columnFilters);
table.setSorting(sorting);
```

Column ids must equal the keys of the row objects, which is what `accessorKey` gives you. `limit` is returned as a number for you to apply (a page size, or a slice).

## SQL

```ts
import { toSql } from "seula/sql";

const { where, orderBy, limit, params } = toSql(result);
// where:  "total" > ? AND ("status" = ?) AND ("country" = ?)
// params: [500, "shipped", "Germany"]
db.prepare(`SELECT * FROM orders WHERE ${where}`).all(...params);
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

All conditions must hold, with one exception: on a single column, the conditions that use `=`, `in`, `contains` or `contains any` are alternatives, and one of them matching is enough. `passes(getValue, conditions)` applies this rule if you filter rows yourself:

```ts
import { passes } from "seula";
rows.filter((row) => passes((column) => row[column], result.conditions ?? []));
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

- English queries, one table at a time. No joins, no totals or averages, no nested logic such as `a and (b or c)`.
- On the held-out test set (870 queries on tables the model never saw in training) it selects exactly the right rows for 92.8% of queries. Those queries were written by a language model, so people typing into your product may phrase things differently. The confidence score is there for that.
- Relative dates use the device clock. Pass `parse(query, { today })` to fix the date.
- A column is treated as numeric when at least 80% of its values parse as numbers.

## Licence

Apache-2.0, for the code and the model weights.
