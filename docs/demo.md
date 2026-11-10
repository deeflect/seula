# Invoice finder demo

## Share assets

- [16-second walkthrough](demo-assets/sieve-walkthrough.mp4) — assembled from actual UI captures
- [Raw browser interaction](demo-assets/sieve-invoices.mp4)
- [Desktop screenshot](demo-assets/desktop.png) · [Mobile screenshot](demo-assets/mobile.png)

Local preview: https://sieve.localhost:1355 (not a public URL).

## Draft post

Built a tiny local invoice finder. Type “overdue and balance at least $1k” → 3 sample invoices worth $5,150 → export the list. Bring your own CSV. Filtering runs in your browser, with no inference API call.

## Walkthrough sequence

1. All overdue: four invoices, $5,600.
2. Default search: three invoices, $5,150.
3. Small balances: one invoice, $450.
4. Return to default search.

## Browser checks

Parent-run browser QA verified the counts above; rejected and empty-result queries; export disabled for no results; imported cents and HTML escaping; search after disconnecting the loaded page; no horizontal overflow at 390px; and the older lab route.

## Limits

The trained parser recognizes a narrow invoice schema, with explicit aliases for “at least” and “beneath.” Unsupported queries are rejected. Import requires USD balances and exact column names. Data is held in the page only and is lost on refresh. This is a focused demo, not a general AI assistant or a claim of general accuracy.
