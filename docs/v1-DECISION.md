# Sieve — a focused invoice demo

Date: 2026-11-07. User delegated use-case choice and requested a clear Twitter demo.

## Call
Build a local invoice finder for freelancers and small studios. Type “overdue and balance at least $1k”, inspect the exact filters and matching invoices, export a follow-up CSV. Use explicit sample data and accept a documented USD invoice CSV format. Keep the original parser lab accessible. Prepare a screen recording and draft post; do not publish.

## Why
A visible task and checkable dollar total explain the value better than a generic parser playground. The existing trained tagger, lexical compiler, explicit vocabulary adapter, and rejection guards can support a constrained demonstration without claiming general language understanding. Search and export run locally; no account or inference API is needed.

## Evidence
Core audit repairs and prior challenge results remain in docs/experiment-results.md. The broad challenge achieved only 26/90 correct guarded accepts after development fixes, so it does not establish general parsing reliability or neural superiority. Weight surgery cannot recover opposite meanings from identical input features.

The new explicitly taught adapter was frozen before an independent agent-authored 60-case check. Initial result: 39/40 valid queries returned expected rows, 38 exact clause representations, 20/20 contract rejection cases rejected. One accepted clause is logically equivalent (negated > versus <=); one prefix word order was unsupported. The general prefix-order repair reaches 40/40 exact/semantic valid cases and 20/20 safe rejects on the same set; that rerun is development evidence, not a new holdout. None of these numbers estimates real-user accuracy.

## Rejected
- Universal spreadsheet assistant: insufficient evidence and scope.
- Larger checkpoint or blind weight edits: the lexical information problem comes first.
- Billing integration, email sending, accounts, and dashboards: unnecessary to demonstrate finding and exporting invoices.
- “Breakthrough”, “99.8% accurate”, and “zero forgetting”: unsupported claims. Existing work includes gpu-query and ROME; see prior research links in docs.

## Smallest next step / kill criterion
The complete invoice flow, mobile layout, malformed imports, inclusive amount boundary, and export passed targeted checks; walkthrough assets are saved. Next test with people using their own invoice exports and queries. Keep the neural component only if it adds useful correct coverage over a matched rules baseline at comparable wrong-accept rates. Reframe as a conventional filter tool if people cannot find/export their intended invoices reliably.
