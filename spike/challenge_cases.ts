// Frozen, independently authored agent challenge. Do not edit after the first scored run.
export const validCases = `
I01|issues with points topping 5|points:gt:5
I02|issues with points beneath 5|points:lt:5
I03|estimates of at least 5 points|points:gte:5
I04|estimates of at most 5 points|points:lte:5
I05|effort exceeds 5 points|points:gt:5
I06|effort falls short of 5 points|points:lt:5
I07|story points 5 or higher|points:gte:5
I08|story points 5 or lower|points:lte:5
I09|more than 5 points|points:gt:5
I10|fewer than 5 points|points:lt:5
I11|still open issues|status:eq:open
I12|exclude closed issues|!status:eq:closed
I13|tagged bug|labels:eq:bug
I14|anything except docs issues|!labels:eq:docs
I15|priority P0|priority:eq:p0
I16|routine priority|priority:eq:routine
I17|assignee contains Ana|assignee:contains:Ana
I18|owner name includes Zoë|assignee:contains:Zoë
I19|filed before 2026-01-11|created:before:2026-01-11
I20|opened after 2026-01-11|created:after:2026-01-11
I21|locked issues|locked:eq:true
I22|issues that are not locked|locked:eq:false
I23|open with over 5 points|status:eq:open & points:gt:5
I24|triaged and priority P1|status:eq:triaged & priority:eq:p1
I25|bugs that aren't locked|labels:eq:bug & locked:eq:false
I26|owner includes Ana; filed before 2026-01-11|assignee:contains:Ana & created:before:2026-01-11
I27|urgent priority, estimate no greater than 3 points|priority:eq:urgent & points:lte:3
I28|blocked, but not locked|status:eq:blocked & locked:eq:false
I29|priority P2 — 5+ points|priority:eq:p2 & points:gte:5
I30|rank P3; effort ≤ 2 points|priority:eq:p3 & points:lte:2
C01|revision count past 4|revisions:gt:4
C02|revision count under 4|revisions:lt:4
C03|no fewer than 4 revisions|revisions:gte:4
C04|no more than 4 revisions|revisions:lte:4
C05|revisions beyond 4|revisions:gt:4
C06|revisions below 4|revisions:lt:4
C07|minimum 4 rounds of revisions|revisions:gte:4
C08|maximum 4 rounds of revisions|revisions:lte:4
C09|revisions ≥ 4|revisions:gte:4
C10|revisions ≤ 4|revisions:lte:4
C11|panicking clients|vibe:eq:panicking
C12|leave out ghosting clients|!vibe:eq:ghosting
C13|requests to make the logo bigger|ask:eq:logo_bigger
C14|only quick tweaks|ask:eq:quick_tweak
C15|invoices marked overdue|invoice_state:eq:overdue
C16|exclude paid invoices|!invoice_state:eq:paid
C17|founder name contains Inez|founder:contains:Inez
C18|client name includes Zoë|founder:contains:Zoë
C19|deadlines before 2026-01-11|due:before:2026-01-11
C20|due dates after 2026-01-11|due:after:2026-01-11
C21|deposit has cleared|deposit_cleared:eq:true
C22|deposit hasn't cleared|deposit_cleared:eq:false
C23|panicking with overdue invoices|vibe:eq:panicking & invoice_state:eq:overdue
C24|logo bigger and at least 4 revisions|ask:eq:logo_bigger & revisions:gte:4
C25|not paid, due before 2026-01-11|!invoice_state:eq:paid & due:before:2026-01-11
C26|ghosted payment and no cleared deposit|invoice_state:eq:ghosted & deposit_cleared:eq:false
C27|founder includes Inez, 4 revisions or fewer|founder:contains:Inez & revisions:lte:4
C28|vibe is chill; ask is a quick tweak|vibe:eq:chill & ask:eq:quick_tweak
C29|deposit cleared — revisions above 4|deposit_cleared:eq:true & revisions:gt:4
C30|deadline after 2026-01-11 and not panicking|due:after:2026-01-11 & !vibe:eq:panicking
D01|latency over 300 ms|latency_ms:gt:300
D02|latency under 300 ms|latency_ms:lt:300
D03|at least 300 milliseconds of latency|latency_ms:gte:300
D04|at most 300 milliseconds of latency|latency_ms:lte:300
D05|latency exceeds 300 ms|latency_ms:gt:300
D06|latency falls below 300 ms|latency_ms:lt:300
D07|latency 300 ms or longer|latency_ms:gte:300
D08|latency 300 ms or shorter|latency_ms:lte:300
D09|latency higher than 0.3 seconds|latency_ms:gt:300
D10|latency lower than 0.3 seconds|latency_ms:lt:300
D11|minimum 300 ms latency|latency_ms:gte:300
D12|maximum 300 ms latency|latency_ms:lte:300
D13|production environment|environment:eq:prod
D14|EU West cluster|environment:eq:eu_west
D15|exclude sandbox deployments|!environment:eq:sandbox
D16|deployments that succeeded|outcome:eq:succeeded
D17|releases that weren't rolled back|!outcome:eq:rolled_back
D18|initiator name includes Zoë|initiator:contains:Zoë
D19|deployed by someone named Ana|initiator:contains:Ana
D20|shipped before 2026-01-11|deployed:before:2026-01-11
D21|released after 2026-01-11|deployed:after:2026-01-11
D22|canary enabled|canary_enabled:eq:true
D23|canary disabled|canary_enabled:eq:false
D24|successful production releases|outcome:eq:succeeded & environment:eq:prod
D25|running canary deployments|outcome:eq:running & canary_enabled:eq:true
D26|rolled back outside production|outcome:eq:rolled_back & !environment:eq:prod
D27|initiated by Ana, over 300 ms latency|initiator:contains:Ana & latency_ms:gt:300
D28|prod releases before 2026-01-11, canary off|environment:eq:prod & deployed:before:2026-01-11 & canary_enabled:eq:false
D29|EU West and pending|environment:eq:eu_west & outcome:eq:pending
D30|not aborted; no more than 0.3 seconds latency|!outcome:eq:aborted & latency_ms:lte:300
`.trim().split("\n");

export const rejectedCases = `
R01|issues|open or triaged issues|unsupported OR
R02|client_work|(panicking and overdue) or paid|unsupported grouping/OR
R03|deployments|latency =~ /300/|unrecognized operator
R04|issues|points roughly equal to five|unsupported approximate operator
R05|issues|points above|incomplete comparison
R06|client_work|due before 2026-13-40|malformed date
R07|deployments|latency more than|incomplete comparison
R08|issues|Show me every issue that is still open and has an estimate of at least five points and was filed before 2026-01-11 and is not locked.|over 24 tokens
R09|client_work|clients who are late|ambiguous field or threshold
R10|deployments|recent deployments|ambiguous date threshold
`.trim().split("\n");

export const oracleRows: Record<string, Array<{ id: string; fields: Record<string, string | number | boolean> }>> = {
  issues: [
    { id: "I1", fields: { status: "open", priority: "p1", assignee: "Ana", labels: "bug", points: 4, created: "2026-01-10", locked: false } },
    { id: "I2", fields: { status: "closed", priority: "p2", assignee: "Zoë", labels: "docs", points: 5, created: "2026-01-11", locked: true } },
    { id: "I3", fields: { status: "triaged", priority: "p0", assignee: "Minh", labels: "perf", points: 6, created: "2026-01-12", locked: false } },
    { id: "I4", fields: { status: "blocked", priority: "urgent", assignee: "Ana", labels: "chore", points: 0, created: "2026-01-09", locked: true } },
    { id: "I5", fields: { status: "open", priority: "routine", labels: "feature", points: 10, created: "2026-01-15" } },
  ],
  client_work: [
    { id: "C1", fields: { vibe: "panicking", ask: "logo_bigger", invoice_state: "overdue", revisions: 3, founder: "Inez", due: "2026-01-10", deposit_cleared: false } },
    { id: "C2", fields: { vibe: "chill", ask: "quick_tweak", invoice_state: "paid", revisions: 4, founder: "Zoë", due: "2026-01-11", deposit_cleared: true } },
    { id: "C3", fields: { vibe: "ghosting", ask: "ai_pivot", invoice_state: "ghosted", revisions: 5, founder: "Ana", due: "2026-01-12", deposit_cleared: false } },
    { id: "C4", fields: { vibe: "passive_aggressive", ask: "logo_bigger", invoice_state: "disputed", revisions: 6, founder: "Inez", due: "2026-01-09", deposit_cleared: true } },
    { id: "C5", fields: { vibe: "delusional", ask: "pop_more", revisions: 0, due: "2026-01-15" } },
  ],
  deployments: [
    { id: "D1", fields: { environment: "prod", outcome: "succeeded", initiator: "Ana", latency_ms: 299, deployed: "2026-01-10", canary_enabled: true } },
    { id: "D2", fields: { environment: "eu_west", outcome: "pending", initiator: "Zoë", latency_ms: 300, deployed: "2026-01-11", canary_enabled: false } },
    { id: "D3", fields: { environment: "sandbox", outcome: "rolled_back", initiator: "Ana", latency_ms: 301, deployed: "2026-01-12", canary_enabled: true } },
    { id: "D4", fields: { environment: "prod", outcome: "running", initiator: "Minh", latency_ms: 0, deployed: "2026-01-09", canary_enabled: false } },
    { id: "D5", fields: { environment: "preview", outcome: "aborted", latency_ms: 1000, deployed: "2026-01-15" } },
  ],
};
