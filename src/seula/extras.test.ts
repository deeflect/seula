// Run: npx tsx src/seula/extras.test.ts
import assert from "node:assert/strict";
import type { Table } from "./model.js";
import { readFileSync } from "node:fs";
import { dateCols, dateOf, extras, isIso, matchesDate } from "./extras.js";
import { Seula, filter, rowsOf } from "./index.js";

let n = 0;
const eq = (a: unknown, b: unknown, msg = "") => (assert.deepEqual(a, b, msg), n++);
const today = new Date(2026, 10, 9); // Monday 9 Nov 2026
const day = (s: unknown) => { const d = dateOf(s); return d === null ? null : new Date(d * 864e5).toISOString().slice(0, 10); };

const inv: Table = {
  header: ["Invoice", "Client", "Balance", "Due date", "Status"],
  types: ["text", "text", "real", "text", "text"],
  rows: [["A1", "Acme", 120, "2026-10-01", "overdue"], ["A2", "Bolt", 80, "2026-11-12", "open"]],
};
const nfl: Table = {
  header: ["Week", "Date", "Opponent", "Result", "Attendance"],
  types: ["real", "text", "text", "text", "real"],
  rows: [[1, "September 21, 1975", "Bears", "W 20-7", 51000], [2, "September 28, 1975", "Lions", "L 3-10", 48000]],
};
const two: Table = { // two date columns
  header: ["Invoice", "Issued", "Due date", "Balance"],
  types: ["text", "text", "text", "real"],
  rows: [["A1", "2026-09-01", "2026-10-01", 5], ["A2", "2026-09-15", "2026-10-15", 7]],
};
const guard: Table = {
  header: ["Top scorer", "First elected", "Highest point", "Most recent result", "Date", "Goals"],
  types: ["text", "text", "text", "text", "text", "real"],
  rows: [["Messi", "1990", "K2", "win", "2026-01-01", 30]],
};
const plain: Table = { header: ["Name", "Score"], types: ["text", "real"], rows: [["a", 1], ["b", 2]] };
const x = (q: string, t: Table) => extras(q, t, today);

// dateOf
eq(day("2026-10-01"), "2026-10-01");
eq(day("2026-10-01T08:30:00Z"), "2026-10-01");
eq(day("September 21, 1975"), "1975-09-21");
eq(day("Sep 21 1975"), "1975-09-21");
eq(day("21 September 1975"), "1975-09-21");
eq(day("21 Sept 2008"), "2008-09-21");
eq(day("September21,1975"), "1975-09-21");
eq(day("9/21/1975"), "1975-09-21");
eq(day("March 2024"), "2024-03-01");
eq(day("dec 25 2025"), "2025-12-25");
for (const bad of [1975, "1975", "42", "hello world", "2026-02-30", "13/01/2026", "Feb 30 2024"]) eq(dateOf(bad), null, String(bad));
eq(isIso("2026-10-01"), true), eq(isIso("2026-10-1"), false);

// dateCols
eq(dateCols(inv), [false, false, false, true, false]);
eq(dateCols(nfl), [false, true, false, false, false]);
eq(dateCols({ header: ["d"], types: ["text"], rows: [[""], [""]] }), [false]);

// date phrases (Due date = col 3)
const D = (op: string, v: string) => [3, op, v];
eq(x("overdue today", inv), { rest: "overdue", conds: [D("=", "2026-11-09")] });
eq(x("yesterday", inv).conds, [D("=", "2026-11-08")]);
eq(x("Tomorrow", inv).conds, [D("=", "2026-11-10")]);
eq(x("due this week", inv), { rest: "due", conds: [D(">=", "2026-11-09"), D("<=", "2026-11-15")] });
eq(x("last week", inv).conds, [D(">=", "2026-11-02"), D("<=", "2026-11-08")]);
eq(x("next week", inv).conds, [D(">=", "2026-11-16"), D("<=", "2026-11-22")]);
eq(x("this month", inv).conds, [D(">=", "2026-11-01"), D("<=", "2026-11-30")]);
eq(x("previous quarter", inv).conds, [D(">=", "2026-07-01"), D("<=", "2026-09-30")]);
eq(x("next year", inv).conds, [D(">=", "2027-01-01"), D("<=", "2027-12-31")]);
eq(x("in the last 30 days", inv).conds, [D(">=", "2026-10-10"), D("<=", "2026-11-09")]);
eq(x("past 2 months", inv).conds, [D(">=", "2026-09-09"), D("<=", "2026-11-09")]);
eq(x("next 2 weeks", inv).conds, [D(">=", "2026-11-09"), D("<=", "2026-11-23")]);
eq(x("older than 90 days", inv).conds, [D("<", "2026-08-11")]);
eq(x("more than 1 year ago", inv).conds, [D("<", "2025-11-09")]);
eq(x("3 days ago", inv), { rest: "3 days ago", conds: [] });
eq(x("since 2026-10-01", inv).conds, [D(">=", "2026-10-01")]);
eq(x("before March", inv).conds, [D("<", "2026-03-01")]);
eq(x("until 2026-01-31", inv).conds, [D("<=", "2026-01-31")]);
eq(x("through Feb 2026", inv).conds, [D("<=", "2026-02-28")]);
eq(x("Acme in December", inv), { rest: "Acme", conds: [D(">=", "2025-12-01"), D("<=", "2025-12-31")] });
eq(x("open March 2024", inv).conds, [D(">=", "2024-03-01"), D("<=", "2024-03-31")]);
const F = (op: string, v: string) => [1, op, v];
eq(x("games since October 1975", nfl), { rest: "games", conds: [F(">=", "1975-10-01")] });
eq(x("after 21 Sept 1975", nfl).conds, [F(">", "1975-09-21")]);
eq(x("on September 28, 1975", nfl).conds, [F("=", "September 28, 1975")]); // the table's own text for that day
eq(x("on September 30, 1975", nfl).conds, [F("=", "1975-09-30")]); // no such cell
eq(x("bears 21 sept 1975", nfl), { rest: "bears", conds: [F("=", "September 21, 1975")] }); // bare full date, day not left behind
eq(x("not on September 28, 1975", nfl), { rest: "not on September 28, 1975", conds: [] }); // negated: the model's job
eq(x("neither 21 sept 1975 nor 28 sept 1975", nfl).conds, []); // a negated list stays whole for the model
for (const q of [`Date!="21 sept 1975"`, "games that weren't on 21 sept 1975", "don't show anything from September 21, 1975", "bears !september21,1975", `bears -date:"september 21 1975"`])
  eq(x(q, nfl).conds, []); // more ways to say not
eq(x("games in september", nfl), { rest: "games in september", conds: [] }); // months spelled in the cells: any September, the model's job
eq(x("issued in september", two).conds, [[1, ">=", "2026-09-01"], [1, "<=", "2026-09-30"]]); // ISO cells: the latest September
// Left whole for the model: a time after the date, a non-date column named right before it, a day two date columns share, "top" that is not top-N.
const both: Table = { ...two, rows: [["A1", "2026-09-01", "2026-09-15", 5], ["A2", "2026-09-15", "2026-10-15", 7]] };
eq(x("21 sept 1975 20:30", nfl).conds, []);
eq(x("opponent is 21 sept 1975", nfl).conds, []);
eq(x("exclude 21 sept 1975", nfl).conds, []);
eq(x("2026-09-15", both).conds, []);
eq(x("2026-09-01", both).conds, [[1, "=", "2026-09-01"]]); // only Issued has that day
eq(x("issued 2026-09-15", both).conds, [[1, "=", "2026-09-15"]]); // column named
eq(x("week on top <= 1", nfl).limit, undefined);
// A bare year needs a date column named next to it; alone it could be a number for the model.
eq(x("issued after 2025", two), { rest: "", conds: [[1, ">", "2025-12-31"]] });
eq(x("due date in 2026", two), { rest: "", conds: [[2, ">=", "2026-01-01"], [2, "<=", "2026-12-31"]] });
eq(x("issued since 2026, balance over 5", two), { rest: "balance over 5", conds: [[1, ">=", "2026-01-01"]] });
eq(x("after 2025", two).conds, []);
eq(x("balance after 2025", two).conds, []);
eq(x("newest first", two), { rest: "", conds: [], sort: [2, "desc"] });
eq(x("highest balance first", two), { rest: "", conds: [], sort: [3, "desc"] });
eq(x("date not 21 sept 1975, 28 sept 1975", nfl).conds, []);
eq(x("21 sept 1975/28 sept 1975", nfl).conds, [F("=", "September 21, 1975"), F("=", "September 28, 1975")]); // slash list: two full dates
eq(x("15 sept 2026", two).conds, [[1, "=", "2026-09-15"]]); // no column named: the date column that has the day
eq(x("date != sept 1975", nfl), { rest: "date != sept 1975", conds: [] });
eq(x("date!=21 september 1975", nfl), { rest: "date!=21 september 1975", conds: [] }); // nor may "september 1975" be taken from it
eq(x("games from September 28, 1975", nfl).conds, [F("=", "September 28, 1975")]);
eq(x("between 9/21/1975 and Oct 5 1975 and bears", nfl), { rest: "bears", conds: [F(">=", "1975-09-21"), F("<=", "1975-10-05")] });
eq(x("games in 1975", nfl), { rest: "games in 1975", conds: [] });
eq(x("before 1980", nfl), { rest: "before 1980", conds: [] });

// two date columns: header mention picks the column and is consumed
eq(x("issued last week", two), { rest: "", conds: [[1, ">=", "2026-11-02"], [1, "<=", "2026-11-08"]] });
eq(x("balance over 5 and due date is before 2026-10-10", two), { rest: "balance over 5", conds: [[2, "<", "2026-10-10"]] });
eq(x("last week", two).conds, [[2, ">=", "2026-11-02"], [2, "<=", "2026-11-08"]]);

// no date column -> nothing date-related
eq(x("last week score above 5", plain), { rest: "last week score above 5", conds: [] });

// sort / limit
eq(x("top 5 by balance", inv), { rest: "", conds: [], sort: [2, "desc"], limit: 5 });
eq(x("open top 3", inv), { rest: "open", conds: [], sort: [2, "desc"], limit: 3 });
eq(x("bottom 2 by attendance", nfl), { rest: "", conds: [], sort: [4, "asc"], limit: 2 });
// after "by", part of a header names the column; two columns that fit equally are left to the model
const units: Table = { header: ["Client", "Balance USD", "Days late"], types: ["text", "real", "real"], rows: [["Acme", 120, 3], ["Bolt", 80, 0]] };
eq(x("top 5 by balance", units), { rest: "", conds: [], sort: [1, "desc"], limit: 5 });
eq(x("sort by balance desc", units), { rest: "", conds: [], sort: [1, "desc"] });
eq(x("top 2 by late", units), { rest: "", conds: [], sort: [2, "desc"], limit: 2 });
const twoBalances: Table = { ...units, header: ["Client", "Balance USD", "Balance EUR"] };
eq(x("top 5 by balance", twoBalances), { rest: "top 5 by balance", conds: [] });
eq(x("top 5 by balance eur", twoBalances), { rest: "", conds: [], sort: [2, "desc"], limit: 5 });
eq(x("top 3", nfl), { rest: "top 3", conds: [] });
eq(x("top 3 by opponent", nfl), { rest: "top 3 by opponent", conds: [] });
eq(x("bears sorted by weeks descending", nfl), { rest: "bears", conds: [], sort: [0, "desc"] });
eq(x("order by opponent", nfl), { rest: "", conds: [], sort: [2, "asc"] });
eq(x("highest attendance", nfl), { rest: "", conds: [], sort: [4, "desc"] });
eq(x("Acme lowest balance", inv), { rest: "Acme", conds: [], sort: [2, "asc"] });
eq(x("latest games", nfl), { rest: "games", conds: [], sort: [1, "desc"] });
eq(x("oldest unpaid", inv), { rest: "unpaid", conds: [], sort: [3, "asc"] });
eq(x("most recent issued", two), { rest: "", conds: [], sort: [1, "desc"] });

// guard: trigger words inside a header mention are left alone
eq(x("top scorer is messi", guard), { rest: "top scorer is messi", conds: [] });
eq(x("first elected 1990", guard), { rest: "first elected 1990", conds: [] });
eq(x("highest point is K2", guard), { rest: "highest point is K2", conds: [] });
eq(x("most recent result win", guard), { rest: "most recent result win", conds: [] });
eq(x("most goals", guard), { rest: "", conds: [], sort: [5, "desc"] });

// mixed
eq(x("overdue clients last 30 days top 5 by balance", inv), {
  rest: "overdue clients", conds: [D(">=", "2026-10-10"), D("<=", "2026-11-09")], sort: [2, "desc"], limit: 5,
});

// matchesDate
const c = "September 21, 1975";
eq(matchesDate(c, "=", "1975-09-21"), true);
eq(matchesDate(c, "!=", "1975-09-21"), false);
eq(matchesDate(c, ">", "1975-09-20"), true);
eq(matchesDate(c, ">=", "1975-09-21"), true);
eq(matchesDate(c, "<", "1975-09-21"), false);
eq(matchesDate(c, "<=", "1975-09-21"), true);
eq(matchesDate("TBD", "!=", "1975-09-21"), false);

// trigger words that are a cell value
const awards: Table = { header: ["Year", "Award", "Result"], types: ["real", "text", "text"], rows: [[2010, "Top 10 Most Popular Male Artiste", "Won"], [2011, "Best Actor", "Nominated"]] };
eq(extras("top 10 most popular male artiste won", awards), { rest: "top 10 most popular male artiste won", conds: [] });
eq(extras("top 1 won", awards), { rest: "won", conds: [], sort: [0, "desc"], limit: 1 });

// yes/no columns
const staff: Table = {
  header: ["Name", "Department", "Remote", "Is active", "Paid", "Status"],
  types: ["text", "text", "text", "text", "text", "text"],
  rows: [["Ann", "Sales", "Yes", "true", "yes", "paid"], ["Bo", "Support", "No", "true", "no", "open"]],
};
eq(extras("remote sales", staff), { rest: "sales", conds: [[2, "=", "Yes"]] });
eq(extras("sales, not remote", staff), { rest: "sales", conds: [[2, "=", "No"]] });
eq(extras("non-remote", staff).conds, [[2, "=", "No"]]);
eq(extras("not active", staff).conds, [[3, "!=", "true"]]); // no "false" cell to name
eq(extras("remote: yes", staff), { rest: "remote: yes", conds: [] }); // value given: the model's job
eq(extras("paid", staff), { rest: "paid", conds: [] }); // also a Status value: the model's job

// end to end: pre-pass + model + row order
const model = new Seula(readFileSync(new URL("./weights.bin", import.meta.url)));
const due: Table = { ...inv, rows: [...inv.rows, ["A3", "Core", 900, "2026-10-20", "overdue"], ["A4", "Dune", 300, "2026-08-02", "overdue"]] };
eq(rowsOf(model, due, filter(model, "overdue since oct 2026 top 1 by balance", due, today)), [2]);
eq(rowsOf(model, due, filter(model, "sort by client desc", due, today)), [3, 2, 1, 0]);
eq(rowsOf(model, due, { conds: [[4, "=", "overdue"], [4, "=", "open"], [2, ">", "100"]] }), [0, 2, 3]); // any-of on one column
eq(rowsOf(model, due, { conds: [[1, "contains", "acme"], [1, "=", "Dune"]] }), [0, 3]); // "acme or dune": contains counts too

console.log(`extras: ${n} checks passed`);
