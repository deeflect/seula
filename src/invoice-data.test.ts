import assert from "node:assert/strict";
import { exportCsv, findInvoices, importInvoices, sampleInvoices } from "./invoice-data";

const defaultResult = findInvoices("overdue and balance at least $1k", sampleInvoices);
assert.equal(defaultResult.error, "");
assert.deepEqual(defaultResult.matches.map((row) => row.invoice), ["INV-1048", "INV-1051", "INV-1054"]);
assert.equal(defaultResult.matches.reduce((sum, row) => sum + row.balance_usd, 0), 5150);
assert.equal(findInvoices("overdue", sampleInvoices).matches.length, 4);
assert.deepEqual(findInvoices("overdue and balance beneath $1k", sampleInvoices).matches.map((row) => row.invoice), ["INV-1056"]);
assert.ok(findInvoices("overdue banana", sampleInvoices).error);
assert.equal(findInvoices("balance > $10k", sampleInvoices).matches.length, 0);
for (const query of ["at least 1k", "at least1k", "At   Least $1k"]) {
  const result = findInvoices(query, sampleInvoices);
  assert.equal(result.error, "", query);
  assert.deepEqual(result.matches.map((row) => row.invoice), ["INV-1048", "INV-1051", "INV-1054", "INV-1058", "INV-1060"], query);
  assert.equal(result.matches.reduce((sum, row) => sum + row.balance_usd, 0), 9600, query);
  assert.deepEqual(result.clauses, [{ field: "balance_usd", cmp: "gte", value: "1000", negated: false }], query);
}
assert.deepEqual(findInvoices("over$1k", sampleInvoices).matches.map((row) => row.invoice), ["INV-1048", "INV-1051", "INV-1058", "INV-1060"]);
assert.deepEqual(findInvoices("under500", sampleInvoices).matches.map((row) => row.invoice), ["INV-1056", "INV-1041", "INV-1043"]);
for (const query of ["from fieldwork", "from Fieldwork Co", "from fieldwork and at least1k", "overdue and from fieldwork and at least1k"]) {
  const result = findInvoices(query, sampleInvoices);
  assert.equal(result.error, "", query);
  assert.deepEqual(result.matches.map((row) => row.invoice), ["INV-1051"], query);
  assert.equal(result.matches[0].balance_usd, 1750, query);
  assert.ok(result.clauses.some((clause) => clause.field === "client" && clause.cmp === "contains" && clause.value.includes("fieldwork")), query);
}
const imported = importInvoices('invoice,client,balance_usd,days_late,status\nINV-2,"Acme 2 and Sons",1800,8,overdue\nINV-3,"The Long Studio Name",900,0,due');
assert.deepEqual(findInvoices('from "Acme 2 and Sons"', imported).matches.map((row) => row.invoice), ["INV-2"]);
assert.deepEqual(findInvoices("from The Long Studio Name", imported).matches.map((row) => row.invoice), ["INV-3"]);
assert.deepEqual(findInvoices("from Acme 2", imported).matches.map((row) => row.invoice), ["INV-2"]);
assert.deepEqual(findInvoices('from "overdue and balance > $1k"', imported).matches, []);
for (const query of ["from fieldwork overdue", "from fieldwork 500", "at least 1k extra 500", "at least 5 days", "under 2gb", "from fieldwork or overdue", "at least 1k or overdue", "(at least 1k)", "not from fieldwork", "from fieldwork and", "from fieldwork and at least -1k", `${"overdue and ".repeat(24)}at least 1k`]) {
  const result = findInvoices(query, sampleInvoices);
  assert.ok(result.error, query);
  assert.deepEqual(result.matches, [], query);
}
assert.match(findInvoices("at least 1k or overdue", sampleInvoices).error, /OR is not supported/);
assert.equal(findInvoices("", sampleInvoices).error, "Enter a filter to search.");

const csv = "invoice,client,balance_usd,days_late,status\nINV-1,Acme,12.50,2,overdue";
assert.deepEqual(importInvoices(csv), [{ invoice: "INV-1", client: "Acme", balance_usd: 12.5, days_late: 2, status: "overdue" }]);
for (const balance of ["0.29", "1.1", "19.99"]) {
  const row = importInvoices(`invoice,client,balance_usd,days_late,status\nINV-1,Acme,${balance},2,overdue`)[0];
  assert.equal(row.balance_usd, Number(balance));
  assert.equal(importInvoices(`[{"invoice":"INV-1","client":"Acme","balance_usd":${balance},"days_late":2,"status":"overdue"}]`)[0].balance_usd, Number(balance));
}
for (const bad of [
  "invoice,client,balance_usd,days_late,status\nINV-1,Acme,-1,2,overdue",
  "invoice,client,balance_usd,days_late,status\nINV-1,Acme,2,2,late",
  "invoice,client,balance_usd,days_late,status\nINV-1,Acme,2,2,overdue\nINV-1,Other,3,3,due",
  '[{"invoice":null,"client":"Acme","balance_usd":2,"days_late":2,"status":"due"}]',
  '[{"invoice":"INV-1","client":"Acme","balance_usd":false,"days_late":2,"status":"due"}]',
  '[{"invoice":"INV-1","client":"Acme","balance_usd":0.001,"days_late":2,"status":"due"}]',
  "invoice,client,balance_usd,days_late,status\nINV-1,Acme,1.234,2,due",
  '[{"invoice":"INV-1","client":"Acme","balance_usd":"1.230","days_late":2,"status":"due"}]',
  '[{"invoice":"INV-1","client":"Acme","balance_usd":90071992547410,"days_late":2,"status":"due"}]',
]) assert.throws(() => importInvoices(bad), bad);

const output = exportCsv([{ invoice: "=cmd", client: 'Acme, "West"', balance_usd: 12.5, days_late: 2, status: "overdue" }]);
assert.match(output, /"'=cmd"/);
assert.match(output, /"Acme, ""West"""/);
assert.deepEqual(importInvoices(output), [{ invoice: "'=cmd", client: 'Acme, "West"', balance_usd: 12.5, days_late: 2, status: "overdue" }]);
console.log("invoice demo: passed");
