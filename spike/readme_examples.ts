// Prints what the README example queries actually produce (run: npx tsx spike/readme_examples.ts).
import { readFileSync } from "node:fs";
import { Seula } from "../src/seula/model";
import { benchTables, invoiceTable, runConds } from "../src/live-tables";

const seula = new Seula(readFileSync(new URL("../src/seula/weights.bin", import.meta.url)));
const tables = [invoiceTable, ...benchTables];
const qs: [string, string][] = [
  ["invoice", "overdue and balance at least 1k"],
  ["eu", "gdp per cap over 30k, not euro"],
  ["eu", "population over 10 million"],
  ["episodes", "directed by marcos siega"],
  ["invoice", "days late between 10 and 40"],
  ["race", "grid between 3 and 6"],
  ["eu", "richest countries"],
];
for (const [key, q] of qs) {
  const t = tables.find((x) => x.key.includes(key))!;
  const { conds, conf } = seula.parse(q, t);
  const shown = conds?.map(([c, o, v]) => `${t.header[c]} ${o} ${v}`).join("  AND  ");
  const rows = conds ? runConds(t.rows, conds, t.types.map((x) => x === "real")).length : 0;
  console.log(`${t.key.padEnd(10)} ${q.padEnd(36)} → ${shown}  (conf ${conf.toFixed(2)}, ${rows}/${t.rows.length} rows)`);
}
