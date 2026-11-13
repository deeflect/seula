// Parity: TS Seula vs Python (bench/export_seula.py writes the fixture with the same int8 weights).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Seula, fmtNum, parseNumber, tokenize } from "../src/seula/model";

const root = new URL("..", import.meta.url);
const model = new Seula(readFileSync(new URL("src/seula/weights.bin", root)));
const fx = JSON.parse(readFileSync(new URL("spike/seula_parity.json", root), "utf8"));

for (const [s, toks] of fx.units.tok) assert.deepEqual(tokenize(s), toks, `tokenize ${JSON.stringify(s)}`);
for (const [s, want] of fx.units.num) { const n = parseNumber(s); assert.equal(n === null ? null : fmtNum(n), want, `parse_number ${s}`); }

let maxDiff = 0;
const ms: number[] = [];
for (const c of fx.cases) {
  const r = model.parse(c.q, c.table, { executionGuided: true }); // historical execution-guided Python parity
  assert.deepEqual(r.conds, c.conds, `conds for ${JSON.stringify(c.q)}`);
  maxDiff = Math.max(maxDiff, Math.abs(r.conf - c.conf));
  const t0 = performance.now();
  for (let i = 0; i < 20; i++) model.parse(c.q, c.table, { executionGuided: true });
  ms.push((performance.now() - t0) / 20);
}
assert.ok(maxDiff < 1e-3, `conf diff ${maxDiff}`);
ms.sort((a, b) => a - b);
console.log(`seula parity ok: ${fx.cases.length} cases, max |conf diff| ${maxDiff.toExponential(2)}, median ${ms[ms.length >> 1].toFixed(3)} ms/query (cached table index)`);
