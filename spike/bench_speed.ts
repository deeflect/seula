// Speed: median ms/query of the TS runtime on FilterBench dev (each table indexed once, then reused).
// Also writes the predictions so they can be scored: npx tsx spike/bench_speed.ts [bench.json] [out.jsonl]
import { readFileSync, writeFileSync } from "node:fs";
import { Seula } from "../src/seula/model";

const root = new URL("..", import.meta.url);
const model = new Seula(readFileSync(new URL("src/seula/weights.bin", root)));
const d = JSON.parse(readFileSync(process.argv[2] ?? new URL("bench/out/fb.dev.json", root), "utf8"));
const tables: Record<string, any> = {};
for (const [k, t] of Object.entries<any>(d.tables)) tables[k] = { header: t.header, rows: t.rows, types: t.types };
const out: string[] = [], ms: number[] = [];
for (let rep = 0; rep < 3; rep++)
  d.examples.forEach((e: any, i: number) => {
    const t = tables[e.table];
    if (rep === 0) model.parse(e.q, t); // warm: table index + JIT
    const t0 = performance.now();
    const r = model.parse(e.q, t);
    if (rep === 2) { ms.push(performance.now() - t0); out.push(JSON.stringify({ i, conds: r.conds, conf: r.conf })); }
  });
ms.sort((a, b) => a - b);
console.log(`${ms.length} queries: median ${ms[ms.length >> 1].toFixed(3)} ms, p90 ${ms[Math.floor(ms.length * 0.9)].toFixed(3)} ms`);
if (process.argv[3]) writeFileSync(process.argv[3], out.join("\n") + "\n");
