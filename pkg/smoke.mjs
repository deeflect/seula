// Build, pack, install the tarball into a fresh dir, and use it from there. Run: npm run pkg:smoke
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const pkg = fileURLToPath(new URL(".", import.meta.url));
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, stdio: ["ignore", "pipe", "inherit"] }).toString().trim();
run("npm", ["run", "pkg"], join(pkg, ".."));
const tgz = join(pkg, run("npm", ["pack", "--silent"], pkg).split("\n").pop());
const dir = mkdtempSync(join(tmpdir(), "seula-smoke-"));
try {
  writeFileSync(join(dir, "package.json"), '{ "type": "module", "private": true }');
  run("npm", ["install", "--silent", "--no-audit", "--no-fund", tgz], dir);
  writeFileSync(join(dir, "check.mjs"), `
import assert from "node:assert/strict";
import { load } from "seula";
import { toSql } from "seula/sql";
import { toTanStack } from "seula/tanstack";
const s = await load();
const t = s.table([{ Client: "Acme", Balance: 1200, Status: "overdue" }, { Client: "Bolt", Balance: 80, Status: "paid" }, { Client: "Cove", Balance: 3000, Status: "paid" }]);
const r = t.parse("paid and balance over 1k");
assert.deepEqual(t.apply(r).map((x) => x.Client), ["Cove"]);
assert.equal(toTanStack(r).columnFilters.length, 2);
assert.deepEqual(toSql(r).params.sort(), [1000, "paid"].sort());
console.log("smoke ok:", JSON.stringify(r.conditions));
`);
  console.log(run("node", ["check.mjs"], dir));
} finally {
  rmSync(dir, { recursive: true, force: true });
  rmSync(tgz, { force: true });
}
