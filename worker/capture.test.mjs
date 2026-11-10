import assert from "node:assert/strict";
import worker from "./capture.js";

const kept = new Map(), env = { CAPTURE: { put: async (k, v) => void kept.set(k, v) } };
const send = (body, init = {}) => worker.fetch(new Request("https://x/capture", { method: "POST", body, ...init }), env).then((r) => r.status);
const good = JSON.stringify({ verdict: "wrong", query: "overdue", header: ["status"], conds: [[0, "=", "overdue"]], conf: 0.9 });

assert.equal(await send(good), 204);
assert.deepEqual([...kept.values()], [good]);
assert.equal(await send("not json"), 400);
assert.equal(await send(JSON.stringify({ verdict: "maybe", query: "x", header: [] })), 400);
assert.equal(await send(JSON.stringify({ verdict: "right", query: "x".repeat(9000), header: [] })), 413);
assert.equal((await worker.fetch(new Request("https://x/capture"), env)).status, 404);
assert.equal((await worker.fetch(new Request("https://x/other", { method: "POST", body: good }), env)).status, 404);
assert.equal(kept.size, 1);
console.log("capture ok");
