// Node entry: same API, but load() with no argument reads weights.bin from disk (Node's fetch has no file: URLs).
import { readFile } from "node:fs/promises";
import { load as loadAny } from "./index.js";

export * from "./index.js";
export const load = async (weights?: Parameters<typeof loadAny>[0]) => loadAny(weights ?? (await readFile(new URL("./weights.bin", import.meta.url))));
