"""Cloud LLM reference on FilterBench (Claude via `claude -p`), the quality ceiling to compare against.

  python3 bench/cloud_baseline.py out/fb.test.json claude-haiku-5-5 haiku
  python3 bench/cloud_baseline.py out/fb.test.json claude-haiku-5-5 haiku --single 30   # per-query latency

Batched runs write ms = batch wall time / batch size (amortized); --single times one query per call.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OPS = ["=", "!=", "contains", ">", ">=", "<", "<="]
PROMPT = """Turn each filter-box query into conditions on its table.
Each condition is [column_number, op, value] with op one of {ops}.
- "contains" = the column includes that word; "!=" = exclude that exact value.
- Numbers as plain numbers ("1.5k" -> "1500"). Text values: copy the real cell value from the table when the query refers to one.
Return ONLY a JSON array with one array of conditions per item, same order.

{items}"""


def item(t, q):
    cols = "; ".join(f"{k}: {h}" for k, h in enumerate(t["header"]))
    rows = "\n".join(" | ".join(str(c)[:40] for c in r) for r in t["rows"][:3])
    return f"Columns: {cols}\nExample rows:\n{rows}\nQuery: {q}"


def call(model, items):
    text = PROMPT.format(ops=OPS, items="\n\n".join(f"ITEM {i + 1}\n{x}" for i, x in enumerate(items)))
    out = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json", text], capture_output=True, text=True, timeout=300)
    res = json.loads(out.stdout)["result"]
    arr = json.loads(res[res.index("["): res.rindex("]") + 1])
    return arr if len(arr) == len(items) else None


def main(bench, model, name, single=0):
    d = json.load(open(bench))
    ex = d["examples"]
    idx = list(range(len(ex)))[: single or None]
    size = 1 if single else 15
    batches = [idx[k:k + size] for k in range(0, len(idx), size)]

    def work(b):
        for _ in range(2):
            try:
                t0 = time.perf_counter()
                arr = call(model, [item(d["tables"][ex[i]["table"]], ex[i]["q"]) for i in b])
                ms = (time.perf_counter() - t0) * 1000 / len(b)
                if arr:
                    return [{"i": i, "conds": [c for c in a if isinstance(c, list) and len(c) == 3 and c[1] in OPS] or None,
                             "conf": 1.0, "ms": ms} for i, a in zip(b, arr)]
            except Exception as e:  # noqa: BLE001 - retry once, then count as abstain
                print("batch failed:", e, file=sys.stderr)
        return [{"i": i, "conds": None, "conf": 0.0, "ms": None} for i in b]

    with ThreadPoolExecutor(1 if single else 6) as pool:
        rows = [r for rs in pool.map(work, batches) for r in rs]
    out = Path(bench).parent / "preds" / f"{name}.{Path(bench).stem.replace('fb.', 'fb-')}{'-single' if single else ''}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[a.index("--single") + 1]) if "--single" in a else 0)
