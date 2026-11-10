"""Turn WikiSQL into a filter-extraction benchmark.

Task: given a question and a table (header + rows), return the WHERE conditions
as a set of (column, op, value). The SELECT/aggregation part is ignored: that is
the "what to show" half, Sieve only does the "which rows" half.

Download (BSD-3, Salesforce):
  curl -L https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2 | tar xj
  mv data/*.jsonl bench/data/

Run: python3 bench/prepare.py  ->  bench/out/{train,dev,test}.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
OPS = ["=", ">", "<"]  # WikiSQL op ids 0,1,2
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def find_span(toks: list[str], vtoks: list[str], taken: set[int]) -> tuple[int, int] | None:
    n = len(vtoks)
    if n == 0:
        return None
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == vtoks and not taken & set(range(i, i + n)):
            return (i, i + n)
    return None


def norm_value(v) -> str:
    s = " ".join(tokenize(str(v)))
    try:
        f = float(s.replace(",", "").replace(" ", ""))
        return repr(f)
    except ValueError:
        return s


def load_split(split: str) -> dict:
    tables = {}
    for line in open(ROOT / "data" / f"{split}.tables.jsonl"):
        t = json.loads(line)
        tables[t["id"]] = {"header": t["header"], "types": t["types"], "rows": t["rows"]}
    examples, aligned = [], 0
    for line in open(ROOT / "data" / f"{split}.jsonl"):
        q = json.loads(line)
        conds = q["sql"]["conds"]
        if not conds:
            continue  # nothing to filter
        toks = tokenize(q["question"])
        spans, taken = [], set()
        for col, op, val in conds:
            sp = find_span(toks, tokenize(str(val)), taken)
            spans.append(sp)
            if sp:
                taken |= set(range(*sp))
        ok = all(spans)
        aligned += ok
        examples.append({
            "q": q["question"], "toks": toks, "table": q["table_id"],
            "conds": [[c, OPS[o], norm_value(v)] for c, o, v in conds],
            "spans": spans if ok else None,
        })
    used = {e["table"] for e in examples}
    return {"tables": {k: v for k, v in tables.items() if k in used}, "examples": examples, "aligned": aligned}


if __name__ == "__main__":
    (ROOT / "out").mkdir(exist_ok=True)
    for split in ["train", "dev", "test"]:
        d = load_split(split)
        n = len(d["examples"])
        print(f"{split}: {n} questions with filters, {d['aligned']} ({d['aligned'] / n:.1%}) value spans found in question")
        json.dump(d, open(ROOT / "out" / f"{split}.json", "w"))
        if split == "test":
            # Fixed 1000-question sample for slow methods (LLMs); same for every method.
            import random
            idx = sorted(random.Random(0).sample(range(n), 1000))
            json.dump(idx, open(ROOT / "out" / "test_sample.json", "w"))
