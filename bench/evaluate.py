"""Score filter predictions against WikiSQL gold conditions.

Prediction file (jsonl), one line per example index:
  {"i": 12, "conds": [[col_idx, "=", "value"], ...] | null, "conf": 0.93, "ms": 0.4}
conds null = abstained. Values are compared after prepare.norm_value.

Run: python3 bench/evaluate.py test bench/out/preds/rules.test.jsonl [more.jsonl ...]
A correct answer = the exact set of (column, op, value) conditions.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare import norm_value  # noqa: E402

OUT = Path(__file__).parent / "out"


def gold(split: str) -> list[set]:
    d = json.load(open(OUT / f"{split}.json"))
    return [{(c, o, v) for c, o, v in e["conds"]} for e in d["examples"]]


def score(golds: list[set], preds: dict[int, dict]) -> dict:
    rows = []
    for i, p in preds.items():
        ans = p.get("conds")
        ok = ans is not None and {(int(c), o, norm_value(v)) for c, o, v in ans} == golds[i]
        rows.append((p.get("conf", 1.0), ans is not None, ok, p.get("ms")))
    n = len(rows)
    answered = [r for r in rows if r[1]]
    correct = sum(r[2] for r in rows)
    wrong = len(answered) - correct
    out = {
        "n": n,
        "answered": len(answered) / n,
        "accuracy": correct / n,            # correct out of everything asked
        "wrong_rate": wrong / n,            # confidently wrong out of everything asked
        "precision": correct / max(1, len(answered)),  # right when it answers
        "ms_p50": statistics.median([r[3] for r in rows if r[3] is not None] or [0]),
    }
    # Coverage at fixed precision: sweep confidence threshold over answered rows.
    ranked = sorted(answered, key=lambda r: -r[0])
    for target in (0.95, 0.99):
        best, hits = 0, 0
        for k, r in enumerate(ranked, 1):
            hits += r[2]
            if hits / k >= target:
                best = k
        out[f"cov@{int(target * 100)}"] = best / n
    return out


def load_preds(path: str) -> dict[int, dict]:
    return {p["i"]: p for p in map(json.loads, open(path))}


if __name__ == "__main__":
    split, paths = sys.argv[1], sys.argv[2:]
    golds = gold(split)
    print(f"{'method':28} {'n':>6} {'answer%':>8} {'acc%':>6} {'wrong%':>7} {'prec%':>6} {'cov@95':>7} {'cov@99':>7} {'ms p50':>8}")
    for p in paths:
        s = score(golds, load_preds(p))
        print(f"{Path(p).stem:28} {s['n']:6} {s['answered']*100:8.1f} {s['accuracy']*100:6.1f} {s['wrong_rate']*100:7.1f} "
              f"{s['precision']*100:6.1f} {s['cov@95']*100:7.1f} {s['cov@99']*100:7.1f} {s['ms_p50']:8.2f}")
