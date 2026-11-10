"""Cascade: Sieve answers when its confidence clears a threshold, else the query goes to a cloud LLM.
Threshold picked on dev (target precision), applied unchanged to test.

  python3 bench/cascade.py out/fb.dev.json out/preds/sieve3b.fb-dev.jsonl out/fb.test.json out/preds/sieve3.fb-test.jsonl out/preds/haiku.fb-test.jsonl
"""
import json
import sys

from filterbench import col_numeric, run


def correct(bench, preds):
    d = json.load(open(bench))
    out = {}
    for p in preds:
        e = d["examples"][p["i"]]
        t = d["tables"][e["table"]]
        num = [col_numeric(t["rows"], j) for j in range(len(t["header"]))]
        try:
            out[p["i"]] = bool(p["conds"]) and run(t["rows"], [[int(c), o, v] for c, o, v in p["conds"]], num) == run(t["rows"], e["conds"], num)
        except (ValueError, KeyError, TypeError, IndexError):
            out[p["i"]] = False
    return out


def load(p):
    return [json.loads(l) for l in open(p)]


def threshold(dev_bench, dev_preds, target):
    ok = correct(dev_bench, dev_preds)
    ranked = sorted((p for p in dev_preds if p["conds"]), key=lambda p: -p["conf"])
    best, hits = 1.0, 0
    for k, p in enumerate(ranked, 1):
        hits += ok[p["i"]]
        if hits / k >= target:
            best = p["conf"]
    return best


if __name__ == "__main__":
    dev_bench, dev_preds, test_bench, sieve, cloud = sys.argv[1], load(sys.argv[2]), sys.argv[3], load(sys.argv[4]), load(sys.argv[5])
    s_ok, c_ok = correct(test_bench, sieve), correct(test_bench, cloud)
    n = len(sieve)
    print(f"cloud only: {sum(c_ok.values()) / n * 100:.1f}% right, 100% of queries call the LLM")
    for target in (0.90, 0.95, 0.98):
        th = threshold(dev_bench, dev_preds, target)
        local = [p for p in sieve if p["conds"] and p["conf"] >= th]
        li = {p["i"] for p in local}
        right = sum(s_ok[i] for i in li) + sum(c_ok[i] for i in c_ok if i not in li)
        print(f"dev target {target:.0%} (conf>={th:.3f}): Sieve answers {len(li) / n * 100:.1f}% locally at "
              f"{sum(s_ok[i] for i in li) / max(1, len(li)) * 100:.1f}% precision; cascade {right / n * 100:.1f}% right, "
              f"LLM calls cut {len(li) / n * 100:.0f}%")
