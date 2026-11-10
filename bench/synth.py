"""Template-written power-user queries ("home team: liverpool, attendance>=16k") for training.

LLM writers rarely produce the symbol style, and never enough look-alike headers
(Home team / Away team). Conditions are sampled like FilterBench; no LLM involved.

  python3 bench/synth.py 4200 bench/out/fb.synth2.json          # training, 30% with an or-list
  python3 bench/synth.py 300 bench/out/fb.ordev.json dev 1      # held-out or-list probe (template-written)
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from align import align  # noqa: E402
from filterbench import ROOT, col_numeric, parse_number, run, sample_conds  # noqa: E402
from prepare import tokenize  # noqa: E402

SYM = {"=": ["=", " = ", ":", ": ", " is ", " "], "!=": ["!=", " != ", "<>", " not ", " isnt ", " is not ", " except ", " other than "],
       "contains": [":", ": ", "~", " has ", " contains ", " includes "], ">": [">", " > ", " over ", " above "],
       ">=": [">=", " >= ", " at least ", " min "], "<": ["<", " < ", " under ", " below "], "<=": ["<=", " <= ", " at most ", " max "]}
JOIN = [", ", ", ", " ", " and ", " & ", "; ", " n ", ","]
base = lambda h: re.sub(r"\s*\(.*?\)", "", h).strip().lower()  # noqa: E731


def header(rng, h, all_headers):
    f = rng.random()
    if f < 0.45:
        return h
    b = base(h)
    return b if f > 0.8 and b and sum(base(x) == b for x in all_headers) == 1 else h.lower()


def value(rng, op, v):
    n = parse_number(v)
    if n is not None:
        if n >= 1000 and n % 100 == 0 and rng.random() < 0.4:
            return "%gk" % (n / 1000)
        return f"{int(n):,}" if n >= 1000 and n.is_integer() and rng.random() < 0.3 else v
    f = rng.random()
    return v.lower() if f < 0.5 else f'"{v}"' if f < 0.65 and op in ("=", "!=") and '"' not in v else v


def any_of(rng, t):
    """A list on one text column: "overdue or pending" (any of them) or "not a or b" (none of them)."""
    rows, numeric = t["rows"], [col_numeric(t["rows"], j) for j in range(len(t["header"]))]
    j = rng.randrange(len(numeric))
    cells = sorted({str(r[j]).strip() for r in rows if 0 < len(tokenize(str(r[j]))) <= 4})
    k = rng.choice([2, 2, 2, 3])
    if numeric[j] or len(cells) <= k:
        return None
    vals = rng.sample(cells, k)
    toks = [set(tokenize(v)) for v in vals]
    if any(toks[a] & toks[b] for a in range(k) for b in range(a)):
        return None
    op = "=" if rng.random() < 0.75 else "!="
    conds = [[j, op, v] for v in vals]
    vs = [v.lower() if rng.random() < 0.6 else v for v in vals]
    h = header(rng, t["header"][j], t["header"])
    if op == "=":
        lst = rng.choice([" or ", " or ", "/", ", "]).join(vs) if k == 2 else rng.choice([f"{vs[0]}, {vs[1]} or {vs[2]}", ", ".join(vs), " or ".join(vs), "/".join(vs)])
        text = rng.choice(["", "", "either ", h + "=", h + ": ", h + " is ", h + " ", h + " in "]) + lst
    else:
        lst = rng.choice([" or ", " or ", ", "]).join(vs) if k == 2 else rng.choice([f"{vs[0]}, {vs[1]} or {vs[2]}", " or ".join(vs)])
        text = rng.choice(["not ", "except ", "no ", "excluding ", h + " not ", h + "!=", h + " isnt "]) + lst
        if k == 2 and rng.random() < 0.15:
            text = f"neither {vs[0]} nor {vs[1]}"
    return j, conds, text


def main(n, out_path, split="train", share_or=0.3, seed=11):
    rng = random.Random(seed)
    tables = json.load(open(ROOT / f"{split}.json"))["tables"]
    ids = [k for k, t in tables.items() if len(t["rows"]) >= 5]
    examples = []
    while len(examples) < n:
        tid = rng.choice(ids)
        t = tables[tid]
        numeric = [col_numeric(t["rows"], j) for j in range(len(t["header"]))]
        want = rng.random() < share_or
        group = any_of(rng, t) if want else None
        if want and not group:
            continue
        conds = sample_conds(rng, t, ranges=rng.random() < 0.2) if not group or rng.random() < 0.5 else []
        if conds is None or (not group and not conds) or (group and any(c[0] == group[0] for c in conds)):
            continue
        j = rng.choice(JOIN)
        parts = [header(rng, t["header"][c], t["header"]) + (rng.choice(SYM[op]) + value(rng, op, v) if op != ">=" or rng.random() > 0.2
                                                             else " " + value(rng, op, v) + "+") for c, op, v in conds]  # "attendance 16k+"
        if group:
            parts.insert(rng.randrange(len(parts) + 1), group[2])
            conds = conds + group[1]
            j = rng.choice([", ", "; ", " and ", " "]) if j in (",", ", ") else j  # a comma would read as one more list item
            hit = run(t["rows"], conds, numeric)
            if not hit or len(hit) == len(t["rows"]):
                continue
        q = j.join(parts)
        if align(tokenize(q), conds):
            examples.append({"table": tid, "conds": conds, "q": q, "style": 3})
    json.dump({"model": "template", "split": split, "examples": examples, "tables": {e["table"]: tables[e["table"]] for e in examples}}, open(out_path, "w"))
    print(f"wrote {len(examples)} to {out_path}")
    for e in examples[:14]:
        print("  ", e["q"])


if __name__ == "__main__":  # synth.py N out.json [split] [share of queries with an or-list]
    main(int(sys.argv[1]), sys.argv[2], *sys.argv[3:4], *map(float, sys.argv[4:5]))
