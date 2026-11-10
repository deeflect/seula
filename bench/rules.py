"""Hand-written rules baseline: what a developer could build in an afternoon.

1. Values = longest question n-grams that equal a cell in the table (longest first,
   no overlaps, skip pure stopwords), condition op "=".
2. Numbers next to a comparison word ("more than 5", "after 1990") become > / <
   on the numeric column whose header is mentioned nearest.
3. Column ties broken by nearest header mention. Nothing found -> abstain.
Designed on dev only. Run: python3 bench/rules.py dev|test
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import TableIndex, cell_matches, is_number, stem  # noqa: E402

GT = {"more", "greater", "larger", "higher", "over", "above", "after", "later", "bigger", "exceeds", "exceeding", "beyond", "longer", "older"}
LT = {"less", "fewer", "smaller", "lower", "under", "below", "before", "earlier", "shorter", "younger"}
STOP = {"the", "a", "an", "of", "in", "on", "is", "was", "what", "which", "who", "how", "and", "for", "to", "at", "with", "by", "when", "did", "does", "do", "are", "were", "s", "?", ",", ".", "-", "'", "1", "0"}


def mentions(toks, tab):
    pos = {}
    for t, w in enumerate(toks):
        s = stem(w)
        for j in range(tab.ncols):
            if s in tab.header_stems[j] and w not in STOP:
                pos.setdefault(j, []).append(t)
    return pos


def nearest(cols, a, b, pos):
    def dist(j):  # header words usually come right before the value ("played <= 22")
        return min(((a - t) if t < a else (t - b + 1) * 1.5 + 0.5 for t in pos.get(j, [])), default=99)
    return min(cols, key=lambda j: (dist(j), j))


def predict(toks, tab):
    pos = mentions(toks, tab)
    header_tok = {t for ts in pos.values() for t in ts}
    conds, used = [], set()
    # comparisons first: they explain numbers that are not cells
    for t, w in enumerate(toks):
        if not is_number(w):
            continue
        window = set(toks[max(0, t - 3):t])
        op = ">" if window & GT else "<" if window & LT else None
        if op:
            cand = [j for j in range(tab.ncols) if tab.numeric[j]] or list(range(tab.ncols))
            conds.append([nearest(cand, t, t + 1, pos), op, w])
            used.add(t)
    for a, b, j in sorted(cell_matches(toks, tab), key=lambda m: (-(m[1] - m[0]), m[0])):
        span = set(range(a, b))
        if span & used or all(toks[t] in STOP for t in span) or span <= header_tok:
            continue
        cols = sorted({c for x, y, c in cell_matches(toks[a:b], tab) if x == 0 and y == b - a})
        conds.append([nearest(cols, a, b, pos), "=", " ".join(toks[a:b])])
        used |= span
    return conds or None


NEG = {"not", "isnt", "isn", "except", "excluding", "exclude", "without", "no", "!", "aren", "wasn", "other", "besides"}
HAS = {"contains", "containing", "has", "includes", "including", "like", "mentions"}
SUFFIX = {"k", "m", "b", "bn", "thousand", "million", "billion"}


def fb_op(toks, t, end):
    """Comparator for the number spanning toks[t:end], from the words around it."""
    before = " ".join(toks[max(0, t - 3):t])
    after = " ".join(toks[end:end + 2])
    if before.endswith("> =") or "at least" in before or after.startswith(("or more", "or higher", "or greater", "or above", "+")) or {"min", "minimum"} & set(toks[max(0, t - 2):t]):
        return ">="
    if before.endswith("< =") or "at most" in before or "up to" in before or after.startswith(("or less", "or fewer", "or lower", "or below", "or under")) or {"max", "maximum"} & set(toks[max(0, t - 2):t]):
        return "<="
    if before.endswith(">") or set(toks[max(0, t - 3):t]) & GT:
        return ">"
    if before.endswith("<") or set(toks[max(0, t - 3):t]) & LT:
        return "<"
    if before.endswith(("! =", "not", "isnt", "except")):
        return "!="
    return None


def predict_fb(toks, tab):
    from filterbench import parse_number
    pos = mentions(toks, tab)
    header_tok = {t for ts in pos.values() for t in ts}
    conds, used = [], set()
    for t, w in enumerate(toks):  # comparisons on numbers
        if t in used or not is_number(w):
            continue
        end = t + 2 if t + 1 < len(toks) and toks[t + 1] in SUFFIX else t + 1
        op = fb_op(toks, t, end)
        cand = [j for j in range(tab.ncols) if tab.numeric[j]] or list(range(tab.ncols))
        if op:
            conds.append([nearest(cand, t, end, pos), op, "%g" % parse_number(" ".join(toks[t:end]))])
            used |= set(range(t, end))
    for a, b, j in sorted(cell_matches(toks, tab), key=lambda m: (-(m[1] - m[0]), m[0])):
        span = set(range(a, b))
        if span & used or all(toks[t] in STOP for t in span) or span <= header_tok:
            continue
        cols = sorted({c for x, y, c in cell_matches(toks[a:b], tab) if x == 0 and y == b - a})
        neg = set(toks[max(0, a - 3):a]) & NEG or " ".join(toks[max(0, a - 2):a]).endswith("! =") or (a > 0 and toks[a - 1] == "-")
        conds.append([nearest(cols, a, b, pos), "!=" if neg else "=", " ".join(toks[a:b])])
        used |= span
    for t, w in enumerate(toks):  # leftover word inside some cells -> contains
        if t in used or t in header_tok or w in STOP or len(w) < 3 or is_number(w):
            continue
        cols = [j for j in range(tab.ncols) if any(w in k.split() for k in tab.cells if j in tab.cells[k])]
        if cols and (set(toks[max(0, t - 3):t]) & HAS or len(cols) == 1):
            conds.append([nearest(cols, t, t + 1, pos), "contains", w])
            used.add(t)
    return conds or None


def run_fb(bench_path):
    from filterbench import tokenize as tk
    d = json.load(open(bench_path))
    tabs = {k: TableIndex(v) for k, v in d["tables"].items()}
    out = Path(__file__).parent / "out" / "preds" / f"rules.{Path(bench_path).stem.replace('fb.', 'fb-')}.jsonl"
    with open(out, "w") as f:
        for i, e in enumerate(d["examples"]):
            t0 = time.perf_counter()
            conds = predict_fb(tk(e["q"]), tabs[e["table"]])
            f.write(json.dumps({"i": i, "conds": conds, "conf": 1.0, "ms": (time.perf_counter() - t0) * 1000}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "fb":
        run_fb(sys.argv[2])
        sys.exit()
    split = sys.argv[1]
    root = Path(__file__).parent / "out"
    d = json.load(open(root / f"{split}.json"))
    tabs = {k: TableIndex(v) for k, v in d["tables"].items()}
    (root / "preds").mkdir(exist_ok=True)
    with open(root / "preds" / f"rules.{split}.jsonl", "w") as f:
        for i, e in enumerate(d["examples"]):
            t0 = time.perf_counter()
            conds = predict(e["toks"], tabs[e["table"]])
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"i": i, "conds": conds, "conf": 1.0, "ms": ms}) + "\n")
