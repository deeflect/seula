"""FilterBench: filter-box queries over real tables, with the full operator set.

WikiSQL only has = > < and its values are copied from cells. Real filter boxes see
"over 1k", "not acme", "name has fc", ">= 3 goals". This builds that benchmark:
sample gold conditions from a real WikiSQL table, then ask an LLM to write what a
person would type to get exactly those rows. Gold is known by construction.

  python3 bench/filterbench.py gen  test  600 claude-sonnet-5-5 bench/out/fb.test.json
  python3 bench/filterbench.py gen  train 6000 claude-haiku-5-5 bench/out/fb.train.json
  python3 bench/filterbench.py score bench/out/fb.test.json preds.jsonl [...]

Ops: = != contains > >= < <=. Scored by exact condition set AND by returned rows.
"""
import json
import os
import random
import re
import statistics
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare import tokenize  # noqa: E402

ROOT = Path(__file__).parent / "out"
OPS = ["=", "!=", "contains", ">", ">=", "<", "<="]
STYLES = [
    "terse keyword style, like typing into a search box (e.g. 'red cars under 5k')",
    "a natural sentence or request (e.g. 'show me the red cars that cost less than 5000')",
    "casual and messy: lowercase, abbreviations, shorthand like 1k/2.5m, maybe one small typo",
    "operator-ish power-user style mixing symbols and words (e.g. 'color: red, price<5000')",
]
STOP = {"the", "and", "for", "with", "from", "of", "in", "on", "at", "to"}


def parse_number(s: str):
    """'1.5k' '$2,000' '30%' '1e3' -> float, else None. Shared by every method."""
    t = str(s).strip().lower().replace(",", "").replace("$", "").replace("€", "").replace("£", "").rstrip("%").strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(k|m|b|bn|thousand|million|billion)?", t)
    if not m:
        return None
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}.get(m.group(2) or "", 1)
    return float(m.group(1)) * mult


def norm_text(s) -> str:
    return " ".join(tokenize(str(s)))


def col_numeric(rows, j) -> bool:
    vals = [parse_number(r[j]) for r in rows]
    return sum(v is not None for v in vals) >= 0.8 * len(rows)


def matches(row, cond, numeric) -> bool:
    j, op, v = cond
    cell = row[j]
    if op in (">", ">=", "<", "<=") or (op in ("=", "!=") and numeric[j] and parse_number(v) is not None):
        a, b = parse_number(cell), parse_number(v)
        if a is None or b is None:
            return False
        return {"=": a == b, "!=": a != b, ">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]
    c, x = norm_text(cell), norm_text(v)
    if op == "=":
        return c == x
    if op == "!=":
        return c != x
    return bool(x) and f" {x} " in f" {c} "  # contains: whole-word substring


def run(rows, conds, numeric) -> frozenset:
    """Rows matching every condition. Several "=" / "contains" on one column mean any of them ("overdue or pending")."""
    eq, rest = {}, []
    for c in conds:
        (eq.setdefault(c[0], []) if c[1] in ("=", "contains") else rest).append(c)
    return frozenset(i for i, r in enumerate(rows) if all(matches(r, c, numeric) for c in rest)
                     and all(any(matches(r, c, numeric) for c in g) for g in eq.values()))


def sample_conds(rng, table, ranges=False):
    rows, header = table["rows"], table["header"]
    numeric = [col_numeric(rows, j) for j in range(len(header))]
    k = rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]
    cols = rng.sample(range(len(header)), min(k, len(header)))
    conds = []
    for j in cols:
        cells = [r[j] for r in rows if str(r[j]).strip()]
        if not cells:
            return None
        if numeric[j]:
            op = rng.choice([">", ">="]) if ranges and rng.random() < 0.7 else rng.choice([">", ">=", "<", "<=", "="])
            nums = sorted(n for n in (parse_number(c) for c in cells) if n is not None)
            if op == "=":
                v = rng.choice(nums)
            else:
                v = statistics.quantiles(nums, n=4)[rng.randrange(3)] if len(nums) >= 4 else rng.choice(nums)
                if abs(v) >= 100:  # people type round thresholds
                    mag = 10 ** (len(str(int(abs(v)))) - 2)
                    v = round(v / mag) * mag
            conds.append([j, op, ("%g" % v)])
            if ranges and op in (">", ">=") and len(nums) >= 4:  # "between 3 and 6"
                hi = statistics.quantiles(nums, n=4)[2]
                if hi > v:
                    conds.append([j, rng.choice(["<", "<="]), "%g" % hi])
        else:
            op = rng.choices(["=", "!=", "contains"], [0.6, 0.2, 0.2])[0]
            cell = rng.choice(cells)
            if op == "contains":
                words = [w for w in tokenize(cell) if len(w) >= 3 and w not in STOP and not w.isdigit()]
                if len(tokenize(cell)) < 2 or not words:
                    op, v = "=", cell
                else:
                    v = rng.choice(words)
            else:
                v = cell
            conds.append([j, op, str(v)])
    hit = run(rows, conds, numeric)
    if not hit or len(hit) == len(rows):  # must actually filter something
        return None
    return conds


PROMPT = """You write realistic test queries for a "filter this table" search box.
For each item below, write the text a person would type into the box to get EXACTLY the rows matching ALL its conditions.
Rules:
- Express every condition and nothing more. No sorting, no "how many", no select-a-column questions.
- Use the requested style. You may paraphrase column names, use synonyms, or drop the column name when the value makes it obvious.
- Numbers may be written naturally (1k, 1,500, "at least 3"). Keep the exact threshold meaning: ">=" means at least/or more, ">" means more than/over, etc.
- "contains" means the column includes that word; "!=" means excluding that exact value.
- Write it the way a person would, not by copying the conditions: say "not", "except", "without", "with", "over", "at least" instead of the symbols, unless the style is power-user.
- Follow each item's style; items differ in style on purpose.
Return ONLY a JSON array of strings, one per item, same order.

{items}"""


def describe(table, conds, style):
    h = table["header"]
    ex = "; ".join(" | ".join(str(c) for c in r) for r in table["rows"][:2])
    cs = " AND ".join(f'[{h[j]}] {op} "{v}"' for j, op, v in conds)
    return f"Table columns: {h}\nExample rows: {ex}\nConditions: {cs}\nStyle: {style}"


def call_llm(model, items):
    text = PROMPT.format(items="\n\n".join(f"ITEM {i + 1}\n{d}" for i, d in enumerate(items)))
    if os.environ.get("LLM_URL"):  # any OpenAI-compatible server, e.g. llama-server with an open model
        req = urllib.request.Request(os.environ["LLM_URL"] + "/v1/chat/completions", headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("LLM_KEY", "none")},
                                     data=json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode())
        res = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    else:
        out = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json", text],
                             capture_output=True, text=True, timeout=300)
        res = json.loads(out.stdout)["result"]
    arr = json.loads(res[res.index("["): res.rindex("]") + 1])
    return arr if len(arr) == len(items) else None


def gen(split, n, model, out_path, batch=15, seed=None, ranges=False):
    rng = random.Random(seed if seed is not None else {"test": 1, "dev": 2, "train": 3}[split])
    tables = json.load(open(ROOT / f"{split}.json"))["tables"]
    ids = [k for k, t in tables.items() if len(t["rows"]) >= 5]
    jobs = []
    while len(jobs) < n:
        tid = rng.choice(ids)
        conds = sample_conds(rng, tables[tid], ranges)
        if conds:
            jobs.append({"table": tid, "conds": conds, "style": rng.randrange(len(STYLES))})
    batches = [jobs[i:i + batch] for i in range(0, n, batch)]

    def work(b):
        for _ in range(2):
            try:
                qs = call_llm(model, [describe(tables[j["table"]], j["conds"], STYLES[j["style"]]) for j in b])
                if qs:
                    return [dict(j, q=q) for j, q in zip(b, qs) if isinstance(q, str) and q.strip()]
            except Exception as e:  # noqa: BLE001 - retry once, then drop the batch
                print("batch failed:", e, file=sys.stderr)
        return []

    with ThreadPoolExecutor(6) as ex:
        examples = [x for r in ex.map(work, batches) for x in r]
    used = {e["table"] for e in examples}
    json.dump({"model": model, "split": split, "examples": examples,
               "tables": {k: tables[k] for k in used}}, open(out_path, "w"))
    print(f"wrote {len(examples)}/{n} to {out_path}")


def score(bench_path, pred_paths):
    d = json.load(open(bench_path))
    ex = d["examples"]
    print(f"{'method':30} {'n':>5} {'answer%':>8} {'rows ok%':>9} {'exact%':>7} {'wrong%':>7} {'prec%':>6} {'cov@95':>7} {'ms p50':>8}")
    for p in pred_paths:
        preds = {x["i"]: x for x in map(json.loads, open(p))}
        stats = []
        for i, pr in preds.items():
            e, t = ex[i], d["tables"][ex[i]["table"]]
            numeric = [col_numeric(t["rows"], j) for j in range(len(t["header"]))]
            gold = run(t["rows"], e["conds"], numeric)
            conds = pr.get("conds")
            if conds is None:
                stats.append((pr.get("conf", 1), False, False, False, pr.get("ms")))
                continue
            try:
                got = run(t["rows"], [[int(c), o, v] for c, o, v in conds], numeric)
                # key=repr: a number and a text value on one column must not make the sort (and with it the row check) fail
                key = lambda cs: sorted(((int(c), o, parse_number(v) if parse_number(v) is not None else norm_text(v)) for c, o, v in cs), key=repr)  # noqa: E731
                exact = key(conds) == key(e["conds"])
            except (ValueError, KeyError, TypeError, IndexError):
                got, exact = None, False
            stats.append((pr.get("conf", 1), True, got == gold, exact, pr.get("ms")))
        n = len(stats)
        ans = [s for s in stats if s[1]]
        ok = sum(s[2] for s in stats)
        ranked, hits, cov = sorted(ans, key=lambda s: -s[0]), 0, 0
        for k, s in enumerate(ranked, 1):
            hits += s[2]
            if hits / k >= 0.95:
                cov = k
        ms = [s[4] for s in stats if s[4] is not None]
        print(f"{Path(p).stem:30} {n:5} {len(ans)/n*100:8.1f} {ok/n*100:9.1f} {sum(s[3] for s in stats)/n*100:7.1f} "
              f"{(len(ans)-ok)/n*100:7.1f} {ok/max(1,len(ans))*100:6.1f} {cov/n*100:7.1f} {statistics.median(ms or [0]):8.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        a = sys.argv  # --ranges: also sample two bounds on one column; --seed N for a fresh draw
        gen(a[2], int(a[3]), a[4], a[5], seed=int(a[a.index("--seed") + 1]) if "--seed" in a else None, ranges="--ranges" in a)
    else:
        score(sys.argv[2], sys.argv[3:])
