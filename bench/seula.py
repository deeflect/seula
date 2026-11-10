"""Seula: tiny schema- and data-aware filter extractor, 7 operators.

Per query token: hashed word + char-trigram embedding, plus links to every column
(header word, fuzzy/abbreviated header word, whole-cell match, word-inside-cell,
numeric). A small BiGRU tags value spans; each span gets an operator and a column
chosen by a pointer over the table's columns. Text values snap to the closest real
cell ("feb 20" -> "February 20"). Confidence = product of the decisions' probabilities.

Trained on WikiSQL train (human questions, = > <) + FilterBench train (Haiku-written
filter-box queries, all 7 ops). WikiSQL ">" supervises "> or >=" since it has no >=.

  uv run --with torch python3 bench/seula.py train
  uv run --with torch python3 bench/seula.py predict wikisql dev|test [--sample]
  uv run --with torch python3 bench/seula.py predict fb bench/out/fb.test.json
"""
import json
import os
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from features import MAX_HEADER_WORDS, MAX_TOKENS, NUM_LINK, TableIndex, bag, cell_matches, hash_ids, is_number, link_matrix, norm_word  # noqa: E402
from prepare import tokenize  # noqa: E402
from align import align, edit1  # noqa: E402
from filterbench import col_numeric, norm_text, parse_number, run  # noqa: E402

ROOT = Path(__file__).parent / "out"
BUCKETS, DIM, HID = 4096, 32, 48
FB_REPEAT = 3
MONTHS = set("jan january feb february mar march apr april may jun june jul july aug august sep sept september oct october nov november dec december".split())
MAGNITUDE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
OPS = ["=", "!=", "contains", ">", ">=", "<", "<="]
WIKI_ALLOWED = {"=": [0], ">": [3, 4], "<": [5, 6]}  # WikiSQL op -> acceptable v3 ops (the second only with an INCLUSIVE cue)
INCLUSIVE = ("at least", "or more", "no less", "not less", "no fewer", "minimum", "or greater", "or higher", "or above", "or later",
             "at most", "or less", "no more", "not more", "or fewer", "maximum", "or lower", "or below", "or earlier", "up to")
STRICT = {"over", "above", "under", "below", "than", "exceeding"}  # right before a number: > or <, though the LLM-written training queries label a third of them >= or <=
AND_FORMS = ["n", "&", "+", ","]
LIST = {"or", "nor", "/", "|", ","}
NUM_LINK3 = NUM_LINK + 2
TAGS = ["O", "B", "I"]
NUM_DENSE = 3 + NUM_LINK3


def encode(toks: list[str], tab: TableIndex) -> dict:
    toks = toks[:MAX_TOKENS]
    L = link_matrix(toks, tab, cell_matches(toks, tab, fuzzy=True))
    words = [set() for _ in range(tab.ncols)]
    for k, cols in tab.cells.items():
        for j in cols:
            words[j].update(map(norm_word, k.split()))
    hwords = [[h for h in hs if norm_word(h)][:MAX_HEADER_WORDS] for hs in tab.header]
    near = lambda w, h: len(w) >= 3 and ((len(h) > 3 and edit1(w, h)) or (len(w) < len(h) and h.startswith(w)))  # noqa: E731
    hw = []  # hw[t][j][k]: token t names word k of header j; span_heads turns this into "how much of the header was typed"
    for t, w in enumerate(toks):
        nw = norm_word(w)
        for j in range(tab.ncols):
            L[t][j] += [1.0 if any(near(w, h) for h in tab.header[j]) else 0.0, 1.0 if len(w) >= 3 and nw in words[j] else 0.0]
        hw.append([[1.0 if k < len(hs) and nw and (nw == norm_word(hs[k]) or near(w, hs[k])) else 0.0 for k in range(MAX_HEADER_WORDS)] for hs in hwords])
    dense = []
    for t, w in enumerate(toks):
        row = [1.0 if is_number(w) else 0.0, min(1.0, sum(L[t][j][2] for j in range(tab.ncols)) / 3), t / MAX_TOKENS]
        row += [max(L[t][j][k] for j in range(tab.ncols)) for k in range(NUM_LINK3)]
        dense.append(row)
    return {
        "ids": [hash_ids(w, BUCKETS) for w in toks],
        "dense": dense,
        "link": L,
        "hw": hw,
        "hn": [max(1, len(hs)) for hs in hwords],
        "hdr": [[i for w in h for i in hash_ids(w, BUCKETS)][:48] or [0] for h in tab.header],
        "num": [1.0 if n else 0.0 for n in tab.numeric],
    }


def collate(items: list[dict]):
    B = len(items)
    T = max(len(x["ids"]) for x in items)
    C = max(len(x["hdr"]) for x in items)
    K = max(len(i) for x in items for i in x["ids"])
    KH = max(len(h) for x in items for h in x["hdr"])
    ids = torch.zeros(B, T, K, dtype=torch.long)
    idm = torch.zeros(B, T, K)
    dense = torch.zeros(B, T, NUM_DENSE)
    link = torch.zeros(B, T, C, NUM_LINK3)
    hdr = torch.zeros(B, C, KH, dtype=torch.long)
    hdm = torch.zeros(B, C, KH)
    num = torch.zeros(B, C)
    hw = torch.zeros(B, T, C, MAX_HEADER_WORDS)
    hn = torch.ones(B, C)
    tmask = torch.zeros(B, T, dtype=torch.bool)
    cmask = torch.zeros(B, C, dtype=torch.bool)
    for b, x in enumerate(items):
        for t, i in enumerate(x["ids"]):
            ids[b, t, :len(i)] = torch.tensor(i)
            idm[b, t, :len(i)] = 1
        n = len(x["ids"])
        tmask[b, :n] = True
        dense[b, :n] = torch.tensor(x["dense"])
        c = len(x["hdr"])
        cmask[b, :c] = True
        link[b, :n, :c] = torch.tensor(x["link"])
        hw[b, :n, :c] = torch.tensor(x["hw"])
        hn[b, :c] = torch.tensor(x["hn"])
        for j, h in enumerate(x["hdr"]):
            hdr[b, j, :len(h)] = torch.tensor(h)
            hdm[b, j, :len(h)] = 1
        num[b, :c] = torch.tensor(x["num"])
    return dict(ids=ids, idm=idm, dense=dense, link=link, hdr=hdr, hdm=hdm, num=num, hw=hw, hn=hn, tmask=tmask, cmask=cmask)


class SeulaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(BUCKETS, DIM)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.inp = nn.Linear(DIM + NUM_DENSE, HID)
        self.gru = nn.GRU(HID, HID, batch_first=True, bidirectional=True)
        self.tag = nn.Linear(2 * HID, len(TAGS))
        self.span = nn.Linear(4 * HID, 2 * HID)
        self.op = nn.Linear(2 * HID, len(OPS))
        self.col = nn.Linear(DIM + 1, 2 * HID)
        self.linkmlp = nn.Sequential(nn.Linear(NUM_LINK3 + 6, 16), nn.ReLU(), nn.Linear(16, 1))

    def bag(self, ids, mask):
        return (self.emb(ids) * mask.unsqueeze(-1)).sum(-2) / mask.sum(-1, keepdim=True).clamp(min=1)

    def encode(self, x):
        e = self.bag(x["ids"], x["idm"])
        h = torch.relu(self.inp(torch.cat([e, x["dense"]], -1)))
        lengths = x["tmask"].sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        cols = self.col(torch.cat([self.bag(x["hdr"], x["hdm"]), x["num"].unsqueeze(-1)], -1))  # B,C,2H
        return out, cols

    def span_heads(self, out, cols, x, b, a, z):
        """Op logits and column logits for span [a, z) of batch item b."""
        s = torch.tanh(self.span(torch.cat([out[b, a], out[b, z - 1]], -1)))
        lk = x["link"][b, a:z].mean(0)  # C,K
        width = torch.full((lk.shape[0], 1), (z - a) / 8, device=lk.device)
        hdr = x["link"][b, :, :, [0, 1, 6]].amax(-1)  # T,C: token names column j (exact, stem, fuzzy)
        before = hdr[max(0, a - 3):a].amax(0) if a > 0 else torch.zeros_like(width[:, 0])
        after = hdr[z:z + 2].amax(0) if z < hdr.shape[0] else torch.zeros_like(width[:, 0])
        # Share of each header typed next to the value: "home team: x" is all of "Home team", half of "Away team".
        hw, zero = x["hw"][b], torch.zeros_like(width[:, 0])
        nb = hw[max(0, a - 4):a].amax(0).sum(-1) if a > 0 else zero
        na = hw[z:z + 3].amax(0).sum(-1) if z < hw.shape[0] else zero
        cover = torch.stack([nb / x["hn"][b], na / x["hn"][b], (nb + na).clamp(max=3) / 3], -1)
        feats = torch.cat([lk, width, before.unsqueeze(-1), after.unsqueeze(-1), cover], -1)
        col_logits = cols[b] @ s + self.linkmlp(feats).squeeze(-1)
        col_logits = col_logits.masked_fill(~x["cmask"][b], -1e9)
        return self.op(s), col_logits


def to(x, dev):
    return {k: v.to(dev) for k, v in x.items()}


def load(split):
    d = json.load(open(ROOT / f"{split}.json"))
    tabs = {k: TableIndex(v) for k, v in d["tables"].items()}
    return d, tabs


def train():
    random.seed(0)
    torch.manual_seed(0)
    import os
    dev = torch.device(os.environ.get("DEVICE", "cpu"))
    limit = int(os.environ.get("LIMIT", "0")) or None
    d, tabs = load("train")
    t0 = time.time()
    data = []
    for e in d["examples"][:limit]:
        if not e["spans"] or max(s[1] for s in e["spans"]) > MAX_TOKENS:
            continue
        x = encode(e["toks"], tabs[e["table"]])
        loose = any(cue in e["q"].lower() for cue in INCLUSIVE)
        x["gold"] = [(a, z, WIKI_ALLOWED[o][:None if loose else 1], c) for (a, z), (c, o, _) in zip(e["spans"], e["conds"])]
        data.append(x)
    nwiki = len(data)
    fb = {"examples": [], "tables": {}}
    for name in os.environ.get("FB_TRAIN", "fb.train.json").split(","):  # e.g. fb.train.json,fb.train2.json
        part = json.load(open(ROOT / name))
        fb["examples"] += part["examples"]
        fb["tables"].update(part["tables"])
    ftabs = {k: TableIndex(v) for k, v in fb["tables"].items()}
    skipped = 0
    for e in fb["examples"][:int(os.environ.get("FB_N", "0")) or limit] if os.environ.get("FB_N") != "0" else []:
        toks = tokenize(e["q"])
        spans = align(toks, e["conds"])
        if not spans or max(z for _, z in spans) > MAX_TOKENS:
            skipped += 1
            continue
        loose = any(cue in e["q"].lower() for cue in INCLUSIVE)
        gold = [(a, z, [OPS.index(o) - (o in (">=", "<=") and not loose and a > 0 and toks[a - 1] in STRICT)], c)
                for (a, z), (c, o, _) in zip(spans, e["conds"])]
        inside = {t for a, z in spans for t in range(a, z)}
        for r in range(FB_REPEAT):  # repeats 2+ swap some "and" for what people type instead: n & + ,
            tk = [random.choice(AND_FORMS) if r and w == "and" and t not in inside and random.random() < 0.4 else w for t, w in enumerate(toks)]
            if r == 0 or tk != toks:
                x = dict(encode(tk, ftabs[e["table"]]), gold=gold)
            data.append(x)
    print(f"encoded {nwiki} wikisql + {(len(data) - nwiki) // FB_REPEAT} filterbench (x{FB_REPEAT}), {skipped} unaligned, {time.time() - t0:.0f}s", flush=True)
    model = SeulaNet().to(dev)
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    epochs, bs = int(os.environ.get("EPOCHS", "8")), 64
    steps = epochs * math.ceil(len(data) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for ep in range(epochs):
        random.shuffle(data)
        tot = 0.0
        for k in range(0, len(data), bs):
            batch = data[k:k + bs]
            x = to(collate(batch), dev)
            out, cols = model.encode(x)
            tags = torch.zeros(x["tmask"].shape, dtype=torch.long, device=dev)
            op_l, op_y, col_l, col_y = [], [], [], []
            for b, item in enumerate(batch):
                for a, z, o, c in item["gold"]:
                    tags[b, a] = 1
                    tags[b, a + 1:z] = 2
                    ol, cl = model.span_heads(out, cols, x, b, a, z)
                    op_l.append(ol)
                    op_y.append(o)
                    col_l.append(cl)
                    col_y.append(c)
            tag_loss = F.cross_entropy(model.tag(out)[x["tmask"]], tags[x["tmask"]])
            logp = F.log_softmax(torch.stack(op_l), -1)
            op_loss = -torch.stack([torch.logsumexp(logp[k, ys], 0) for k, ys in enumerate(op_y)]).mean()
            col_loss = F.cross_entropy(torch.stack(col_l), torch.tensor(col_y, device=dev))
            loss = tag_loss + op_loss + col_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1} loss {tot / math.ceil(len(data) / bs):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), ROOT / os.environ.get("OUT", "seula.pt"))


def words(toks):
    """Tokens with a letter or digit (punctuation like , . ( ) – dropped)."""
    return [w for w in toks if any(ch.isalnum() for ch in w)]


def snap(value_toks, tab, col):
    """Closest real cell in col for a typed text value: exact, else a unique cell matching
    word by word (typo/prefix ok), else unique with spaces ignored ("1 st" = "1st"),
    else the unique cell containing it ("may 23 1808" -> "seated may 23, 1808")."""
    v = " ".join(value_toks)
    cells = [k for k, cols in tab.cells.items() if col in cols]
    if v in cells:
        return v
    same = tab.bags.get(bag(value_toks), {}).get(col, ())  # other word order, punctuation, accents, plural, "sept"
    if len(same) == 1:
        return next(iter(same))
    def ok(a, b):
        return a == b or (len(b) > 3 and edit1(a, b)) or (len(a) >= 3 and b.startswith(a))
    vt = words(value_toks)
    kw = {k: words(k.split()) for k in cells}
    best = [k for k in cells if len(kw[k]) == len(vt) and all(ok(a, b) for a, b in zip(vt, kw[k]))]
    if best:
        return best[0] if len(best) == 1 else v
    squash = lambda toks: "".join(ch for w in toks for ch in w if ch.isalnum())  # noqa: E731
    sq = squash(vt)
    for hit in (lambda s: s == sq, lambda s: len(sq) >= 6 and sq in s):
        found = [k for k in cells if hit(squash(kw[k]))]
        if found:
            return found[0] if len(found) == 1 else v
    return v


@torch.no_grad()
def predict_one(model, toks, tab):
    toks = toks[:MAX_TOKENS]
    x = to(collate([encode(toks, tab)]), "cpu")
    out, cols = model.encode(x)
    probs = F.softmax(model.tag(out)[0], -1)
    T = len(toks)
    tags = probs[:T].argmax(-1).tolist()
    conf = math.prod(probs[t].max().item() for t in range(T))
    spans, t = [], 0
    while t < T:
        if tags[t] != 0:
            a = t
            t += 1
            while t < T and tags[t] == 2:
                t += 1
            spans.append((a, t))
        else:
            t += 1
    def head(a, z):
        ol, cl = model.span_heads(out, cols, x, 0, a, z)
        return F.softmax(ol, -1), F.softmax(cl, -1)

    heads = [head(a, z) for a, z in spans]
    # Two values with nothing between them, same column and text operator, found together in one cell, are one value the tagger split ("julie plec").
    k = 1
    while k < len(spans):
        (po, pc), (qo, qc) = heads[k - 1], heads[k]
        col, both = int(pc.argmax()), f' {" ".join(toks[spans[k - 1][0]:spans[k][1]])} '
        if spans[k - 1][1] == spans[k][0] and col == int(qc.argmax()) and int(po.argmax()) == int(qo.argmax()) \
                and OPS[int(po.argmax())] in ("=", "contains") and any(col in cols and both in f" {c} " for c, cols in tab.cells.items()):
            spans[k - 1:k + 1] = [(spans[k - 1][0], spans[k][1])]
            heads[k - 1:k + 1] = [head(*spans[k - 1])]
        else:
            k += 1
    # Columns decoded jointly: each condition gets its own column, highest probability first.
    # Exception: a lower and an upper bound may share a column ("> 10 and < 20").
    # ponytail: greedy, not optimal assignment; fine for the 1-4 conditions a filter box sees.
    side = lambda o: 1 if o in (">", ">=") else -1 if o in ("<", "<=") else 0  # noqa: E731
    # A number on a numeric column is never "contains"; it only looks like it when the digits sit inside other cells ("10" in "1000").
    for k, (po, pc) in enumerate(heads):
        if tab.numeric[int(pc.argmax())] and parse_number("".join(toks[spans[k][0]:spans[k][1]])) is not None:
            po = po.clone()
            po[OPS.index("contains")] = 0
            heads[k] = (po / po.sum(), pc)
    ops = [OPS[int(po.argmax())] for po, _ in heads]
    # Lists: "a, b or c" (any of them) and "not a or b" (none of them) put several values on one column.
    # Members joined by an "or" must share it, so they vote together; a comma alone only allows sharing,
    # and a comma after the "or" starts something new ("a or b, c").
    grp, forced, pcs = list(range(len(spans))), set(), [pc for _, pc in heads]
    for k in range(1, len(spans)):
        gap = toks[spans[k - 1][1]:spans[k][0]]
        if gap and all(w in LIST for w in gap) and (ops[k] == ops[k - 1] == "!=" or {ops[k], ops[k - 1]} <= {"=", "contains"}):
            if any(w != "," for w in gap):
                grp[k] = grp[k - 1]
                forced.add(grp[k])
            elif grp[k - 1] not in forced:
                grp[k] = grp[k - 1]
    for g in forced:
        mem = [k for k in range(len(spans)) if grp[k] == g]
        joint = torch.stack([pcs[k].clamp(min=1e-9).log() for k in mem]).sum(0).softmax(-1)
        for k in mem:
            pcs[k] = joint
    heads = [(po, pcs[k]) for k, (po, _) in enumerate(heads)]
    pairs = sorted(((sig(pc[j].item()), k, j) for k, (_, pc) in enumerate(heads) for j in range(pc.shape[0])), reverse=True)
    chosen, used = {}, {}
    for p, k, j in pairs:
        if k in chosen or p <= 0:
            continue
        if j in used and not all(grp[k] == grp[m] or side(ops[k]) * side(ops[m]) == -1 for m in used[j]):
            continue
        chosen[k] = (j, p)
        used.setdefault(j, []).append(k)
    cols = [chosen.get(k, (int(heads[k][1].argmax()), 0))[0] for k in range(len(spans))]
    conds = build(toks, tab, spans, cols, ops)
    conf *= bool(spans) and math.prod(heads[k][0].max().item() * heads[k][1][cols[k]].item() for k in range(len(spans)))
    alts = [(sig(conf * pc[j].item() / max(pc[cols[k]].item(), 1e-9)), k, j, ops[k]) for k, (_, pc) in enumerate(heads)
            for j in pc.topk(min(3, pc.shape[0])).indices.tolist() if j != cols[k]]
    alts += [(sig(conf * po[o].item() / po.max().item()), k, cols[k], OPS[o]) for k, (po, _) in enumerate(heads)
             for o in po.topk(2).indices.tolist() if OPS[o] != ops[k]]
    return conds, conf, [(c, k, j, o, spans, cols, ops) for c, k, j, o in sorted(alts, reverse=True)]


def sig(p):
    """6 significant digits: two near-certain probabilities tie the same way here and in the float64 TS runtime."""
    return float("%.6g" % p)


def build(toks, tab, spans, cols, ops):
    T = len(toks)
    conds, starts, covered = [], {a for a, _ in spans}, {t for a, z in spans for t in range(a, z)}
    for k, (a, z) in enumerate(spans):
        j = cols[k]
        if z < T and z not in starts and toks[z] in MAGNITUDE and parse_number("".join(toks[a:z])) is not None \
                and tab.absmax[j] >= MAGNITUDE[toks[z]]:
            z += 1  # "10 million": the tagger often stops at "10" (rare in training data); skip if cells are already in millions
        if a > 0 and toks[a - 1] in MONTHS and a - 1 not in covered and not tab.numeric[j] \
                and parse_number("".join(toks[a:z])) is not None:
            a -= 1  # "feb 25": the tagger often keeps only the day; snap then finds "february 25"
        conds.append([j, ops[k], toks[a:z]])
    return conds


def execution_guided(toks, tab, rows, conds, conf, alts):
    """Benchmark gold always returns rows, so a parse that returns none is wrong: try the next-best
    single change (one condition's column or op), most probable first. Nothing works -> conf 0."""
    num = [col_numeric(rows, j) for j in range(tab.ncols)]
    res = finalize(conds, tab)
    if not res or run(rows, [tuple(c) for c in res], num):
        return res, conf
    for c, k, j, o, spans, cols, ops in alts:
        cols2, ops2 = cols[:k] + [j] + cols[k + 1:], ops[:k] + [o] + ops[k + 1:]
        r2 = finalize(build(toks, tab, spans, cols2, ops2), tab)
        if r2 and run(rows, [tuple(x) for x in r2], num):
            return r2, c
    return res, 0.0


def fmt_num(n):
    """Plain digits for whole numbers ("10000000", never "1e+07"), else %g."""
    return str(int(n)) if n.is_integer() and abs(n) < 1e15 else "%g" % n


def snap_words(value_toks, tab, col):
    """contains: a typed word found in no cell of col becomes the one cell word it normalizes to
    ("managers" -> "manager", "muller" -> "müller")."""
    have = {w for k, cols in tab.cells.items() if col in cols for w in k.split()}
    out = []
    for w in value_toks:
        alt = {x for x in have if norm_word(x) == norm_word(w)} if w not in have and norm_word(w) else ()
        out.append(next(iter(alt)) if len(alt) == 1 else w)
    return " ".join(out)


def finalize(conds, tab):
    """Token conditions -> [col, op, value]: numbers normalized, text snapped to real cells (= / !=) or cell words (contains)."""
    res = []
    for c, o, vt in conds:
        text = " ".join(vt)
        n = parse_number(text.replace(" ", ""))
        if n is None and o in (">", ">=", "<", "<=") and len(vt) > 1:
            n = parse_number(vt[0])  # "10000 gb", "18 yards": a unit the tagger kept after the number
        if o in (">", ">=", "<", "<=") or (n is not None and tab.numeric[c]):
            res.append([c, o, fmt_num(n) if n is not None else text])
        else:
            res.append([c, o, snap(vt, tab, c) if o in ("=", "!=") else snap_words(vt, tab, c)])
    return res or None


def load_model():
    torch.set_num_threads(1)
    model = SeulaNet()
    model.load_state_dict(torch.load(ROOT / os.environ.get("OUT", "seula.pt"), map_location="cpu"))
    return model.eval()


def predict_fb(bench_path):
    model = load_model()
    d = json.load(open(bench_path))
    tabs = {k: TableIndex(v) for k, v in d["tables"].items()}
    name = f"seula.{Path(bench_path).stem.replace('fb.', 'fb-')}.jsonl"
    with open(ROOT / "preds" / name, "w") as f:
        for i, e in enumerate(d["examples"]):
            t0 = time.perf_counter()
            tab = tabs[e["table"]]
            toks = tokenize(e["q"])[:MAX_TOKENS]
            conds, conf, alts = predict_one(model, toks, tab)
            if os.environ.get("EG", "1") == "1":
                res, conf = execution_guided(toks, tab, d["tables"][e["table"]]["rows"], conds, conf, alts)
            else:
                res = finalize(conds, tab)
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"i": i, "conds": res, "conf": conf, "ms": ms}) + "\n")
    print("wrote", name)


def predict(split, sample=False):
    model = load_model()
    d, tabs = load(split)
    idx = json.load(open(ROOT / "test_sample.json")) if sample else range(len(d["examples"]))
    name = f"seula.{split}{'-sample' if sample else ''}.jsonl"
    back = {"=": "=", "!=": "=", "contains": "=", ">": ">", ">=": ">", "<": "<", "<=": "<"}
    with open(ROOT / "preds" / name, "w") as f:
        for i in idx:
            e = d["examples"][i]
            t0 = time.perf_counter()
            conds, conf, _ = predict_one(model, e["toks"], tabs[e["table"]])
            ms = (time.perf_counter() - t0) * 1000
            conds = [[c, back[o], " ".join(vt)] for c, o, vt in conds] or None
            f.write(json.dumps({"i": i, "conds": conds, "conf": conf, "ms": ms}) + "\n")
    print("wrote", name)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[2] == "fb":
        predict_fb(sys.argv[3])
    else:
        predict(sys.argv[3], "--sample" in sys.argv)
