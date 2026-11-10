"""Sieve v2: tiny schema- and data-aware filter extractor.

Per question token: hashed word + char-trigram embedding (sees the words, unlike v1),
plus links to every column (header word, cell match, numeric). A small BiGRU tags value
spans; each span gets an operator and a column chosen by a pointer over the table's
columns. Confidence = product of the decisions' probabilities, so it can abstain.

Train: uv run --with torch python3 bench/sieve2.py train
Predict: uv run --with torch python3 bench/sieve2.py predict dev|test [--sample]
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from features import MAX_TOKENS, NUM_LINK, TableIndex, cell_matches, hash_ids, is_number, link_matrix  # noqa: E402
from prepare import tokenize  # noqa: E402

ROOT = Path(__file__).parent / "out"
BUCKETS, DIM, HID = 4096, 32, 48
OPS = ["=", ">", "<"]
TAGS = ["O", "B", "I"]
NUM_DENSE = 3 + NUM_LINK


def encode(toks: list[str], tab: TableIndex) -> dict:
    toks = toks[:MAX_TOKENS]
    L = link_matrix(toks, tab, cell_matches(toks, tab))
    dense = []
    for t, w in enumerate(toks):
        row = [1.0 if is_number(w) else 0.0, min(1.0, sum(L[t][j][2] for j in range(tab.ncols)) / 3), t / MAX_TOKENS]
        row += [max(L[t][j][k] for j in range(tab.ncols)) for k in range(NUM_LINK)]
        dense.append(row)
    return {
        "ids": [hash_ids(w, BUCKETS) for w in toks],
        "dense": dense,
        "link": L,
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
    link = torch.zeros(B, T, C, NUM_LINK)
    hdr = torch.zeros(B, C, KH, dtype=torch.long)
    hdm = torch.zeros(B, C, KH)
    num = torch.zeros(B, C)
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
        for j, h in enumerate(x["hdr"]):
            hdr[b, j, :len(h)] = torch.tensor(h)
            hdm[b, j, :len(h)] = 1
        num[b, :c] = torch.tensor(x["num"])
    return dict(ids=ids, idm=idm, dense=dense, link=link, hdr=hdr, hdm=hdm, num=num, tmask=tmask, cmask=cmask)


class Sieve2(nn.Module):
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
        self.linkmlp = nn.Sequential(nn.Linear(NUM_LINK + 1, 16), nn.ReLU(), nn.Linear(16, 1))

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
        col_logits = cols[b] @ s + self.linkmlp(torch.cat([lk, width], -1)).squeeze(-1)
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
        x["gold"] = [(a, z, OPS.index(o), c) for (a, z), (c, o, _) in zip(e["spans"], e["conds"])]
        data.append(x)
    print(f"encoded {len(data)} in {time.time() - t0:.0f}s")
    model = Sieve2().to(dev)
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
            op_loss = F.cross_entropy(torch.stack(op_l), torch.tensor(op_y, device=dev))
            col_loss = F.cross_entropy(torch.stack(col_l), torch.tensor(col_y, device=dev))
            loss = tag_loss + op_loss + col_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1} loss {tot / math.ceil(len(data) / bs):.4f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), ROOT / "sieve2.pt")


@torch.no_grad()
def predict_one(model, toks, tab):
    x = to(collate([encode(toks, tab)]), "cpu")
    out, cols = model.encode(x)
    probs = F.softmax(model.tag(out)[0], -1)
    T = len(toks[:MAX_TOKENS])
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
    conds = []
    for a, z in spans:
        ol, cl = model.span_heads(out, cols, x, 0, a, z)
        po, pc = F.softmax(ol, -1), F.softmax(cl, -1)
        conf *= po.max().item() * pc.max().item()
        conds.append([int(pc.argmax()), OPS[int(po.argmax())], " ".join(toks[a:z])])
    return (conds or None), conf


def predict(split, sample=False):
    torch.set_num_threads(1)
    model = Sieve2()
    model.load_state_dict(torch.load(ROOT / "sieve2.pt", map_location="cpu"))
    model.eval()
    d, tabs = load(split)
    idx = json.load(open(ROOT / "test_sample.json")) if sample else range(len(d["examples"]))
    name = f"sieve2.{split}{'-sample' if sample else ''}.jsonl"
    with open(ROOT / "preds" / name, "w") as f:
        for i in idx:
            e = d["examples"][i]
            t0 = time.perf_counter()
            conds, conf = predict_one(model, e["toks"], tabs[e["table"]])
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"i": i, "conds": conds, "conf": conf, "ms": ms}) + "\n")
    print("wrote", name)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        predict(sys.argv[2], "--sample" in sys.argv)
