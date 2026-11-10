"""Small local LLM baseline for filter extraction (same task as Sieve).

uv run --with torch --with transformers --with accelerate python3 bench/llm_baseline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --split test --sample bench/out/test_sample.json \
  --out bench/out/preds/qwen25-05b.test.jsonl [--limit 20] [--device mps|cpu]

Writes evaluate.py-format jsonl (resumable) + <out>.meta.json.
"""
import argparse
import json
import math
import os
import re
import resource
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).parent / "out"


def table_block(t: dict) -> str:
    cols = "\n".join(f"{k + 1}. {h} ({ty})" for k, (h, ty) in enumerate(zip(t["header"], t["types"])))
    rows = "\n".join(" | ".join(str(c)[:40] for c in r) for r in t["rows"][:3])
    return f"Columns:\n{cols}\nExample rows:\n{rows}"


def answer_json(t: dict, conds) -> str:
    return json.dumps({"conditions": [{"column": t["header"][c], "op": o, "value": v} for c, o, v in conds]})


def pick_shots(train: dict) -> list[dict]:
    # Deterministic: first example with 1 cond, first with 2, first with a > or < cond.
    ex = train["examples"]
    tests = [lambda e: len(e["conds"]) == 1 and e["conds"][0][1] == "=",
             lambda e: len(e["conds"]) == 2,
             lambda e: any(o in "<>" for _, o, _ in e["conds"])]
    shots = []
    for f in tests:
        shots.append(next(e for e in ex if f(e) and e not in shots))
    return shots


SYSTEM = ("You convert a question about a table into row filters (SQL WHERE conditions). "
          "Return ONLY the conditions that restrict which rows match; do not include the column being "
          "selected or any aggregation (count, max, sum...). Use the exact column header text. "
          "op is one of \"=\", \">\", \"<\". value is copied from the question. "
          'Reply with JSON only: {"conditions":[{"column":"<exact header>","op":"=|>|<","value":"<text>"}]}')


FB_OPS = ("=", "!=", "contains", ">", ">=", "<", "<=")
FB_SYSTEM = ("You convert a filter-box query about a table into row filters. "
             "Return ONLY conditions that restrict which rows match. Use the exact column header text. "
             "Ops: \"=\" equals the value; \"!=\" excludes rows equal to that exact value; "
             "\"contains\" means the column includes that word or text; \">\" \">=\" \"<\" \"<=\" compare numbers. "
             "value is taken from the query (numbers like 1k or 2.5m are fine). "
             'Reply with JSON only: {"conditions":[{"column":"<exact header>","op":"=|!=|contains|>|>=|<|<=","value":"<text>"}]}')


def pick_fb_shots(dev: dict) -> list[dict]:
    # Deterministic: first single numeric comparison, first !=, first multi-condition.
    ex = dev["examples"]
    tests = [lambda e: len(e["conds"]) == 1 and e["conds"][0][1] in (">", ">=", "<", "<="),
             lambda e: any(o == "!=" for _, o, _ in e["conds"]),
             lambda e: len(e["conds"]) >= 2]
    shots = []
    for f in tests:
        shots.append(next(e for e in ex if f(e) and e not in shots))
    return shots


def user_msg(t: dict, q: str) -> str:
    return f"{table_block(t)}\n\nQuestion: {q}"


def build_messages(shots, train, t, q, system=SYSTEM):
    msgs = [{"role": "system", "content": system}]
    for s in shots:
        st = train["tables"][s["table"]]
        msgs += [{"role": "user", "content": user_msg(st, s["q"])},
                 {"role": "assistant", "content": answer_json(st, s["conds"])}]
    msgs.append({"role": "user", "content": user_msg(t, q)})
    return msgs


def parse(text: str, header: list[str], ops=("=", ">", "<")):
    """First JSON object -> [[col_idx, op, value]] or None (abstain)."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        conds = obj["conditions"]
        lower = [h.lower().strip() for h in header]
        out = []
        for c in conds:
            name, op, val = str(c["column"]).lower().strip(), str(c["op"]).strip(), c["value"]
            if op not in ops or val is None or str(val).strip() == "":
                return None
            if name in lower:
                k = lower.index(name)
            else:
                cand = [k for k, h in enumerate(lower) if h.startswith(name) or name.startswith(h)]
                if len(cand) != 1:
                    return None
                k = cand[0]
            out.append([k, op, str(val)])
        return out or None
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def disk_size(model_id: str) -> int:
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(model_id, local_files_only=True))
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and f.suffix in (".safetensors", ".bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bench", choices=["wikisql", "fb"], default="wikisql")
    ap.add_argument("--sample", help="wikisql: index list json (required); fb: all examples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    a = ap.parse_args()

    if a.bench == "fb":
        train = json.load(open(OUT / "fb.dev.json"))
        data = json.load(open(OUT / f"fb.{a.split}.json"))
        idx = json.load(open(a.sample)) if a.sample else list(range(len(data["examples"])))
        shots, system, ops = pick_fb_shots(train), FB_SYSTEM, FB_OPS
    else:
        train = json.load(open(OUT / "train.json"))
        data = json.load(open(OUT / f"{a.split}.json"))
        idx = json.load(open(a.sample))
        shots, system, ops = pick_shots(train), SYSTEM, ("=", ">", "<")
    if a.limit:
        idx = idx[: a.limit]

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["i"] for l in open(a.out) if l.strip()}
    todo = [i for i in idx if i not in done]

    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float16 if a.device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    tmpl_kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}

    def run(i):
        e = data["examples"][i]
        t = data["tables"][e["table"]]
        prompt = tok.apply_chat_template(build_messages(shots, train, t, e["q"], system), tokenize=False,
                                         add_generation_prompt=True, **tmpl_kw)
        enc = tok(prompt, return_tensors="pt").to(a.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, temperature=None,
                               top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if a.device == "mps":
            torch.mps.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        gen = g.sequences[0, enc["input_ids"].shape[1]:]
        lps = [torch.log_softmax(s[0].float(), -1)[tid].item() for s, tid in zip(g.scores, gen)]
        conf = math.exp(sum(lps) / max(1, len(lps)))
        text = tok.decode(gen, skip_special_tokens=True)
        return {"i": i, "conds": parse(text, t["header"], ops), "conf": conf, "ms": ms, "raw": text}

    run(idx[0])  # warm-up, not recorded
    peak = 0
    with open(a.out, "a") as f:
        for n, i in enumerate(todo, 1):
            r = run(i)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if a.device == "mps":
                peak = max(peak, torch.mps.driver_allocated_memory())
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} ms={r['ms']:.0f} conds={r['conds']}", flush=True)

    rows = [json.loads(l) for l in open(a.out)]
    meta = {"model": a.model, "device": a.device, "bench": a.bench, "dtype": str(dtype), "n": len(rows),
            "params": sum(p.numel() for p in model.parameters()), "disk_bytes": disk_size(a.model),
            "peak_mps_driver_bytes": peak, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "parse_failures": sum(r["conds"] is None for r in rows),
            "shots": [s["q"] for s in shots]}
    json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
