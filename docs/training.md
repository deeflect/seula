# Training Seula

Training data is the trick: an LLM writes realistic queries for conditions sampled from real tables, so the labels are known by construction.

```bash
# 1. Generate queries with any OpenAI-compatible endpoint (OpenRouter, llama-server, vLLM...).
#    WikiSQL tables come from bench/prepare.py. About $1 on OpenRouter for all four files.
export LLM_URL=https://openrouter.ai/api LLM_KEY=$OPENROUTER_API_KEY
Q=qwen/qwen3-235b-a22b-2507; M=mistralai/mistral-small-3.2-24b-instruct
python3 bench/filterbench.py gen train 7500 $Q bench/out/fb.qtrain.json
python3 bench/filterbench.py gen train 2400 $Q bench/out/fb.qtrain2.json --seed 4 --ranges
python3 bench/filterbench.py gen train 6000 $M bench/out/fb.mtrain.json --seed 5
python3 bench/filterbench.py gen train 2000 $M bench/out/fb.mtrain2.json --seed 6 --ranges
#    Template-written queries add what LLM writers rarely produce: or-lists, `col:value`, `16k+`. No LLM needed.
python3 bench/synth.py 4200 bench/out/fb.synth2.json
# 2. Train (~20 min on a laptop CPU)
cd bench && FB_TRAIN=fb.qtrain.json,fb.qtrain2.json,fb.mtrain.json,fb.mtrain2.json,fb.synth2.json OUT=seula.pt uv run --with torch python3 seula.py train
# 3. Export int8 weights for the browser + a parity fixture
OUT=seula.pt uv run --with torch --with numpy python3 export_seula.py && cd ..
npm test    # checks the TS runtime against Python
```

## How much generated data you need

FilterBench dev, rows correct:

| Training data | Rows correct |
|---|---:|
| WikiSQL only (no generated queries) | 46.4% |
| + 2,000 generated queries | 85.3% |
| + 7,485 generated queries | 88.9% |
| + 2,205 queries with ranges (v3c) | 90.0% |
| + better value snapping, `feb 25` dates (decoding only, same weights) | 92.0% |
| + zero-rows check (execution-guided decoding, same weights) | 92.7% |

## Who writes the queries

Open-weight writers work as well as Claude. Same code; the Qwen set uses the same sampled conditions as the Haiku set, and the Mistral set adds new ones, so the shipped set is also larger (part of its gain may be quantity, not the second writer). Scored on dev queries written by Haiku and, as a neutral check, by DeepSeek V3.2 (in no training set):

| Training queries written by | Haiku-written dev | DeepSeek-written dev | Test |
|---|---:|---:|---:|
| Claude Haiku (9.7k) | 92.7% | 88.7% | 88.7% |
| Qwen3-235B (9.9k) | 86.2% | 87.8% | n/a |
| Qwen3-235B + Mistral Small 3.2 (17.8k) | 90.9% | 89.6% | 90.1% |
| **Same + 4.2k template queries, a label fix and decoding fixes, shipped** | **95.3%** | **93.6%** | **92.5%** |

The last row is the model alone; with the rule pass in front (what `filter()` runs) test is 92.8%. Most of its dev gain comes from the label fix (`over 5` was sometimes labelled `>=`) and decoding, not the template queries: without them dev is 95.8%. They are there for or-lists, which went from 65% to 93% on a 300-query template probe.

Qwen3-4B was tried as a local writer and dropped: it copies the conditions almost word for word (`date contains january`).

## Most weaknesses are fixed with data

`between 3 and 6` failed because the generator never sampled two bounds on one column. Adding `--ranges` to the generator and 2,205 more queries took a held-out range-only probe (49 queries) from 40.8% to 71.4%, with no code changes. Test moved 84.4% → 85.1%; the decoding fixes took it to 88.7%, the two-writer open-model data to 90.1%, and the current model to 92.5%.
