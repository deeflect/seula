"""Comparison chart for the README, built from the scored prediction files (no plotting deps).

  python3 bench/chart.py out/fb.test.json ../docs/seula-comparison.svg NAME=preds.jsonl[:ms] ... --curve NAME --cascade "LABEL=PCT"

Left: rows-correct per method. Right: risk-coverage (answer only the most confident X%: how often right?).
"""
import json
import statistics
import sys

from cascade import correct, load

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8d8c86"


def stats(bench, path):
    preds = load(path)
    ok = correct(bench, preds)
    ans = [p for p in preds if p["conds"]]
    ms = [p["ms"] for p in preds if p.get("ms") is not None]
    return {
        "right": 100 * sum(ok.values()) / len(preds),
        "cov": 100 * len(ans) / len(preds),
        "prec": 100 * sum(ok[p["i"]] for p in ans) / max(1, len(ans)),
        "ms": statistics.median(ms) if ms else None,
        "curve": [(100 * k / len(preds), 100 * h / k) for k, h in _ranked(ans, ok)],
    }


def _ranked(ans, ok):
    hits = 0
    for k, p in enumerate(sorted(ans, key=lambda p: -p["conf"]), 1):
        hits += ok[p["i"]]
        yield k, hits


def ms_label(ms):
    return "" if ms is None else f"{ms:.2f} ms" if ms < 10 else f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def svg(rows, curve_name, extra_bars):
    W, H = 940, 400
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="13">',
           """<style>
  .bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df}.ring{stroke:#fcfcfb}
  @media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#3a3a37}.ring{stroke:#1a1a19}}
</style>""",
           f'<rect class="bg" width="{W}" height="{H}" rx="12"/>',
           '<text class="t1" x="24" y="34" font-size="16" font-weight="600">Rows correct on FilterBench test (870 queries, unseen tables)</text>']
    bars = sorted([(n, s["right"], s["ms"], n == curve_name) for n, s in rows.items()] + extra_bars, key=lambda b: -b[1])
    x0, bw, y = 150, 200, 70
    for name, v, ms, hi in bars:
        out.append(f'<text class="t1" x="{x0 - 10}" y="{y + 13}" text-anchor="end">{name}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{bw * v / 100:.1f}" height="18" rx="4" fill="{BLUE if hi or name.startswith(curve_name) else GRAY}"/>')
        out.append(f'<text class="t1" x="{x0 + bw * v / 100 + 6:.1f}" y="{y + 13}">{v:.1f}%</text>')
        out.append(f'<text class="t2" x="{x0 + bw + 150}" y="{y + 13}" text-anchor="end">{ms if isinstance(ms, str) else ms_label(ms)}</text>')
        y += 34
    out.append(f'<text class="t2" x="{x0 + bw + 150}" y="60" text-anchor="end" font-size="11">time per query</text>')

    # risk-coverage panel
    px, py, pw, ph, ylo = 590, 80, 320, 250, 40
    X = lambda c: px + pw * c / 100  # noqa: E731
    Y = lambda p: py + ph * (100 - p) / (100 - ylo)  # noqa: E731
    out.append(f'<text class="t1" x="{px}" y="34" font-size="16" font-weight="600">Answer only when sure</text>')
    for p in range(ylo, 101, 10):
        out.append(f'<line class="grid" x1="{px}" x2="{px + pw}" y1="{Y(p):.1f}" y2="{Y(p):.1f}"/><text class="t2" x="{px - 6}" y="{Y(p) + 4:.1f}" text-anchor="end" font-size="11">{p}%</text>')
    for c in range(0, 101, 25):
        out.append(f'<text class="t2" x="{X(c):.1f}" y="{py + ph + 16}" text-anchor="middle" font-size="11">{c}%</text>')
    out.append(f'<text class="t2" x="{px + pw / 2}" y="{py + ph + 34}" text-anchor="middle" font-size="11">share of queries answered</text>')
    pts = rows[curve_name]["curve"][9:]  # skip the noisy first few points
    out.append(f'<polyline fill="none" stroke="{BLUE}" stroke-width="2" stroke-linejoin="round" points="{" ".join(f"{X(c):.1f},{Y(p):.1f}" for c, p in pts)}"/>')
    out.append(f'<text class="t2" x="{px}" y="56" font-size="12">% correct among answered, as the share answered grows</text>')
    out.append(f'<text class="t1" x="{X(30):.1f}" y="{Y(100) + 22:.1f}">{curve_name}</text>')
    for name, s in rows.items():
        if name == curve_name:
            continue
        cx, cy = X(s["cov"]), Y(max(ylo, s["prec"]))
        col = ORANGE if "Haiku" in name else GRAY
        out.append(f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{col}" stroke-width="2"/>')
        out.append(f'<text class="t2" x="{cx - 9:.1f}" y="{cy + 4:.1f}" text-anchor="end" font-size="11">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    a = sys.argv
    bench, dest = a[1], a[2]
    curve = a[a.index("--curve") + 1]
    specs = [x for x in a[3:] if "=" in x and not x.startswith("--") and a[a.index(x) - 1] not in ("--curve", "--cascade")]
    rows = {}
    for sp in specs:
        name, rest = sp.split("=", 1)
        path, _, ms = rest.partition(":")
        rows[name] = stats(bench, path)
        if ms:
            rows[name]["ms"] = float(ms)
    extra = []
    if "--cascade" in a:
        label, pct, ms = a[a.index("--cascade") + 1].split("=")
        extra.append((label, float(pct), ms, True))
    for n, s in rows.items():
        print(f"{n:24s} right {s['right']:.1f}  answered {s['cov']:.1f}  prec {s['prec']:.1f}  ms {s['ms']}")
    open(dest, "w").write(svg(rows, curve, extra))
    print("wrote", dest)
