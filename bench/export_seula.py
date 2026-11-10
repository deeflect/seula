"""Export Seula weights for the browser + a parity fixture for the TS port.

  cd bench && [OUT=seulab.pt] uv run -q --with torch --with numpy python3 export_seula.py

weights.bin layout: uint32 LE header length N, N bytes UTF-8 JSON header
[{name, shape, scale, offset}], then int8 tensors (value = q * scale), per-tensor symmetric.
Parity outputs are computed with the dequantized weights, so TS and Python share them exactly.
"""
import json
import os
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from seula import MAX_TOKENS, ROOT, SeulaNet, TableIndex, execution_guided, fmt_num, parse_number, predict_one, tokenize  # noqa: E402

OUT_BIN = Path(__file__).parent.parent / "src" / "seula" / "weights.bin"
OUT_PARITY = Path(__file__).parent.parent / "spike" / "seula_parity.json"
N_CASES = 40


def quantize(sd):
    header, blobs, off, deq = [], [], 0, {}
    for name, w in sd.items():
        w = w.float()
        scale = float(w.abs().max()) / 127 or 1.0
        q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
        deq[name] = q.float() * torch.tensor(scale, dtype=torch.float32)
        header.append({"name": name, "shape": list(w.shape), "scale": scale, "offset": off})
        b = q.numpy().tobytes()
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header).encode()
    return struct.pack("<I", len(hj)) + hj + b"".join(blobs), deq


def predict(model, q, table):
    tab, toks = TableIndex(table), tokenize(q)[:MAX_TOKENS]
    conds, conf, alts = predict_one(model, toks, tab)
    return execution_guided(toks, tab, table["rows"], conds, conf, alts)


def main():
    torch.set_num_threads(1)
    ckpt = ROOT / os.environ.get("OUT", "seula.pt")
    print("checkpoint", ckpt)
    sd = torch.load(ckpt, map_location="cpu")
    blob, deq = quantize(sd)
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_BIN.write_bytes(blob)
    print(f"wrote {OUT_BIN} ({len(blob) / 1024:.1f} KB)")

    model = SeulaNet()
    model.load_state_dict(deq)
    model.eval()
    d = json.load(open(ROOT / "fb.dev.json"))
    ex = d["examples"]
    step = max(1, len(ex) // N_CASES)
    # Hand-written: lower+upper bound on one column (shared-column branch) and a quoted/dashed value (snap words()).
    BIG = {"header": ["Country", "Population", "GDP (USD)"], "types": ["text", "real", "real"],
           "rows": [["Malta", 520000, 2.1e7], ["Latvia", 1880000, 4.0e7], ["Austria", 9100000, 4.7e8], ["Poland", 36700000, 8.1e8]]}
    PLACES = {"header": ["Location", "Date", "District"], "types": ["text", "text", "text"],
              "rows": [["Pascagoula, Mississippi", "Oct. 2", "1st"], ["Mobile, Alabama", "Seated May 23, 1808", "2nd"], ["Biloxi, Mississippi", "Oct. 9", "3rd"]]}
    STAFF = {"header": ["Name", "Role", "Status", "Hired"], "types": ["text", "text", "text", "text"],
             "rows": [["Jürgen Müller", "Senior Manager", "Overdue", "September 14, 2004"], ["Anna Schmidt", "Engineer", "Pending", "March 3, 2010"],
                      ["Zoë Brown", "Manager", "Paid", "14 Oct 2011"], ["Li Wei", "Engineer", "Overdue", "2 June 2015"]]}
    SHOW = {"header": ["No.", "Title", "Directed by", "Written by"], "types": ["real", "text", "text", "text"],
            "rows": [[1, "Pilot", "Marcos Siega", "Teleplay by: Kevin Williamson & Julie Plec"], [2, "The Night of the Comet", "Marcos Siega", "Kevin Williamson & Julie Plec"],
                     [3, "Friday Night Bites", "John Dahl", "Barbie Kligman & Bryan M. Holdman"], [4, "Family Ties", "Guy Ferland", "Andrew Kreisberg & Brian Young"]]}
    hand = [
        {"table": "2-12382876-3", "q": "heat over 2 and under 5"},
        {"table": "2-12382876-3", "q": "lane between 3 and 6"},
        {"table": "2-12382876-3", "q": "heat >= 3 < 5"},
        {"table": "2-1763266-9", "q": 'score "1-2"'},
        {"table": "2-12382876-3", "q": "lane over 10 million"},  # small column: "million" not applied
        {"table": BIG, "q": "population over 10 million"},  # big column: "million" applied
        {"table": BIG, "q": "gdp over 2 m and population under 5 million"},
        {"table": "2-12382876-3", "q": "heat over 10000000"},  # plain digits, not 1e+07
        # snap: punctuation ignored, spaces ignored, unique containing cell
        {"table": PLACES, "q": "location pascagoula mississippi"},
        {"table": PLACES, "q": "date oct 2"},
        {"table": PLACES, "q": "district 1 st"},
        {"table": PLACES, "q": "date may 23 1808"},
        # execution-guided: first parse returns no rows -> next-best column / op, or conf 0
        {"table": "1-2818164-8", "q": "show me episodes 160 or higher in the series that aired on March 7, 1991"},
        {"table": "2-13741576-4", "q": "played<=22, tries for>=39.75, drawn>=1"},
        {"table": "2-10814483-2", "q": "grattan stadum, opp at least 23"},
        {"table": "2-16882018-1", "q": "tight -oregon"},
        # fuzzy cells (bag keys), or-lists, snap_words, empty parse
        {"table": STAFF, "q": "name muller"},  # accents
        {"table": STAFF, "q": "hired sept 14 2004"},  # month abbreviation + order/punctuation
        {"table": STAFF, "q": "role managers"},  # plural -> cell
        {"table": STAFF, "q": "status is overdue or pending"},  # or-list on one column
        {"table": STAFF, "q": "role engineer or manager"},
        {"table": STAFF, "q": "status overdue, pending"},  # comma-only: may share
        {"table": STAFF, "q": "status overdue, pending or paid"},
        {"table": STAFF, "q": "overdue or pending, engineer"},  # comma after the or-group: new group
        {"table": STAFF, "q": "status not overdue or pending, role engineer"},
        {"table": STAFF, "q": "role contains managers"},  # snap_words
        {"table": STAFF, "q": "name contains muller"},
        {"table": STAFF, "q": "the"},  # empty parse
        {"table": STAFF, "q": "?"},
        {"table": "2-12382876-3", "q": "lane under 5 yards"},  # unit after the number
        {"table": BIG, "q": "population 5k+"},
        {"table": BIG, "q": "population between 520000 and 1880000"},  # digits inside other cells: still a range, not "contains"
        {"table": SHOW, "q": "written by julie plec"},  # two adjacent values found together in one cell: merged
        {"table": STAFF, "q": "overdue paid"},  # adjacent but in no single cell: stays any-of
    ]
    cases = []
    for e in ex[::step][:N_CASES] + hand:
        t = e["table"] if isinstance(e["table"], dict) else d["tables"][e["table"]]
        table = {"header": t["header"], "types": t["types"], "rows": t["rows"]}
        conds, conf = predict(model, e["q"], table)
        cases.append({"q": e["q"], "table": table, "conds": conds, "conf": conf})

    tok_cases = ["Café Straße ÉLAN naïve", "über 1,000.50 km²", "₹500 – ½ price_tag", "東京 2020年", "Ⅻ ٣٤ apples", "x_y\tz w", "éclair", "ΣΊΣΥΦΟΣ ok"]
    num_cases = ["1.5k", "$2,000", "30%", "-5", "1000000", "1234567", "0.00001", "0.0001", "12.50", "3bn", "7million", "abc", "1e3", "٣٤", "100%%", "2.5m"]
    units = {
        "tok": [[s, tokenize(s)] for s in tok_cases],
        "num": [[s, None if (n := parse_number(s)) is None else fmt_num(n)] for s in num_cases],
    }
    json.dump({"cases": cases, "units": units}, open(OUT_PARITY, "w"), ensure_ascii=False)
    print(f"wrote {OUT_PARITY} ({len(cases)} cases, {sum(c['conds'] is not None for c in cases)} with conds)")


if __name__ == "__main__":
    main()
