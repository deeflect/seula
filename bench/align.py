"""Find where each gold condition's value appears in a typed query (training labels).

Handles exact tokens, numbers written differently ("1k" == 1000, "16,000"), and small
typos ("silvr"). Returns a list of (start, end) per condition, or None if any is missing.
"""
from filterbench import norm_text, parse_number
from prepare import tokenize


def edit1(a: str, b: str) -> bool:
    """Damerau distance <= 1 (cheap check)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


DASHES = {"–": "-", "—": "-", "−": "-"}
QUOTES = {'"', "'", "(", ")", "“", "”", "‘", "’", ","}


def content(toks):
    """Tokens that carry meaning for matching: dashes unified, quotes/brackets dropped."""
    return [DASHES.get(w, w) for w in toks if w not in QUOTES]


def span_score(toks, a, z, value) -> float:
    span = toks[a:z]
    pv = parse_number(value)
    if pv is not None:
        ps = parse_number(" ".join(span)) if span else None
        return 1.0 if ps is not None and abs(ps - pv) < 1e-9 else 0.0
    vt = tokenize(value)
    if span == vt:
        return 1.0
    span, vt = content(span), content(vt)
    if not span or toks[a] in QUOTES or toks[z - 1] in QUOTES:
        return 0.0
    if span == vt:
        return 0.9
    if len(span) == len(vt) and all(x == y or (len(y) > 3 and edit1(x, y)) for x, y in zip(span, vt)):
        return 0.8
    return 0.0


def align(toks, conds):
    taken, spans = set(), []
    for _, _, value in conds:
        n = len(tokenize(value)) if parse_number(value) is None else 0
        nc = len(content(tokenize(value)))
        widths = list(range(max(1, nc - 1), max(n, nc) + 2)) if n else [1, 2]  # numbers: "1000" or "1.5 k"
        best = None
        for w in sorted(set(widths), reverse=True):
            for a in range(len(toks) - w + 1):
                if taken & set(range(a, a + w)):
                    continue
                s = span_score(toks, a, a + w, value)
                if s and (best is None or s > best[0]):
                    best = (s, a, a + w)
        if not best:
            return None
        spans.append((best[1], best[2]))
        taken |= set(range(best[1], best[2]))
    return spans


if __name__ == "__main__":
    assert align(tokenize("att >=16k"), [[0, ">=", "16000"]]) == [(3, 5)]
    assert align(tokenize("silvr 0"), [[0, "=", "0"]]) == [(1, 2)]
    assert align(tokenize("player is bob charls"), [[0, "=", "Bob Charles"]]) == [(2, 4)]
    assert align(tokenize("price under 5"), [[0, "<", "7"]]) is None
    assert align(tokenize("score 6-4 please"), [[0, "=", "6–4"]]) == [(1, 4)]
    assert align(tokenize("title pilot"), [[0, "=", '"Pilot"']]) == [(1, 2)]
    assert align(tokenize('title "pilot"'), [[0, "=", '"Pilot"']]) == [(1, 4)]
    assert align(tokenize("pls tunis tunisia"), [[0, "=", "Tunis , Tunisia"]]) == [(1, 3)]
    assert norm_text("A  B") == "a b"
    print("align ok")
