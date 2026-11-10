"""Shared features for Sieve v2 and the rules baseline.

Everything here must be cheap to reproduce in TypeScript: lowercase regex
tokens, FNV-1a hashing, suffix stemming, exact n-gram lookup in table cells.
"""
import unicodedata

from prepare import tokenize

MAX_TOKENS = 48
MAX_HEADER_WORDS = 6
MONTH_FULL = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june", "jul": "july", "aug": "august",
              "sep": "september", "sept": "september", "oct": "october", "nov": "november", "dec": "december"}
MAX_NGRAM = 8
NUM_LINK = 6  # per (token, column) link features, see link_matrix


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def hash_ids(word: str, buckets: int) -> list[int]:
    """Word id + char trigram ids, all in one hashed table (fastText-style)."""
    ids = [fnv1a("w:" + word) % buckets]
    padded = f"<{word}>"
    for i in range(len(padded) - 2):
        ids.append(fnv1a(padded[i:i + 3]) % buckets)
    return ids[:16]


def stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    for suf, keep in (("ing", 5), ("ed", 4), ("es", 4), ("s", 3)):
        if len(w) > keep and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def norm_word(w: str) -> str:
    """Accents stripped, month abbreviation expanded, stemmed; "" for punctuation."""
    w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c)[0] != "M")
    return stem(MONTH_FULL.get(w, w)) if any(c.isalnum() for c in w) else ""


def bag(toks: list[str]) -> str:
    """Order- and punctuation-free key: "sept 14, 2004" and "14 September 2004" share one.
    Number tokens split at commas, so the cell "September14,2004" gives the same key."""
    return " ".join(sorted(p for w in toks for p in map(norm_word, w.split(",")) if p))


def is_number(w: str) -> bool:
    try:
        float(w.replace(",", ""))
        return True
    except ValueError:
        return False


class TableIndex:
    """Header tokens and a cell-text -> columns lookup for one table."""

    def __init__(self, table: dict):
        self.header = [tokenize(h) for h in table["header"]]
        self.header_stems = [{stem(t) for t in h} for h in self.header]
        self.numeric = [t == "real" for t in table["types"]]
        from filterbench import parse_number
        nums = [[abs(n) for r in table["rows"] if (n := parse_number(r[j])) is not None] for j in range(len(self.header))]
        self.absmax = [max(v, default=0.0) for v in nums]  # column scale: is "10 million" 1e7 or 10 (header "(millions)")?
        self.cells: dict[str, set[int]] = {}
        self.bags: dict[str, dict[int, set[str]]] = {}  # bag key -> column -> the cells that share it
        for row in table["rows"]:
            for j, cell in enumerate(row):
                key = " ".join(tokenize(str(cell)))
                if key and key.count(" ") < MAX_NGRAM:
                    self.cells.setdefault(key, set()).add(j)
                    b = bag(key.split())
                    if b:
                        self.bags.setdefault(b, {}).setdefault(j, set()).add(key)

    @property
    def ncols(self) -> int:
        return len(self.header)


def cell_matches(toks: list[str], tab: TableIndex, fuzzy=False) -> list[tuple[int, int, int]]:
    """All (start, end, col) where toks[start:end] equals a whole cell of col.
    fuzzy: also when they share a bag key (word order, punctuation, accents, plurals, "sept")."""
    out = []
    nw = [bag([w]) for w in toks] if fuzzy else []
    for i in range(len(toks)):
        for n in range(1, MAX_NGRAM + 1):
            if i + n > len(toks):
                break
            cols = set(tab.cells.get(" ".join(toks[i:i + n]), ()))
            if fuzzy and nw[i] and nw[i + n - 1]:
                cols |= tab.bags.get(bag(toks[i:i + n]), {}).keys()
            out.extend((i, i + n, c) for c in sorted(cols))
    return out


def link_matrix(toks: list[str], tab: TableIndex, matches=None) -> list[list[list[float]]]:
    """L[t][j] = features linking token t to column j.

    0 token is a header word of j      1 token stem is a header stem of j
    2 token inside a cell match of j   3 that cell match is multi-token
    4 token numeric and j numeric      5 header of j mentioned anywhere (broadcast)
    """
    matches = cell_matches(toks, tab) if matches is None else matches
    T, C = len(toks), tab.ncols
    L = [[[0.0] * NUM_LINK for _ in range(C)] for _ in range(T)]
    mentioned = [False] * C
    for t, w in enumerate(toks):
        s = stem(w)
        num = is_number(w)
        for j in range(C):
            if w in tab.header[j]:
                L[t][j][0] = 1.0
                mentioned[j] = True
            if s in tab.header_stems[j]:
                L[t][j][1] = 1.0
            if num and tab.numeric[j]:
                L[t][j][4] = 1.0
    for a, b, j in matches:
        for t in range(a, b):
            L[t][j][2] = 1.0
            if b - a > 1:
                L[t][j][3] = 1.0
    for t in range(T):
        for j in range(C):
            L[t][j][5] = 1.0 if mentioned[j] else 0.0
    return L
