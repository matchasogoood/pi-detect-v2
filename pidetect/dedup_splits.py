"""dedup_splits.py : leak-free helpers for pidetect/sessions.py::make_splits.

NOT imported by the running experiment. Apply it only after the current run has finished (see APPLY.md).

What it prevents (the leaks measured by audit_g.py / leak_breakdown_g.py):
  1. the same prompt, up to case/spacing/unicode, in dev and in val/test      -> normalised exact de-duplication
  2. near-duplicates of one prompt (Gandalf / deepset variants) on both sides  -> cluster, then split WHOLE clusters
  3. one text labelled both attack and benign (deepset label conflicts)        -> dropped from both sides
  4. a BIPIA context in the official train AND test split                      -> removed from train
  5. any of the above creeping back in                                         -> leak_gate() stops the run
No model, no label and no test score is used to choose anything here; the threshold is fixed beforehand.
"""
from __future__ import annotations

import random
import unicodedata
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

NEAR_DUP_JACCARD = 0.8        # fixed in advance; checked only by the leak count, never by detection results
SHINGLE = 3                   # word n-gram size


def norm(t) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(t)).lower().split())


def _shingles(t: str) -> frozenset:
    w = norm(t).split()
    if len(w) <= SHINGLE:
        return frozenset([" ".join(w)])
    return frozenset(" ".join(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1))


def clusters(texts: Sequence[str], thr: float = NEAR_DUP_JACCARD) -> List[int]:
    """Cluster id per text: exact (normalised) or near duplicates (Jaccard of word 3-grams >= thr) share an id.
    Uses an inverted index on shingles, so it is fast for a few thousand texts."""
    n = len(texts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sh = [_shingles(t) for t in texts]
    index: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        cand = set()
        for g in s:
            cand.update(index[g])
        for j in cand:
            inter = len(s & sh[j])
            if inter and inter / len(s | sh[j]) >= thr:
                parent[find(i)] = find(j)
        for g in s:
            index[g].append(i)
    return [find(i) for i in range(n)]


def drop_label_conflicts(rows: Sequence[dict], text_key: str = "text", label_key: str = "label") -> List[dict]:
    """Removes every row whose normalised text appears with both labels."""
    seen: Dict[str, set] = defaultdict(set)
    for r in rows:
        seen[norm(r[text_key])].add(r[label_key])
    return [r for r in rows if len(seen[norm(r[text_key])]) == 1]


def group_split(items: Sequence, texts: Sequence[str], seed: int,
                fracs: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> Dict[str, list]:
    """Like split_frac, but a whole duplicate cluster goes to ONE split. Fractions are met by item count (approximately)."""
    cid = clusters(texts)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(cid):
        groups[c].append(i)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    total = len(items)
    cut = (fracs[0] * total, (fracs[0] + fracs[1]) * total)
    out = {"dev": [], "val": [], "test": []}
    filled = 0
    for g in order:
        name = "dev" if filled < cut[0] else ("val" if filled < cut[1] else "test")
        out[name].extend(items[i] for i in g)
        filled += len(g)
    return out


def drop_shared_contexts(train_ctx: Sequence[str], test_ctx: Sequence[str]) -> List[str]:
    """BIPIA: contexts of the official train split that also occur in the official test split are removed from train."""
    t = {norm(c) for c in test_ctx}
    return [c for c in train_ctx if norm(c) not in t]


def leak_report(train_texts: Iterable[str], eval_texts: Iterable[str], thr: float = NEAR_DUP_JACCARD) -> Dict[str, int]:
    train_texts, eval_texts = list(train_texts), list(eval_texts)
    tn = {norm(t) for t in train_texts}
    exact = sum(norm(t) in tn for t in eval_texts)
    cid = clusters(train_texts + eval_texts, thr)
    train_c = set(cid[:len(train_texts)])
    near = sum(c in train_c for c in cid[len(train_texts):])
    return dict(exact=exact, near=near, n_eval=len(eval_texts))


def leak_gate(pairs: Dict[str, Tuple[Sequence[str], Sequence[str]]], max_exact: int = 0, max_near: int = 0) -> Dict[str, dict]:
    """pairs = {name: (dev_texts, val_or_test_texts)}. Raises if any pair leaks; returns the counts for the log."""
    rep = {k: leak_report(a, b) for k, (a, b) in pairs.items()}
    bad = {k: v for k, v in rep.items() if v["exact"] > max_exact or v["near"] > max_near}
    if bad:
        raise RuntimeError(f"data leak between splits: {bad}")
    return rep
