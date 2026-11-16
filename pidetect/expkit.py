"""expkit.py : small helpers shared by the experiment scripts (baselines, pooling across seeds, cost, writing summaries)"""
from __future__ import annotations

import os
import time
import tracemalloc
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .baselines import make_baselines
from .metrics import bootstrap_ci
from .report import save_df, to_markdown


def build_baselines(cfg: dict, emb, names: Iterable[str], skip: Sequence[str] = ()) -> Dict[str, object]:
    sk = set(skip)
    # "--skip-baselines B3a" also skips the session-normalized form of the same model, which would load it again
    names = [n for n in names if n not in sk and n.removesuffix("+sn") not in sk]
    built, skipped = make_baselines(cfg, emb, names)
    for k, v in skipped.items():
        print(f"[baseline {k}] skipped: {v}")
    return built


def mean_ci_table(df: pd.DataFrame, by: Sequence[str], cols: Sequence[str]) -> pd.DataFrame:
    """Mean across seeds with a 95% bootstrap interval -> columns <col>, <col>_lo, <col>_hi"""
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for c in cols:
            if c not in g:
                continue
            m, lo, hi = bootstrap_ci(g[c].values.astype(float))
            row[c], row[c + "_lo"], row[c + "_hi"] = m, lo, hi
        row["n_seeds"] = g["seed"].nunique() if "seed" in g else 1
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(out: str, title: str, sections: List[tuple]) -> str:
    """sections = [(heading, DataFrame or text)] -> summary.md"""
    lines = [f"# {title}\n"]
    for head, body in sections:
        lines.append(f"\n## {head}\n")
        if isinstance(body, pd.DataFrame):
            lines.append(to_markdown(body))
        else:
            lines.append(str(body) + "\n")
    path = os.path.join(out, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


class CountingEmbedder:
    """Counts the calls and texts sent to the embedder (with no cache) -- used to measure cost"""

    def __init__(self, base):
        self.base, self.calls, self.texts = base, 0, 0
        self.dim, self.name, self.is_fallback = base.dim, base.name, base.is_fallback

    def encode(self, texts, batch_size: int = 64):
        texts = list(texts)
        self.calls += 1
        self.texts += len(texts)
        return self.base.encode(texts, batch_size=batch_size)


def measure_cost(det: ProposedDetector, sessions, n_events: int = 100):
    rng = np.random.RandomState(0)
    evs = [(s, i) for s in sessions for i in range(len(s.events))]
    pick = [evs[j] for j in rng.permutation(len(evs))[:n_events]]
    cnt = CountingEmbedder(getattr(det.s2.emb, "base", det.s2.emb))
    old_emb = det.s2.emb
    ms, n_txt = [], []
    for s, i in pick:
        det.s2.emb = old_emb
        st = det.new_state()
        for e in s.events[:i]:
            det.step(e, st)
        det.s2.emb = cnt
        det.s2.clear_cache()
        before = cnt.texts
        t0 = time.perf_counter()
        det.step(s.events[i], st)
        ms.append((time.perf_counter() - t0) * 1000)
        n_txt.append(cnt.texts - before)
    det.s2.emb = old_emb
    det.s2.clear_cache()
    tracemalloc.start()
    st = det.new_state()
    for e in sessions[0].events:
        det.step(e, st)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    d = det.s2.emb.dim
    state_bytes = 4 * d + 24 * det.cfg.W + det.cfg.W + 64 * max(1, len(st.B)) + 8
    return dict(latency_mean_ms=float(np.mean(ms)), latency_p95_ms=float(np.quantile(ms, 0.95)),
                embedded_texts_per_event=float(np.mean(n_txt)), state_bytes_per_session=state_bytes,
                python_peak_bytes_one_session=int(peak))


def verdict(ok: bool, what: str) -> str:
    return f"**{'supported' if ok else 'not supported'}** -- {what}"
