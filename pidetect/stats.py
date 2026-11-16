"""stats.py : the statistics the paper uses to decide whether a hypothesis is supported

paired_bootstrap   : 95% interval of the difference between two methods on the same sessions or events (paired resampling)
                     the criterion in Table IV is that this interval excludes 0
holm               : adjusts p-values when several pairs are tested (Holm correction)
mcnemar            : tests the difference between two classifiers on the same examples (exact binomial)
corrected_ttest    : corrected resampled t-test for repeated cross-validation (used in the preliminary experiment / Table VI)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0,
                     strata: Sequence[int] | None = None) -> Dict[str, float]:
    """a, b = the per-unit values (e.g. detected 0/1 per session) of methods A and B on the same units, aligned.
    Returns diff = mean(a) - mean(b), the 95% interval, p (two-sided), and excludes_zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "a and b must cover the same units"
    n = len(a)
    if n == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, excludes_zero=False, n=0)
    d = a - b
    rng = np.random.RandomState(seed)
    if strata is None:
        boots = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    else:
        st = np.asarray(strata)
        groups = [np.where(st == s)[0] for s in np.unique(st)]
        boots = np.array([np.concatenate([d[g[rng.randint(0, len(g), len(g))]] for g in groups]).mean()
                          for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi), p=float(min(1.0, p)),
                excludes_zero=bool(lo > 0 or hi < 0), n=int(n))


def paired_bootstrap_stat(units: Sequence, stat: Callable[[np.ndarray], float], n_boot: int = 1000,
                          seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap of any statistic computed from unit indices (e.g. a TPR@FPR difference that needs a new threshold)"""
    idx = np.arange(len(units))
    rng = np.random.RandomState(seed)
    vals = [stat(idx[rng.randint(0, len(idx), len(idx))]) for _ in range(n_boot)]
    vals = [v for v in vals if v == v]
    point = stat(idx)
    if not vals:
        return point, np.nan, np.nan
    return point, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(((p, k) for k, p in pvals.items() if p == p))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    for k, p in pvals.items():
        out.setdefault(k, np.nan)
    return out


def mcnemar(correct_a: Sequence[int], correct_b: Sequence[int]) -> float:
    """Exact McNemar: the p-value of the number of discordant pairs"""
    from scipy.stats import binomtest
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def corrected_ttest(diffs: Sequence[float], n_train: int, n_test: int) -> Tuple[float, float, float]:
    """Corrected resampled t-test (for k-fold repeated r times): the variance is multiplied by (1/J + n_test/n_train).
    Returns (mean difference, corrected SE, two-sided p)."""
    from scipy.stats import t as tdist
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    if J < 2:
        return float(d.mean()) if J else np.nan, np.nan, np.nan
    var = d.var(ddof=1)
    se = math.sqrt((1.0 / J + n_test / n_train) * var)
    if se == 0:
        return float(d.mean()), 0.0, (1.0 if d.mean() == 0 else 0.0)
    tstat = d.mean() / se
    p = 2 * tdist.sf(abs(tstat), J - 1)
    return float(d.mean()), float(se), float(p)


def corrected_se(values: Sequence[float], n_train: int, n_test: int) -> float:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.nan
    return float(math.sqrt((1.0 / len(v) + n_test / n_train) * v.var(ddof=1)))
