"""metrics.py : every metric used in the paper

TPR  = share of attacks caught (recall)        FPR = share of benign messages flagged
Precision and F1 are computed from TP/FP/FN
ROC-AUC     = area under the ROC curve (1.0 is perfect, 0.5 is chance)
TPR@FPR=1%  = pick the threshold on validation so that FPR <= 1%, then measure TPR on test (the PromptShield protocol)
ASR         = Attack Success Rate = share of attacks that reach the LLM (in the paper: the detector judged them safe)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np


def rates(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    tpr = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * tpr / (prec + tpr)) if (tp > 0 and prec == prec and tpr == tpr and (prec + tpr) > 0) else (
        0.0 if n_pos else float("nan"))
    return dict(TPR=tpr, FPR=fpr, Precision=prec, F1=f1, TP=tp, FP=fp, TN=tn, FN=fn, n_pos=n_pos, n_neg=n_neg)


def auc_score(y_true: Sequence[int], score: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(score, dtype=float)))


def auc_from_points(fprs: Iterable[float], tprs: Iterable[float]) -> float:
    """AUC from the (FPR, TPR) points of a sweep (the corners (0,0) and (1,1) are added)"""
    pts = sorted(set(zip(map(float, fprs), map(float, tprs))) | {(0.0, 0.0), (1.0, 1.0)})
    xs = np.array([p[0] for p in pts])
    ys = np.maximum.accumulate(np.array([p[1] for p in pts]))   # force a non-decreasing curve
    return float(np.trapz(ys, xs)) if hasattr(np, "trapz") else float(np.trapezoid(ys, xs))


def threshold_for_fpr(neg_scores: Sequence[float], target_fpr: float) -> float:
    """Finds the thr for which "score > thr" flags at most target_fpr of the benign data

    Used when calibrating on validation: at most floor(target * n) benign items may be flagged.
    If n is so small that floor = 0, thr becomes the maximum of the benign data (FPR = 0, which is stricter than the target)
    """
    s = np.sort(np.asarray(list(neg_scores), dtype=float))[::-1]
    n = len(s)
    if n == 0:
        return math.inf
    k = int(math.floor(target_fpr * n + 1e-9))
    if k >= n:
        return -math.inf
    return float(s[k])


def target_reached(achieved_fpr: float, target_fpr: float, tol: float = 0.25) -> bool:
    """Was the operating point actually reached?

    A threshold calibrated on validation only transfers if the FPR measured on test stays near the target.
    PromptShield (Jacob et al., 2025) accepts a threshold whose FPR falls within 25% of the target, and we use
    the same tolerance: a model whose score takes few distinct values (a shallow tree, a rule) often cannot be
    placed at the target at all, and its TPR at that "operating point" is then not comparable with the others.
    """
    if achieved_fpr != achieved_fpr or target_fpr <= 0:
        return False
    return bool(abs(achieved_fpr - target_fpr) <= tol * target_fpr)


def expected_fp(n_neg: int, target_fpr: float) -> Dict[str, float]:
    """How well a target FPR can be estimated from n_neg negatives

    The relative standard error of a rate is about 1/sqrt(expected false positives), so ten expected false
    positives give about 30% error and a hundred about 10% (Section IV-A of the paper).
    """
    k = float(n_neg) * float(target_fpr)
    return dict(n_neg=int(n_neg), expected_fp=k,
                rel_se=float("inf") if k <= 0 else float(1.0 / math.sqrt(k)),
                estimable=bool(k >= 10.0))


def tpr_at_fpr(y_true: Sequence[int], score: Sequence[float], target_fpr: float,
               thr: float | None = None) -> Dict[str, float]:
    """TPR at a target FPR. With no thr given, the threshold is computed on the same data (for use on validation).
    When reporting test results, pass in the threshold obtained on validation (never compute it on test)."""
    y = np.asarray(y_true).astype(int)
    sc = np.asarray(score, dtype=float)
    if thr is None:
        thr = threshold_for_fpr(sc[y == 0], target_fpr)
    out = rates(y, (sc > thr).astype(int))
    out["thr"] = thr
    out["target_fpr"] = target_fpr
    out["resolvable"] = bool(out["n_neg"] * target_fpr >= 1.0)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, level: float = 0.95,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Mean and bootstrap (percentile) interval of a list of numbers (e.g. one value per seed)"""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def bootstrap_stat_ci(y_true: Sequence[int], y_pred: Sequence[int], stat: Callable[[np.ndarray, np.ndarray], float],
                      n_boot: int = 1000, level: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Item-level bootstrap within a single seed (resamples the (y_true, y_pred) pairs)"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    if len(y) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        v = stat(y[idx], p[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def attack_success_rate(attack_passed: Sequence[int]) -> float:
    """ASR = share of attacks the detector judged safe (so they reached the LLM)"""
    a = np.asarray(attack_passed)
    return float(a.mean()) if len(a) else float("nan")


def summarize_over_seeds(per_seed: Sequence[Dict[str, float]], keys: Iterable[str] | None = None,
                         n_boot: int = 2000) -> Dict[str, Dict[str, float]]:
    """Pools several seeds -> {metric: {mean, lo, hi, n}}"""
    if not per_seed:
        return {}
    keys = list(keys) if keys is not None else [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = [d[k] for d in per_seed if k in d and d[k] == d[k]]
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
        out[k] = dict(mean=m, lo=lo, hi=hi, n=len(vals))
    return out
