"""rulebased_tuning.py : tunes the thresholds and parameters on validation data only

The idea (why we do not pick tau by hand)
  We tell the program how many benign messages may be flagged (the target FPR, e.g. 1%) and let it find
  tau, tau_r and tau_b on validation:
    * each stage gets the threshold that flags at most alpha of the benign messages (a quantile of the benign scores)
    * then we search for the largest alpha whose overall pipeline FPR on validation is <= the target (bisection)
  Every configuration (Stages 1+2, 1+2+3, full, baselines) is then compared at the same FPR, which is fair.

Shape parameters (lambda, W, k_min, tau_o-tau_r, the weights w) are chosen by grid search: the combination with
the highest validation TPR at FPR <= target. The whole grid is kept, which gives a sensitivity plot for free.

Never: use test data in this file (the functions taking test_sessions are evaluate/roc_points, which tune nothing)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .rulebased import Detector
from .metrics import auc_from_points, rates, threshold_for_fpr
from .schema import DIRECT, INDIRECT, Session, Theta

NEVER = dict(tau=math.inf, tau_r=-math.inf, tau_b=math.inf)


# ------------------------------------------------------------------ decision table
def decisions_table(det: Detector, sessions: Sequence[Session]) -> pd.DataFrame:
    """Runs every session and returns a table with one row per event (ready for group-by reporting)"""
    det.warm([ev.text for s in sessions for ev in s.events])
    rows = []
    for s in sessions:
        decs = det.run_session(s)
        for i, (ev, d) in enumerate(zip(s.events, decs)):
            row = dict(sid=s.sid, idx=i, channel=ev.channel, label=ev.label, verdict=d.verdict, stage=d.stage,
                       f=d.f, s0=d.s0, s=d.s, r=d.r, b=d.b, ref_updated=d.reference_updated, text=ev.text[:400])
            for k, v in ev.meta.items():
                row["ev_" + k] = v
            for k, v in s.meta.items():
                row["se_" + k] = v
            rows.append(row)
    return pd.DataFrame(rows)


def _select(df: pd.DataFrame, channels: Optional[Iterable[str]]) -> pd.DataFrame:
    if channels is None:
        return df
    return df[df["channel"].isin(list(channels))]


def table_rates(df: pd.DataFrame, channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    d = _select(df, channels)
    return rates(d["label"].values, d["verdict"].values)


# ------------------------------------------------------------------ calibrate
def calibrate(det: Detector, val: Sequence[Session], alpha: float, delta_o: float = 0.15) -> Theta:
    """Sets tau, tau_r, tau_o, tau_b so that each stage flags at most alpha of the benign validation messages"""
    th, sw = det.theta, det.sw
    nf = det.with_theta(**NEVER)
    df0 = decisions_table(nf, val)
    ben = df0[df0["label"] == 0]

    tau = math.inf
    if sw.stage2 and sw.rule_mode in ("soft", "hard"):
        s_neg = ben["s"] if sw.rule_mode == "soft" else ben.loc[ben["f"] == 0, "s"]
        tau = threshold_for_fpr(s_neg.values, alpha)

    tau_r = -math.inf
    if sw.stage3 and sw.rule_mode in ("soft", "hard"):
        mask = ben["r"].notna()
        if not sw.channel_agnostic:
            mask &= ben["channel"] == INDIRECT
        r_neg = ben.loc[mask, "r"].values
        if len(r_neg):
            tau_r = -threshold_for_fpr(-r_neg, alpha)

    tau_o = (tau_r + delta_o) if math.isfinite(tau_r) else th.tau_o
    tau_b = math.inf
    if sw.stage4 and sw.reference_mode == "single" and sw.rule_mode in ("soft", "hard"):
        d2 = det.with_theta(tau=tau, tau_r=tau_r, tau_o=tau_o, tau_b=math.inf)
        df2 = decisions_table(d2, val)
        b_neg = df2.loc[(df2["label"] == 0) & df2["b"].notna(), "b"].values
        if len(b_neg):
            tau_b = threshold_for_fpr(b_neg, alpha)
    return replace(th, tau=tau, tau_r=tau_r, tau_o=tau_o, tau_b=tau_b)


def _apply(det: Detector, theta: Theta) -> Detector:
    d = Detector(det.emb, det.bank, theta, replace(det.sw), det.name)
    d._f, d._s0 = det._f, det._s0
    return d


def find_alpha(det: Detector, val: Sequence[Session], target_fpr: float, delta_o: float = 0.15,
               channels: Optional[Iterable[str]] = None, n_iter: int = 12,
               lo: float = 1e-6, hi: float = 0.6) -> Dict[str, object]:
    """Finds the largest alpha whose overall pipeline FPR on validation is <= target_fpr (bisection on a log scale)"""
    cache: Dict[float, Tuple[Theta, Dict[str, float]]] = {}

    def at(alpha: float):
        if alpha not in cache:
            th = calibrate(det, val, alpha, delta_o)
            df = decisions_table(_apply(det, th), val)
            cache[alpha] = (th, table_rates(df, channels))
        return cache[alpha]

    th_lo, r_lo = at(lo)
    if r_lo["FPR"] > target_fpr + 1e-12:          # too high even at the strictest setting (e.g. hard reject flagging many benign messages)
        return dict(alpha=lo, theta=th_lo, val=r_lo, feasible=False)
    th_hi, r_hi = at(hi)
    if r_hi["FPR"] <= target_fpr + 1e-12:
        return dict(alpha=hi, theta=th_hi, val=r_hi, feasible=True)
    a_lo, a_hi = math.log(lo), math.log(hi)
    best = (lo, th_lo, r_lo)
    for _ in range(n_iter):
        mid = math.exp((a_lo + a_hi) / 2)
        th, rr = at(mid)
        if rr["FPR"] <= target_fpr + 1e-12:
            best = (mid, th, rr)
            a_lo = math.log(mid)
        else:
            a_hi = math.log(mid)
    return dict(alpha=best[0], theta=best[1], val=best[2], feasible=True)


# ------------------------------------------------------------------ grid search (shape parameters)
def grid_search(det: Detector, val: Sequence[Session], target_fpr: float, grid: Dict[str, Sequence],
                channels: Optional[Iterable[str]] = None, base_delta_o: float = 0.15,
                verbose: bool = True) -> Tuple[Dict[str, object], pd.DataFrame]:
    """grid: e.g. {"lam": [0, .1, .2, .3]} or {"W":[3,5,10], "k_min":[2,3], "delta_o":[.05,.1,.15],
                    "weights":[(.4,.3,.3),(.6,.2,.2)]}
    Returns (the best result, the full grid table on validation)"""
    keys = list(grid.keys())
    rows, best = [], None
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        kw = {k: v for k, v in params.items() if k not in ("delta_o", "weights")}
        if "weights" in params:
            kw["w1"], kw["w2"], kw["w3"] = params["weights"]
        d_o = params.get("delta_o", base_delta_o)
        res = find_alpha(det.with_theta(**kw), val, target_fpr, d_o, channels)
        row = dict(params)
        row["weights"] = str(params["weights"]) if "weights" in params else None
        row.update(alpha=res["alpha"], val_TPR=res["val"]["TPR"], val_FPR=res["val"]["FPR"],
                   feasible=res["feasible"])
        rows.append(row)
        key = (res["feasible"], res["val"]["TPR"] if res["val"]["TPR"] == res["val"]["TPR"] else -1.0,
               -res["val"]["FPR"])
        if best is None or key > best["key"]:
            best = dict(key=key, params=params, kw=kw, delta_o=d_o, result=res)
        if verbose:
            print(f"   grid {params} -> val TPR={res['val']['TPR']:.3f} FPR={res['val']['FPR']:.4f}"
                  f"{'' if res['feasible'] else '  (infeasible)'}")
    return best, pd.DataFrame(rows)


def finalize(det: Detector, val: Sequence[Session], target_fpr: float, grid: Optional[Dict[str, Sequence]] = None,
             channels: Optional[Iterable[str]] = None, base_delta_o: float = 0.15, verbose: bool = True,
             shape: Optional[Dict[str, object]] = None):
    """The standard procedure: (if a grid is given) pick the best shape -> find alpha -> return a detector ready for test
    shape = an already chosen {"kw":..., "delta_o":...} (reused for another target FPR without searching again)
    Returns (detector, info); info holds alpha, delta_o, kw, val_rates and grid_table"""
    grid_df = None
    kw, d_o = {}, base_delta_o
    if shape is not None:
        kw, d_o = dict(shape["kw"]), float(shape["delta_o"])
    elif grid:
        best, grid_df = grid_search(det, val, target_fpr, grid, channels, base_delta_o, verbose)
        kw, d_o = best["kw"], best["delta_o"]
    base = det.with_theta(**kw)
    res = find_alpha(base, val, target_fpr, d_o, channels)
    final = _apply(base, res["theta"])
    final.name = det.name
    return final, dict(alpha=res["alpha"], delta_o=d_o, kw=kw, val=res["val"], feasible=res["feasible"],
                       theta=res["theta"], grid=grid_df)


# ------------------------------------------------------------------ evaluation (tunes nothing)
def evaluate(det: Detector, sessions: Sequence[Session], channels: Optional[Iterable[str]] = None):
    df = decisions_table(det, sessions)
    return table_rates(df, channels), df


def roc_points(det: Detector, val: Sequence[Session], test: Sequence[Session], delta_o: float = 0.15,
               channels: Optional[Iterable[str]] = None, alphas: Optional[Sequence[float]] = None
               ) -> Tuple[float, pd.DataFrame]:
    """Sweeps alpha -> (FPR, TPR) points on test using thresholds calibrated on validation -> computes the AUC"""
    alphas = list(alphas) if alphas is not None else list(np.geomspace(1e-4, 0.9, 14))
    rows = []
    for a in alphas:
        th = calibrate(det, val, a, delta_o)
        rr, _ = evaluate(_apply(det, th), test, channels)
        rows.append(dict(alpha=a, FPR=rr["FPR"], TPR=rr["TPR"]))
    pts = pd.DataFrame(rows)
    return auc_from_points(pts["FPR"], pts["TPR"]), pts


# ------------------------------------------------------------------ sensitivity (a by-product of tuning)
def stage2_sensitivity(det: Detector, val: Sequence[Session], lam_grid: Sequence[float],
                       tau_grid: Sequence[float]) -> pd.DataFrame:
    """Validation TPR/FPR of Stages 1+2 as lambda and tau vary (Stage 2 is stateless, so this is vectorized)"""
    nf = det.with_theta(**NEVER).with_switches(stage3=False, stage4=False)
    df = decisions_table(nf, val)
    rows = []
    for lam in lam_grid:
        s = df["s0"].values + lam * df["f"].values
        for tau in tau_grid:
            rr = rates(df["label"].values, (s > tau).astype(int))
            rows.append(dict(lam=lam, tau=tau, TPR=rr["TPR"], FPR=rr["FPR"]))
    return pd.DataFrame(rows)


def stage4_sensitivity(det: Detector, val: Sequence[Session], theta: Theta, W_grid: Sequence[int],
                       taub_grid: Sequence[float], channels: Optional[Iterable[str]] = (INDIRECT,)) -> pd.DataFrame:
    """Validation TPR/FPR of the full pipeline as W and tau_b vary (the other thresholds stay as in theta)"""
    rows = []
    for W in W_grid:
        for tb in taub_grid:
            d = _apply(det, replace(theta, W=W, tau_b=tb))
            rr, _ = evaluate(d, val, channels)
            rows.append(dict(W=W, tau_b=tb, TPR=rr["TPR"], FPR=rr["FPR"]))
    return pd.DataFrame(rows)
