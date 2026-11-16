"""fit.py : fits and calibrates the whole system (on dev and val only, never on test)

Order (fit_system):
  1) Stage 2 : train the heads on dev                                  (stage2.fit_heads)
  2) Stage 2 : kappa per length bin, the fusion weights, tau_f on the first half of val (val_a)
  3) session constants : rbar, tau_o, rho, tau_a, tau_same on val_a
  4) Decision Module : fit the LR on the second half of val (val_b) in two passes
        pass 0 : no LR yet (everything is forwarded except the fast path) -> collect features -> fit the LR
        pass 1 : rerun with that LR (so the state, e.g. u and R, matches the real policy) -> fit the LR again
     the thresholds t_blk and t_rev are quantiles of the out-of-fold probabilities (GroupKFold by session) of benign events
     (out-of-fold because probabilities on the training data are overconfident -- Section III-F of the paper)

A variant of the system reuses the same Stage 2 and the same constants, then refits the LR on val_b the same way,
so that the comparison is fair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .progress import track
from .pipeline import PipeConfig, ProposedDetector
from .schema import BLOCK, DIRECT, INDIRECT, Session
from .stage2 import Stage2, Stage2Config


@dataclass
class FitConfig:
    nu: float = 0.01               # target event-level FPR for Block
    nu_r: float = 0.05             # target FPR for Forward-Untrusted
    tau_o_quantile: float = 0.10   # tau_o = this quantile of r over benign indirect events
    rho_quantile: float = 0.90     # rho   = this quantile of s over benign direct events
    tau_a_quantile: float = 0.95   # tau_a = this quantile of s over benign direct events (higher = fewer tasks lose trust)
    C_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    n_pass: int = 2
    cv_folds: int = 5


# ------------------------------------------------------------------ Stage 2
def fit_stage2(emb, bank, head_data, cfg: Stage2Config, val_a: Sequence[Session], seed: int = 0,
               heads_from: Optional[Stage2] = None) -> Tuple[Stage2, Dict[str, float]]:
    s2 = Stage2(emb, bank, cfg)
    if heads_from is not None:
        s2.heads = dict(heads_from.heads)
    else:
        s2.fit_heads(head_data, seed)
    evs = [e for s in val_a for e in s.events]
    s2.calibrate_kappa([e.text for e in evs if e.channel == INDIRECT and e.label == 0])
    info = s2.fit_fusion([e.text for e in evs], [e.channel for e in evs], [e.label for e in evs])
    s2.calibrate_fast_path([e.text for e in evs if e.label == 0], [e.channel for e in evs if e.label == 0])
    info.update(tau_f=s2.tau_f, n_fit_events=len(evs))
    return s2, info


# ------------------------------------------------------------------ session constants
def calibrate_tau_same(det: ProposedDetector, sessions: Sequence[Session]) -> Tuple[float, Dict[str, float]]:
    """tau_same from labeled pairs: follow-ups (should be >= tau_same) and new tasks (should be < tau_same).
    Uses the true last task from the meta role, not the detector's state -> picks the best balanced accuracy."""
    fu, new = [], []
    for s in sessions:
        R = None
        for e in s.events:
            if e.channel != DIRECT:
                continue
            role = e.meta.get("role")
            vec = det.s2.process(e.text, e.channel).raw.e
            if role in ("instruction",) and R is None:
                R = vec
                continue
            if R is None:
                continue
            c = float(np.dot(vec, R))
            if role == "followup":
                fu.append(c)
            elif role in ("switch", "instruction"):
                new.append(c)
                R = vec
    if not fu or not new:
        return det.cfg.tau_same, dict(n_followup=len(fu), n_new=len(new))
    cands = np.unique(np.concatenate([fu, new]))
    best, best_ba = det.cfg.tau_same, -1.0
    fu_a, new_a = np.array(fu), np.array(new)
    for t in cands:
        ba = 0.5 * ((fu_a >= t).mean() + (new_a < t).mean())
        if ba > best_ba:
            best, best_ba = float(t), ba
    return best, dict(n_followup=len(fu), n_new=len(new), balanced_acc=float(best_ba))


def calibrate_constants(det: ProposedDetector, val_a: Sequence[Session], fc: FitConfig) -> Dict[str, float]:
    """rbar, tau_o, rho, tau_a, tau_same from val_a (run with no LR yet)"""
    det.model = None
    tau_same, ts_info = calibrate_tau_same(det, val_a)
    det.cfg = replace(det.cfg, tau_same=tau_same)
    s_dir, r_ind = [], []
    for s in val_a:
        recs = det.run_session(s)
        for e, r in zip(s.events, recs):
            if e.label != 0 or r.stage in ("fast", "hard"):
                continue
            if e.channel == DIRECT:
                s_dir.append(r.s)
            elif r.x["z"] == 0.0 or det.cfg.reference_mode == "accumulated":
                r_ind.append(r.x["r"])
    upd = {}
    if r_ind:
        upd["rbar"] = float(np.mean(r_ind))
        upd["tau_o"] = float(np.quantile(r_ind, fc.tau_o_quantile))
    if s_dir:
        upd["rho"] = float(np.quantile(s_dir, fc.rho_quantile))
        upd["tau_a"] = float(np.quantile(s_dir, fc.tau_a_quantile))
    det.cfg = replace(det.cfg, **upd)
    upd["tau_same"] = tau_same
    upd.update({f"tau_same_{k}": v for k, v in ts_info.items()})
    return upd


# ------------------------------------------------------------------ Decision Module (LR)
def _collect(det: ProposedDetector, sessions: Sequence[Session]):
    X, y, groups, ch = [], [], [], []
    for gi, s in enumerate(sessions):
        for e, r in zip(s.events, det.run_session(s)):
            if r.stage in ("fast", "hard"):
                continue
            X.append(det.vector(r.x))
            y.append(e.label)
            groups.append(gi)
            ch.append(e.channel)
    return np.array(X), np.array(y), np.array(groups), np.array(ch)


def fit_decision(det: ProposedDetector, sessions: Sequence[Session], fc: FitConfig, seed: int = 0) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    det.model = None
    info: Dict[str, float] = {}
    for p in range(fc.n_pass):
        X, y, groups, ch = _collect(det, sessions)
        if len(set(y.tolist())) < 2:
            raise ValueError("fit_decision: the val data must contain both attack and benign events")
        n_groups = len(set(groups.tolist()))
        cv = GroupKFold(n_splits=max(2, min(fc.cv_folds, n_groups)))
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(class_weight="balanced", max_iter=5000))])
        gs = GridSearchCV(pipe, {"lr__C": list(fc.C_grid)}, cv=cv, scoring="neg_log_loss")
        gs.fit(X, y, groups=groups)
        best = gs.best_estimator_
        oof = cross_val_predict(best, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
        neg = oof[y == 0]
        t_blk = float(np.quantile(neg, 1 - fc.nu, method="higher"))
        t_rev = float(np.quantile(neg, 1 - fc.nu_r, method="higher"))
        t_rev = min(t_rev, t_blk)
        det.model = best.fit(X, y)
        det.cfg = replace(det.cfg, t_blk=t_blk, t_rev=t_rev, nu=fc.nu, nu_r=fc.nu_r)
        info = dict(n_events=int(len(y)), n_pos=int(y.sum()), C=float(best.named_steps["lr"].C), t_blk=t_blk,
                    t_rev=t_rev, oof_pass=p)
    # per-channel medians over benign events -> used to compute the reasons psi_j
    X, y, _, ch = _collect(det, sessions)
    for c in (DIRECT, INDIRECT):
        m = (y == 0) & (ch == c)
        if m.any():
            det.medians[c] = np.median(X[m], axis=0)
    return info


def fit_system(emb, bank, head_data, val_a: Sequence[Session], val_b: Sequence[Session], s2cfg: Stage2Config,
               pcfg: PipeConfig, fc: FitConfig, seed: int = 0, name: str = "full",
               heads_from: Optional[Stage2] = None, ext=None) -> Tuple[ProposedDetector, Dict[str, float]]:
    s2, s2info = fit_stage2(emb, bank, head_data, s2cfg, val_a, seed, heads_from)
    # ext (a public classifier used as the content score) is in place before the constants and the LR are fitted,
    # so the system is calibrated on the scores it will actually see
    det = ProposedDetector(s2, pcfg, name, ext=ext)
    cinfo = calibrate_constants(det, val_a, fc)
    dinfo = fit_decision(det, val_b, fc, seed)
    info = {**{f"stage2_{k}": v for k, v in s2info.items()}, **cinfo, **{f"lr_{k}": v for k, v in dinfo.items()}}
    return det, info


def refit_variant(base: ProposedDetector, name: str, val_b: Sequence[Session], fc: FitConfig, seed: int = 0,
                  stage2: Optional[Stage2] = None, **cfg_kw) -> ProposedDetector:
    """Builds a variant of an already fitted system (same constants) and refits its LR the same way"""
    d = base.variant(name, stage2=stage2, **cfg_kw)
    fit_decision(d, val_b, fc, seed)
    return d


# ------------------------------------------------------------------ session-level threshold (H1: compare at equal FPR)
def session_false_positive(det: ProposedDetector, benign: Sequence[Session]) -> float:
    if not benign:
        return float("nan")
    return float(np.mean([any(r.action == BLOCK for r in det.run_session(s)) for s in benign]))


def calibrate_session_threshold(det: ProposedDetector, benign_val: Sequence[Session], target: float,
                                n_iter: int = 14) -> Dict[str, float]:
    """Finds the smallest t_blk whose share of benign sessions with at least one block is <= target (bisection).
    Every step reruns the sessions, because blocking changes the state (e.g. u and R)."""
    lo, hi = det.cfg.t_rev, 1.0
    det.cfg = replace(det.cfg, t_blk=hi)
    fpr_hi = session_false_positive(det, benign_val)
    if fpr_hi > target:                               # still too high even with no LR block at all (the fast path fires)
        return dict(t_blk=hi, val_session_fpr=fpr_hi, feasible=False)
    best = hi
    for _ in track(range(n_iter), total=n_iter, desc=f"threshold bisection ({det.name})", first_after=20.0):
        mid = (lo + hi) / 2
        det.cfg = replace(det.cfg, t_blk=mid)
        if session_false_positive(det, benign_val) <= target:
            best, hi = mid, mid
        else:
            lo = mid
    det.cfg = replace(det.cfg, t_blk=best)
    return dict(t_blk=best, val_session_fpr=session_false_positive(det, benign_val), feasible=True)
