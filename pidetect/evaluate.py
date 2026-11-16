"""evaluate.py : runs the system or a baseline over sessions and summarizes event-level and session-level metrics

Session level (the main metrics of H1 and H3):
  session detection rate = share of attack sessions in which at least one label-1 event is blocked
  detection delay        = number of events from the start of the attack (the first phase=attack event) to the first block of a label-1 event
  censored delay         = the same, but a missed session is charged the whole remaining horizon, so a detector that
                           catches only the easy sessions cannot look fast (the quickest-detection convention)
  session false-positive    = share of benign sessions with at least one block
  session Precision / F1 = attack sessions detected vs benign sessions falsely blocked, as the multi-turn
                           detection literature reports it (with the raw TP / FP / TN / FN counts)
Event level: TPR, FPR, Precision, F1, ROC-AUC, ASR (= attack events not blocked), share of Forward-Untrusted
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import auc_score, rates, threshold_for_fpr
from .pipeline import ProposedDetector
from .progress import track
from .schema import BLOCK, FORWARD_UNTRUSTED, Session


def records_frame(det: ProposedDetector, sessions: Sequence[Session], seed: int = 0) -> pd.DataFrame:
    rows = []
    for s in track(sessions, desc=f"scoring sessions ({det.name})"):
        for i, (e, r) in enumerate(zip(s.events, det.run_session(s))):
            row = dict(method=det.name, seed=seed, sid=s.sid, idx=i, channel=e.channel, label=e.label,
                       source=e.source, turn=e.turn, action=r.action, blocked=int(r.action == BLOCK),
                       untrusted=int(r.action == FORWARD_UNTRUSTED), score=r.score, p=r.p, stage=r.stage,
                       new_task=int(r.new_task), ref_updated=int(r.reference_updated), reasons="; ".join(r.reasons),
                       phase=e.meta.get("phase", ""), role=e.meta.get("role", ""), text=e.text[:300])
            row.update({f"x_{k}": v for k, v in r.x.items()})
            row.update({f"ev_{k}": v for k, v in e.meta.items() if k not in ("phase", "role")})
            row.update({f"se_{k}": v for k, v in s.meta.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def baseline_frame(bl, sessions: Sequence[Session], thr: float, seed: int = 0, name: Optional[str] = None
                   ) -> pd.DataFrame:
    """Baselines are stateless: score each event and block when score > thr

    A baseline that scores differently per channel (Prompt-Guard, see pidetect/baselines.py) is given the channel
    of every event; the others ignore it.  Calibration goes through this same function, so the threshold is always
    computed from the same scores that are later applied to the test events.
    """
    texts = [e.text for s in sessions for e in s.events]
    channels = [e.channel for s in sessions for e in s.events]
    if not texts:
        scores = np.array([])
    elif hasattr(bl, "score_channels"):
        scores = bl.score_channels(texts, channels)
    else:
        scores = bl.score(texts)
    if getattr(bl, "session_norm", False):     # judge each event against the events before it in its own session
        from .baselines import SessionNormalized
        pos, parts = 0, []
        for sess in sessions:
            n = len(sess.events)
            parts.append(SessionNormalized.transform(scores[pos:pos + n]))
            pos += n
        scores = np.concatenate(parts) if parts else scores
    rows, k = [], 0
    for s in sessions:
        for i, e in enumerate(s.events):
            sc = float(scores[k])
            k += 1
            rows.append(dict(method=name or bl.name, seed=seed, sid=s.sid, idx=i, channel=e.channel, label=e.label,
                             source=e.source, turn=e.turn, action=BLOCK if sc > thr else 0, blocked=int(sc > thr),
                             untrusted=0, score=sc, p=sc, stage="baseline", phase=e.meta.get("phase", ""),
                             role=e.meta.get("role", ""), **{f"se_{kk}": v for kk, v in s.meta.items()}))
    return pd.DataFrame(rows)


def baseline_threshold(bl, val_sessions: Sequence[Session], target: float, level: str = "event",
                       channels: Optional[Iterable[str]] = None) -> float:
    """Baseline threshold from val: level=event -> event-level FPR; level=session -> session-level false positive"""
    if getattr(bl, "fixed_point", False):          # B1 (regex) has no threshold: a pattern match is a block
        return 0.5
    df = baseline_frame(bl, val_sessions, np.inf)
    if channels is not None:
        df = df[df["channel"].isin(list(channels))]
    if level == "session":
        sess = df.groupby("sid").agg(score=("score", "max"), y=("label", "max"))
        return threshold_for_fpr(sess.loc[sess["y"] == 0, "score"].values, target)
    return threshold_for_fpr(df.loc[df["label"] == 0, "score"].values, target)


def session_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, seed, session).  Same definitions as the per-session loop kept below as
    _session_table_slow (which the tests compare it with), computed with grouped operations: the loop took over a
    minute on the H1 events and was run several times."""
    keys = ["method", "seed", "sid"]
    d = df.reset_index(drop=True)
    lab1 = d["label"] == 1
    blk1 = d["blocked"] == 1
    inj = lab1 & blk1
    att_phase = d["phase"] == "attack"
    early = (d["label"] == 0) & blk1 & att_phase
    idx = d["idx"].astype(float)
    t = pd.DataFrame({
        "method": d["method"], "seed": d["seed"], "sid": d["sid"],
        "_lab1": lab1, "_inj": inj, "_blk": blk1, "_early": early,
        "_idx_inj": idx.where(inj), "_idx_att": idx.where(att_phase), "_idx_lab1": idx.where(lab1),
    })
    g = t.groupby(keys, sort=False)
    agg = g.agg(is_attack=("_lab1", "max"), n_inj=("_inj", "sum"), any_blk=("_blk", "max"),
                early_any=("_early", "max"), first_inj=("_idx_inj", "min"), first_att=("_idx_att", "min"),
                first_lab1=("_idx_lab1", "min"), n_events=("_lab1", "size"))
    is_attack = agg["is_attack"].astype(bool)
    detected = (is_attack & (agg["n_inj"] > 0)).astype(int)
    start = agg["first_att"].where(agg["first_att"].notna(), agg["first_lab1"].where(is_attack, -1.0))
    start = start.fillna(-1.0)
    delay = (agg["first_inj"] - start).where(detected == 1)
    horizon = (agg["n_events"] - 1 - start).where(is_attack & (start >= 0))
    delay_censored = delay.where(detected == 1, horizon).where(is_attack)
    out = pd.DataFrame({
        "method": agg.index.get_level_values("method"), "seed": agg.index.get_level_values("seed"),
        "sid": agg.index.get_level_values("sid"),
        "is_attack": is_attack.astype(int).values, "detected": detected.values,
        "delay_censored": delay_censored.values, "horizon": horizon.values,
        "any_block": agg["any_blk"].astype(int).values,
        "false_positive": ((~is_attack) & agg["any_blk"].astype(bool)).astype(int).values,
        "early_block": (is_attack & agg["early_any"].astype(bool)).astype(int).values,
        "delay": delay.values, "n_events": agg["n_events"].values,
    })
    se_cols = [c for c in d.columns if c.startswith("se_")]
    if se_cols:
        first_rows = d.assign(_i=idx).sort_values("_i", kind="stable").drop_duplicates(keys)
        first_rows = first_rows.set_index(keys)[se_cols]
        meta = first_rows.reindex(agg.index)
        for c in se_cols:
            out[c] = meta[c].values
    return out


def _session_table_slow(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed, sid), g in df.groupby(["method", "seed", "sid"], sort=False):
        g = g.sort_values("idx")
        is_attack = int(g["label"].max() == 1)
        inj_blocked = g[(g["label"] == 1) & (g["blocked"] == 1)]
        detected = int(len(inj_blocked) > 0) if is_attack else 0
        start = g.loc[g["phase"] == "attack", "idx"]
        start = int(start.min()) if len(start) else (int(g.loc[g["label"] == 1, "idx"].min()) if is_attack else -1)
        delay = float(inj_blocked["idx"].min() - start) if detected else np.nan
        # censored delay: a missed session is charged the whole remaining horizon, so that a detector which
        # catches only the easy sessions cannot look fast (the convention of the quickest-detection literature)
        horizon = float(len(g) - 1 - start) if is_attack and start >= 0 else np.nan
        delay_censored = (delay if detected else horizon) if is_attack else np.nan
        meta = {c: g[c].iloc[0] for c in g.columns if c.startswith("se_")}
        rows.append(dict(method=method, seed=seed, sid=sid, is_attack=is_attack, detected=detected,
                         delay_censored=delay_censored, horizon=horizon,
                         any_block=int(g["blocked"].max() == 1), false_positive=int((not is_attack) and g["blocked"].max() == 1),
                         early_block=int(is_attack and len(g[(g["label"] == 0) & (g["blocked"] == 1) &
                                                             (g["phase"] == "attack")]) > 0),
                         delay=delay, n_events=len(g), **meta))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: Sequence[str] = ("method",), channels: Optional[Iterable[str]] = None
              ) -> pd.DataFrame:
    """Per-group summary: event level and session level"""
    d = df if channels is None else df[df["channel"].isin(list(channels))]
    st = session_table(d)
    out = []
    for key, g in d.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rr = rates(g["label"].values, g["blocked"].values)
        sg = st
        for k, v in zip(by, key):
            sg = sg[sg[k] == v]
        att, ben = sg[sg["is_attack"] == 1], sg[sg["is_attack"] == 0]
        inj = g[g["label"] == 1]
        row = dict(zip(by, key))
        row.update(TPR=rr["TPR"], FPR=rr["FPR"], Precision=rr["Precision"], F1=rr["F1"],
                   AUC=auc_score(g["label"].values, g["score"].values),
                   ASR=float(1 - inj["blocked"].mean()) if len(inj) else np.nan,
                   untrusted_share_attacks=float(inj["untrusted"].mean()) if len(inj) else np.nan,
                   session_detection=float(att["detected"].mean()) if len(att) else np.nan,
                   session_fpr=float(ben["any_block"].mean()) if len(ben) else np.nan,
                   delay_median=float(att["delay"].median()) if att["delay"].notna().any() else np.nan,
                   delay_mean=float(att["delay"].mean()) if att["delay"].notna().any() else np.nan,
                   delay_censored_mean=float(att["delay_censored"].mean())
                   if "delay_censored" in att and att["delay_censored"].notna().any() else np.nan,
                   n_events=len(g), n_attack_sessions=len(att), n_benign_sessions=len(ben))
        # session level, the same way the multi-turn detection literature reports it.
        # An attack session counts as a true positive only when an injected event is blocked (detected);
        # a benign session counts as a false positive on any block at all.
        s_tp = int(att["detected"].sum())
        s_fn = int(len(att) - s_tp)
        s_fp = int(ben["any_block"].sum())
        s_tn = int(len(ben) - s_fp)
        s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else float("nan")
        s_rec = s_tp / len(att) if len(att) else float("nan")
        row.update(session_precision=s_prec,
                   session_F1=(2 * s_prec * s_rec / (s_prec + s_rec)) if (s_tp > 0) else (0.0 if len(att) else np.nan),
                   session_TP=s_tp, session_FP=s_fp, session_TN=s_tn, session_FN=s_fn)
        out.append(row)
    return pd.DataFrame(out)


def session_ranking(df: pd.DataFrame, fprs: Sequence[float] = (0.05, 0.075, 0.10)) -> pd.DataFrame:
    """Session-level ranking, free of the calibration that every method fails to transfer exactly.

    A session is scored by its highest event score.  For each method we report the session-level ROC-AUC and the
    detection rate at thresholds that give each method the *same achieved* false-positive rate on benign sessions,
    rather than the same target.  A threshold set on validation lands somewhere else on test, by a different
    amount for each method, so comparing at a common target compares calibration as much as detection; comparing
    at a common achieved rate compares only what the scores rank.
    """
    rows = []
    for (method, seed), g in df.groupby(["method", "seed"], sort=False):
        sess = g.groupby("sid").agg(mx=("score", "max"), y=("label", "max"))
        att = sess.loc[sess["y"] == 1, "mx"].values
        ben = sess.loc[sess["y"] == 0, "mx"].values
        row = dict(method=method, seed=seed, session_AUC=auc_score(sess["y"].values, sess["mx"].values),
                   n_attack_sessions=len(att), n_benign_sessions=len(ben))
        for f in fprs:
            thr = float(np.quantile(ben, 1 - f)) if len(ben) else float("inf")
            row[f"det_at_fpr_{f:g}"] = float((att > thr).mean()) if len(att) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def event_threshold_metrics(df: pd.DataFrame, thr: float, channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Applies a score (e.g. the LR probability) with the threshold from val -> event-level TPR/FPR"""
    d = df if channels is None else df[df["channel"].isin(list(channels))]
    rr = rates(d["label"].values, (d["score"].values > thr).astype(int))
    rr["AUC"] = auc_score(d["label"].values, d["score"].values)
    return rr
