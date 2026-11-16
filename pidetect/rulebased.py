"""rulebased.py : the rule-based baseline = the earlier version of the system, which thresholds stage by stage
(the paper keeps it only as a baseline -- the proposed system lives in pipeline.py)


For each event (message x, channel c), in order:
  1) e <- Embed(x),  f <- RuleFlag(x),  s <- max_j cos(e, a_j) + λ f
  2) if s > tau                        -> Suspicious (Semantic)
  3) if c = direct                     -> R <- e, H <- empty, Safe   (only a Safe direct message may write R)
  4) if there is no R yet              -> Safe  (Stages 3-4 are skipped)
  5) r <- cos(e, R); if r < tau_r      -> Suspicious (Relevance)
  6) H <- the last W entries of (r, f);  b <- BehaviorScore(H)
     if b > tau_b                      -> Suspicious (Behavior)
  7) otherwise                         -> Safe

Experiment switches (see Switches in schema.py)
  rule_mode="hard"            : the hard-rejection baseline of the secondary experiment
  reference_mode="accumulated": the earlier design (a profile averaged over the session), used in H3
  channel_agnostic=True       : Stage 3 also runs on direct messages, used in H3a
  use_o / use_fbar / use_delta: turn off one Stage 4 sub-signal at a time (H3b)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .attackbank import AttackBank
from .rules import explain, rule_flag
from .schema import DIRECT, INDIRECT, Decision, Session, Switches, Theta


@dataclass
class SessionState:
    """Per-session state, O(d + W) memory"""
    R: Optional[np.ndarray] = None                       # the single reference
    H: List[Tuple[float, float]] = field(default_factory=list)   # [(r_i, f_i)], at most the last W
    P_sum: Optional[np.ndarray] = None                   # sum of the Safe event embeddings (accumulated mode)


def state_bytes(dim: int, W: int) -> int:
    """Approximate per-session state size: R (d float32) + H (W float32 pairs)"""
    return 4 * dim + 8 * W


def behavior_score(H: Sequence[Tuple[float, float]], th: Theta, sw: Switches | None = None) -> Optional[float]:
    """BehaviorScore(H, Theta) of Algorithm 2; returns None until there are k_min events"""
    sw = sw or Switches()
    k = len(H)
    if k < th.k_min or k == 0:
        return None
    rs = [x[0] for x in H]
    fs = [x[1] for x in H]
    o = sum(1 for r in rs if th.tau_r <= r < th.tau_o) / k
    fbar = sum(fs) / k
    delta = max(0.0, rs[0] - rs[-1])
    return (th.w1 * o * (1.0 if sw.use_o else 0.0)
            + th.w2 * fbar * (1.0 if sw.use_fbar else 0.0)
            + th.w3 * delta * (1.0 if sw.use_delta else 0.0))


class Detector:
    def __init__(self, embedder, bank: AttackBank, theta: Theta | None = None,
                 switches: Switches | None = None, name: str = "pipeline"):
        self.emb = embedder
        self.bank = bank
        self.theta = theta or Theta()
        self.sw = switches or Switches()
        self.name = name
        self._f: Dict[str, float] = {}
        self._s0: Dict[str, float] = {}

    # ---------- features (cached, to keep the grid search fast) ----------
    def warm(self, texts: Sequence[str]) -> None:
        """Precomputes the embedding / rule flag / s0 of every text in one batch"""
        uniq = list(dict.fromkeys(texts))
        E = self.emb.encode(uniq)
        S0 = self.bank.max_cos_batch(E)
        for t, s0 in zip(uniq, S0):
            self._s0[t] = float(s0)
            if t not in self._f:
                self._f[t] = rule_flag(t)

    def _features(self, text: str, fresh: bool = False) -> Tuple[np.ndarray, float, float]:
        if fresh:  # no cache at all (used when measuring real latency)
            e = self.emb.base.encode([text])[0]
            return e, (rule_flag(text) if self.sw.stage1 else 0.0), (
                self.bank.max_cos(e) if self.sw.stage2 else 0.0)
        e = self.emb.encode_one(text)
        if text not in self._f:
            self._f[text] = rule_flag(text)
        if text not in self._s0:
            self._s0[text] = self.bank.max_cos(e)
        f = self._f[text] if self.sw.stage1 else 0.0
        s0 = self._s0[text] if self.sw.stage2 else 0.0
        return e, f, s0

    # ---------- reference ----------
    def _reference(self, st: SessionState) -> Optional[np.ndarray]:
        if self.sw.reference_mode == "accumulated":
            if st.P_sum is None:
                return None
            n = float(np.linalg.norm(st.P_sum))
            return st.P_sum / n if n > 0 else None
        return st.R

    def new_state(self) -> SessionState:
        return SessionState()

    # ---------- Algorithm 1 ----------
    def process(self, text: str, channel: str, st: SessionState, fresh: bool = False) -> Decision:
        th, sw = self.theta, self.sw
        e, f, s0 = self._features(text, fresh=fresh)
        s = s0 + th.lam * f

        def out(verdict: int, stage: str, why: str, r=None, b=None, upd=False) -> Decision:
            return Decision(verdict, stage, channel, f, s0, s, r, b, upd, why)

        # --- the pure-rule variant ---
        if sw.rule_mode == "hard" and f > 0:
            return out(1, "rule", f"hard reject: rule pattern matched ({', '.join(explain(text))})")
        if sw.rule_mode == "rule_only":
            if f >= 1.0:
                return out(1, "rule", f"rule flag = 1 ({', '.join(explain(text))})")
            return out(0, "none", "rule flag < 1")

        # --- Stage 2 ---
        if sw.stage2 and s > th.tau:
            return out(1, "semantic", f"combined score {s:.3f} > tau {th.tau:.3f} (rule flag {f})")

        # --- direct channel ---
        if channel == DIRECT:
            ref = self._reference(st)
            if sw.channel_agnostic and sw.stage3 and ref is not None:
                r = float(np.dot(e, ref))
                if r < th.tau_r:
                    return out(1, "relevance", f"direct msg relevance {r:.3f} < tau_r {th.tau_r:.3f}", r=r)
            self._accept_direct(e, st)
            return out(0, "none", "safe direct instruction -> reference updated", upd=True)

        # --- indirect channel ---
        ref = self._reference(st)
        if ref is None or not (sw.stage3 or sw.stage4):
            self._accept_indirect(e, st)
            return out(0, "none", "no reference yet" if ref is None else "stages 3-4 off")
        r = float(np.dot(e, ref))
        if sw.stage3 and r < th.tau_r:
            return out(1, "relevance", f"relevance {r:.3f} < tau_r {th.tau_r:.3f}", r=r)

        b = None
        if sw.reference_mode == "single":
            st.H.append((r, f))
            if len(st.H) > th.W:
                st.H = st.H[-th.W:]
            if sw.stage4:
                b = behavior_score(st.H, th, sw)
                if b is not None and b > th.tau_b:
                    return out(1, "behavior", f"behavior score {b:.3f} > tau_b {th.tau_b:.3f}", r=r, b=b)
        self._accept_indirect(e, st)
        return out(0, "none", "passed all stages", r=r, b=b)

    # ---------- state update ----------
    def _accept_direct(self, e: np.ndarray, st: SessionState) -> None:
        if self.sw.reference_mode == "accumulated":
            st.P_sum = e.copy() if st.P_sum is None else st.P_sum + e
        else:
            st.R = e
            st.H = []

    def _accept_indirect(self, e: np.ndarray, st: SessionState) -> None:
        if self.sw.reference_mode == "accumulated":      # indirect text can drag the accumulated profile (the weakness we measure)
            st.P_sum = e.copy() if st.P_sum is None else st.P_sum + e

    # ---------- run a whole session ----------
    def run_session(self, sess: Session, fresh: bool = False) -> List[Decision]:
        st = self.new_state()
        return [self.process(ev.text, ev.channel, st, fresh=fresh) for ev in sess.events]

    def run_sessions(self, sessions: Sequence[Session]) -> List[List[Decision]]:
        self.warm([ev.text for s in sessions for ev in s.events])
        return [self.run_session(s) for s in sessions]

    def with_theta(self, **kw) -> "Detector":
        """A copy of the detector with a different Theta (the feature cache is shared, since the bank is the same)"""
        d = Detector(self.emb, self.bank, replace(self.theta, **kw), replace(self.sw), self.name)
        d._f, d._s0 = self._f, self._s0
        return d

    def with_switches(self, name: str | None = None, **kw) -> "Detector":
        d = Detector(self.emb, self.bank, replace(self.theta), replace(self.sw, **kw), name or self.name)
        d._f, d._s0 = self._f, self._s0
        return d


def never_fire(theta: Theta) -> Theta:
    """A Theta under which no stage ever says Suspicious -- used to collect raw scores during calibration"""
    return replace(theta, tau=math.inf, tau_r=-math.inf, tau_b=math.inf)
