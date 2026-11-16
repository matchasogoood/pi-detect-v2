"""pipeline.py : the proposed system -- the Detection Module of Algorithm 1 (Section III-F)

For each event (message x, channel c, retrieval turn t, source d):
  1) Stage 1-2  : s, fast path  (stage2.py)
  2) Stage 4 (direct) : a <- max(0, a + s - rho)  (CUSUM, updated even when the event is blocked)
  3) fast path -> Block immediately (also counted in B[d]; R and H are unchanged)
  4) build the 13 features x = (s, a, k, u, r, c, z, n_imp, n_ai, o, fbar, delta, h)   (Table II)
  5) p = LR(x)  -> Block / Forward-Untrusted / Forward via t_blk and t_rev
  6) indirect with no trusted reference (z = 1) -> at least Forward-Untrusted
  7) update the state with the authority rule (which is not learned):
       direct, not blocked, a new task -> R <- e, H <- empty, trust <- [Forward and s < tau_a]
       indirect, not blocked           -> H <- the last W entries of H || (t, r, f)
       blocked                         -> B[d] += 1

Experiment switches (PipeConfig):
  features            : the feature set the LR sees (memoryless / ref-only / no Stage 4 ...)     -> H1, H2, H3
  relevance_unit      : "window" (omega*, as in the paper) | "chunk" (the whole chunk)           -> H2
  reference_mode      : "single" (ours) | "accumulated" (a running profile = the earlier system)  -> H3
  channel_agnostic    : True = no authority principle (any message may set the reference, Stage 3 also runs on direct)
  use_trust           : False = no tau_a (a new reference is always trusted)                      -> H3b anchor case
  drop_low_trust      : True = drop the reference entirely instead of keeping it as low-trust
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lexical import lexical_counts
from .schema import ACTION_NAMES, BLOCK, DIRECT, FORWARD, FORWARD_UNTRUSTED, INDIRECT, Event, Session
from .stage2 import Stage2, Stage2Out

FEATURES: Tuple[str, ...] = ("s", "a", "k", "u", "r", "c", "z", "n_imp", "n_ai", "o", "fbar", "delta", "h")
CONTENT_FEATURES: Tuple[str, ...] = ("s", "c", "n_imp", "n_ai")                 # memoryless (reads no state)
REF_ONLY_FEATURES: Tuple[str, ...] = CONTENT_FEATURES + ("r", "z")               # a reference but no history
# H3d and the anchor case showed no measurable effect of u, k (and of tau_a), so the simplified system drops
# them; whether that costs anything is decided on validation, not on the test split.
FEATURES_NO_COUNTERS: Tuple[str, ...] = tuple(f for f in FEATURES if f not in ("u", "k"))
STAGE4_IND: Tuple[str, ...] = ("o", "fbar", "delta", "h")
REASON_FEATURES = ("s", "a", "k", "u", "r", "n_imp", "n_ai", "o", "fbar", "delta")   # used for the reasons (psi_j)
FEATURE_TEXT = {"s": "fused score", "a": "session suspicion (CUSUM)", "k": "task resets", "u": "earlier blocks",
                "r": "relevance to task", "n_imp": "imperative count", "n_ai": "AI-address count",
                "o": "share of off-task chunks", "fbar": "rule flags in window", "delta": "relevance decline"}

# ----------------------------------------------------------------- follow-up vs new task (IsTask)
_STOP = set("""a an the and or but if then so to of in on at by for with about as is are was were be been being do does
did doing have has had i me my we our you your it its this that these those there here what which who whom whose when
where why how can could would should will shall may might must please pls thanks thank ok okay yes no not just also
too very more most less some any all again still really much many one two first second last next previous same other
another give tell show me us explain say go continue elaborate detail details example examples bit little part above
below point points that's it's i'm don't can't won't""".split())
_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def has_content(text: str) -> bool:
    """Has at least one content word (not a stopword and not a follow-up word)"""
    return any(w.lower() not in _STOP and len(w) > 1 for w in _WORD.findall(text))


@dataclass
class PipeConfig:
    W: int = 5                  # Stage 4 window size (both H on the indirect side and D on the direct side)
    k_min: int = 3              # how many events are needed before the indirect Stage 4 signals count
    tau_o: float = 0.2          # r < tau_o means off-task (counted in o)
    rho: float = 0.5            # reference level of the CUSUM
    tau_same: float = 0.5       # cos(e, R) >= tau_same means a follow-up (R is kept)
    tau_a: float = 0.8          # anchor threshold: a new reference is trusted when s < tau_a
    t_rev: float = 0.5          # p > t_rev = Forward-Untrusted
    t_blk: float = 0.9          # p > t_blk = Block
    rbar: float = 0.3           # the r used when there is no relevance (the mean over benign indirect text)
    nu: float = 0.01            # target FPR for Block (sets t_blk)
    nu_r: float = 0.05          # target FPR for Forward-Untrusted (sets t_rev)
    features: Tuple[str, ...] = FEATURES
    relevance_unit: str = "window"
    reference_mode: str = "single"
    channel_agnostic: bool = False
    untrusted_floor: bool = True
    use_trust: bool = True
    drop_low_trust: bool = False
    # A forwarded direct follow-up may refine the reference: R <- normalize(R + gamma * e).  The authority
    # principle is unchanged, because indirect content still cannot move R and only a direct event that the
    # pipeline forwarded has any effect; gamma = 0 is the system as evaluated so far.  It exists because the
    # single reference costs false positives when the user narrows the topic (H3a): a follow-up that drills into one
    # aspect lowers the relevance of what follows while R still points at the broader original question.
    gamma_follow: float = 0.0
    # Features listed here are also given to the decision model multiplied by the channel indicator c, i.e. the
    # model fits a separate slope per channel for them (the per-channel intercept is the feature c itself).  Two
    # features need this.  The content score s means different things on the two channels as soon as it comes from
    # a different model on each (Section IV-B), and a single linear weight then has to serve two scales.  The
    # lexical counts fire on benign third-party documents about as often as on attacks (they mention "assistant",
    # "system", "the model"), while on messages the user writes they do separate the two.  Empty = the system as
    # published, with one weight per feature.
    interactions: Tuple[str, ...] = ()


@dataclass
class PipeState:
    R: Optional[np.ndarray] = None
    trust: bool = False
    H: List[Tuple[int, float, float]] = field(default_factory=list)     # (t_i, r_i, f_i)
    a: float = 0.0
    D: Deque[bool] = field(default_factory=deque)
    B: Dict[str, int] = field(default_factory=dict)
    P_sum: Optional[np.ndarray] = None                                   # accumulated mode


@dataclass
class Record:
    """The decision record of one event (Table III in the paper)"""
    action: int
    p: float
    stage: str                    # "fast" | "hard" | "lr" | "none"
    x: Dict[str, float]
    new_task: bool
    reference_updated: bool
    reasons: List[str]
    f: float
    g: int
    m: float
    q: float
    s: float

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK

    @property
    def score(self) -> float:
        """Continuous score for the ROC: the fast and hard paths give 1.0"""
        return 1.0 if self.stage in ("fast", "hard") else self.p


class ProposedDetector:
    def __init__(self, stage2: Stage2, cfg: PipeConfig | None = None, name: str = "full", ext=None):
        self.s2 = stage2
        self.cfg = cfg or PipeConfig()
        self.name = name
        self.ext = ext                           # baselines.ExternalContentScore, or None for our own Stage 2
        self.model = None                        # sklearn Pipeline(StandardScaler, LogisticRegression)
        self.medians: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ helpers
    def new_state(self) -> PipeState:
        return PipeState(D=deque(maxlen=self.cfg.W))

    def _ref(self, st: PipeState) -> Optional[np.ndarray]:
        if self.cfg.reference_mode == "accumulated":
            if st.P_sum is None:
                return None
            n = float(np.linalg.norm(st.P_sum))
            return st.P_sum / n if n > 0 else None
        return st.R

    def is_task(self, text: str, e: np.ndarray, R: Optional[np.ndarray]) -> bool:
        if not has_content(text):
            return False
        return R is None or float(np.dot(e, R)) < self.cfg.tau_same

    def _behavior(self, H_ext: Sequence[Tuple[int, float, float]]) -> Tuple[float, float, float]:
        """(o, delta, h) over the window including the current event"""
        if len(H_ext) < self.cfg.k_min:
            return 0.0, 0.0, 1.0
        rs = [r for _, r, _ in H_ext]
        o = sum(1 for r in rs if r < self.cfg.tau_o) / len(rs)
        t_first, t_last = H_ext[0][0], H_ext[-1][0]
        delta = 0.0
        if t_first != t_last:
            first = [r for t, r, _ in H_ext if t == t_first]
            last = [r for t, r, _ in H_ext if t == t_last]
            delta = max(0.0, float(np.mean(first) - np.mean(last)))
        return o, delta, 0.0

    @property
    def feature_names(self) -> List[str]:
        """The columns the decision model actually sees: the features, then one channel interaction per entry of
        cfg.interactions (named "<f>:ind", equal to f on indirect events and 0 on direct ones)."""
        ix = [f for f in self.cfg.interactions if f in self.cfg.features]
        return list(self.cfg.features) + [f"{f}:ind" for f in ix]

    def vector(self, x: Dict[str, float]) -> np.ndarray:
        base = [x[k] for k in self.cfg.features]
        ix = [f for f in self.cfg.interactions if f in self.cfg.features]
        if ix:
            c = float(x["c"])
            base += [x[f] * c for f in ix]
        return np.array(base, dtype=float)

    def predict_p(self, x: Dict[str, float]) -> float:
        if self.model is None:
            return 0.0
        return float(self.model.predict_proba(self.vector(x)[None, :])[0, 1])

    def reasons(self, x: Dict[str, float], channel: str, k: int = 3) -> List[str]:
        """psi_j = beta_j (x_j - mean_j^(c)) / sd_j -> the 3 features that push the probability up most (Eq. 13)"""
        if self.model is None or channel not in self.medians:
            return []
        sc, lr = self.model.named_steps["scaler"], self.model.named_steps["lr"]
        v = self.vector(x)
        psi = lr.coef_[0] * (v - self.medians[channel]) / np.where(sc.scale_ > 0, sc.scale_, 1.0)
        # a feature and its channel interaction are the same signal to the reader, so their contributions are added
        agg: Dict[str, float] = {}
        for i, f in enumerate(self.feature_names):
            base = f[:-4] if f.endswith(":ind") else f
            if base in REASON_FEATURES:
                agg[base] = agg.get(base, 0.0) + psi[i]
        pairs = [(v_, f) for f, v_ in agg.items() if v_ > 0.05]
        pairs.sort(reverse=True)
        return [f"{FEATURE_TEXT.get(f, f)} +{v:.1f}" for v, f in pairs[:k]]

    # ------------------------------------------------------------------ Algorithm 1
    def step(self, ev: Event, st: PipeState) -> Record:
        cfg = self.cfg
        s2: Stage2Out = self.s2.process(ev.text, ev.channel)
        if self.ext is None:
            s = s2.s
        else:
            wins = (self.s2.windows(ev.text)
                    if getattr(self.ext, "window_scan", False) and self.s2._use_windows(ev.text, ev.channel) else None)
            s = self.ext.fuse(s2.s, ev.text, ev.channel, wins, ours_offset=float(getattr(self.s2, 'b0', 0.0)))
        e = s2.raw.e
        ref = self._ref(st)
        direct = ev.channel == DIRECT
        if cfg.reference_mode == "accumulated":
            new = False
        elif direct or cfg.channel_agnostic:
            new = self.is_task(ev.text, e, ref)
        else:
            new = False
        if direct:
            st.a = max(0.0, st.a + s - cfg.rho)                         # CUSUM (updated even when blocked)
        src = ev.source or ("user" if direct else "doc")
        u = float(st.B.get(src, 0))
        k = float(sum(st.D))
        f_raw = s2.raw.f

        def blocked_early(stage: str) -> Record:
            st.B[src] = st.B.get(src, 0) + 1
            x = dict(s=s, a=st.a if direct else 0.0, k=k if direct else 0.0, u=u, r=cfg.rbar,
                     c=0.0 if direct else 1.0, z=1.0, n_imp=0.0, n_ai=0.0, o=0.0, fbar=0.0, delta=0.0, h=1.0)
            return Record(BLOCK, 1.0, stage, x, new, False, [f"{stage} path"], f_raw, s2.g, s2.fired.m,
                          s2.fired.q, s)

        if s2.hard:
            return blocked_early("hard")
        if s2.fast:
            return blocked_early("fast")

        # ---- Stage 3 / Stage 4 (indirect) ----
        e_rel = s2.raw.e_top if cfg.relevance_unit == "window" else e
        if ref is not None and (not direct or cfg.channel_agnostic):
            r = float(np.dot(e_rel, ref))
        else:
            r = cfg.rbar
        if not direct and ref is not None:
            H_ext = st.H + [(ev.turn, r, f_raw)]
            o, delta, h = self._behavior(H_ext)
            fbar = float(np.mean([f for _, _, f in st.H])) if (st.H and h == 0.0) else 0.0
        else:
            o, delta, h, fbar = 0.0, 0.0, 1.0, 0.0
        trusted = ref is not None and (st.trust or cfg.reference_mode == "accumulated")
        z = 0.0 if trusted else 1.0
        n_imp, n_ai = lexical_counts(ev.text)
        x = dict(s=s, a=st.a if direct else 0.0, k=k if direct else 0.0, u=u, r=r, c=0.0 if direct else 1.0, z=z,
                 n_imp=float(n_imp), n_ai=float(n_ai), o=o, fbar=fbar, delta=delta, h=h)

        p = self.predict_p(x)
        action = BLOCK if p > cfg.t_blk else (FORWARD_UNTRUSTED if p > cfg.t_rev else FORWARD)
        if not direct and z == 1.0 and cfg.untrusted_floor:
            action = max(action, FORWARD_UNTRUSTED)

        # ---- update the state (the authority rule, which is not learned) ----
        updated = False
        if action == BLOCK:
            st.B[src] = st.B.get(src, 0) + 1
        elif cfg.reference_mode == "accumulated":
            st.P_sum = e.copy() if st.P_sum is None else st.P_sum + e
            if not direct:
                st.H = (st.H + [(ev.turn, r, f_raw)])[-cfg.W:]
        elif new:
            trust = (action == FORWARD and s < cfg.tau_a) if cfg.use_trust else True
            if cfg.drop_low_trust and not trust:
                st.R, st.trust = None, False
            else:
                st.R, st.trust = e, trust
            st.H = []
            updated = True
        elif not direct and ref is not None:
            st.H = (st.H + [(ev.turn, r, f_raw)])[-cfg.W:]
        elif direct and cfg.gamma_follow > 0.0 and action == FORWARD and st.R is not None:
            # a follow-up the pipeline forwarded nudges the reference towards itself (it never replaces it, and
            # it never grants trust); indirect content still cannot move R
            v = st.R + cfg.gamma_follow * e
            n = float(np.linalg.norm(v))
            if n > 0:
                st.R = v / n
        if direct:
            st.D.append(bool(new and action != BLOCK))
        return Record(action, p, "lr" if self.model is not None else "none", x, new, updated,
                      self.reasons(x, ev.channel), f_raw, s2.g, s2.fired.m, s2.fired.q, s)

    def run_session(self, sess: Session) -> List[Record]:
        st = self.new_state()
        return [self.step(ev, st) for ev in sess.events]

    def run_sessions(self, sessions: Sequence[Session]) -> List[List[Record]]:
        return [self.run_session(s) for s in sessions]

    # ------------------------------------------------------------------ copy with overrides
    def variant(self, name: str, stage2: Stage2 | None = None, ext="keep", **kw) -> "ProposedDetector":
        d = ProposedDetector(stage2 or self.s2, replace(self.cfg, **kw), name,
                             self.ext if ext == "keep" else ext)
        return d

    def describe(self) -> Dict[str, object]:
        c = self.cfg
        out = dict(name=self.name, features=",".join(c.features), W=c.W, k_min=c.k_min, tau_o=c.tau_o, rho=c.rho,
                   tau_same=c.tau_same, tau_a=c.tau_a, t_rev=c.t_rev, t_blk=c.t_blk, rbar=c.rbar,
                   relevance_unit=c.relevance_unit, reference_mode=c.reference_mode,
                   channel_agnostic=c.channel_agnostic, gamma_follow=c.gamma_follow,
                   interactions=",".join(c.interactions),
                   content_source=getattr(self.ext, "name", "ours"))
        if self.model is not None:
            lr = self.model.named_steps["lr"]
            out.update({f"beta_{f}": float(b) for f, b in zip(self.feature_names, lr.coef_[0])})
            out["beta_0"] = float(lr.intercept_[0])
            out["C"] = float(lr.C)
        return out


def action_name(a: int) -> str:
    return ACTION_NAMES[a]
