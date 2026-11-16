"""stage2.py : Stage 2 -- decides whether a message is an instruction-override injection (Section III-E, Eq. 3-6)

Each message is scored in two views: the original (raw) and the canonical copy from Stage 1.
For each view we compute
  m_v = max_j cos(e_v, a_j)                        similarity to the attack bank (Eq. 3)
  q_v = a logistic-regression head on the embedding (Eq. 4) -- long documents are scored per sentence window (Eq. 5)
  s_v = w_q q_v + w_m m_v + min(η, λ f_v + λ_g g_v)  (Eq. 6)  ;  s = max_v s_v
fast path: if m_v > tau_f and f_v != 0.5 -> block immediately (high precision, low recall)

Learned     : the per-channel head, w_q, w_m, lambda, lambda_g (fitted non-negative with L2, under the clip)
Calibrated  : tau_f (a very high quantile of m on benign text), kappa per length bin (quantile of the top window score)
Design choices: eta (the clip), the window size, the window budget
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .attackbank import AttackBank
from .canon import CanonConfig, canonicalize
from .rules import explain, rule_flag
from .schema import DIRECT, INDIRECT

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Stage2Config:
    doc_len_threshold: int = 400        # indirect text longer than this (characters) is scored per window
    window_sents: int = 2               # sentences per window
    max_windows: int = 16               # window budget per message
    length_bins: Tuple[int, ...] = (1, 4, 8)   # window-count bin edges: 1 | 2-4 | 5-8 | 9+
    head_C: float = 1.0                 # regularization strength of the head (larger C = weaker)
    eta: float = 0.3                    # eta, the clip bound on (lambda f + lambda_g g)
    clip: bool = True                   # False = no clip (used in the secondary experiment)
    fusion_l2: float = 0.05             # L2 penalty when fitting the fusion weights
    nu: float = 0.01                    # target document-level FPR (kappa uses nu/10)
    nu_fast: float = 0.001              # target FPR of the fast path
    tau_f_min: float = 0.85             # lowest tau_f we accept (keeps the fast path tight when data is scarce)
    fast_path: str = "gated"            # gated | ungated | off
    use_canonical: bool = True          # False = turn off Stage 1 canonicalization (secondary experiment)
    use_rule: bool = True               # False = λ f = 0
    use_obf: bool = True                # False = λ_g g = 0
    fixed_lambda: Optional[float] = None    # pin lambda (secondary: soft flag with lambda in {0, .1, .2, .3})
    hard_reject: bool = False           # True = block as soon as the rule matches (the "hard reject" baseline)


@dataclass
class ViewScore:
    name: str           # "raw" | "canonical"
    text: str
    f: float
    g: int
    m: float
    q: float
    base: float         # w_q q + w_m m
    rule_term: float    # min(η, λ f + λ_g g)
    s: float
    n_windows: int
    top_window: str
    e: np.ndarray       # embedding of the whole message
    e_top: np.ndarray   # embedding of the most instruction-like window (omega*)
    nearest_bank: str


@dataclass
class Stage2Out:
    s: float
    fast: bool                 # fast path fired (b_n)
    hard: bool                 # hard reject fired (only when cfg.hard_reject)
    views: List[ViewScore]
    fired: ViewScore           # the view with the highest s
    raw: ViewScore
    g: int
    changes: List[str] = field(default_factory=list)

    @property
    def f_raw(self) -> float:
        return self.raw.f


class Stage2:
    def __init__(self, embedder, bank: AttackBank, cfg: Stage2Config | None = None,
                 canon_cfg: CanonConfig | None = None):
        self.emb = embedder
        self.bank = bank
        self.cfg = cfg or Stage2Config()
        self.canon_cfg = canon_cfg or CanonConfig()
        self.heads: Dict[str, object] = {}
        # fusion weights (defaults, before fitting)
        self.b0, self.w_q, self.w_m, self.lam, self.lam_g = -2.0, 1.0, 1.0, 0.2, 0.2
        self.tau_f = 0.9
        self.kappa: Dict[int, float] = {}     # bin -> κ
        self._cache: Dict[Tuple[str, str], Stage2Out] = {}

    # ------------------------------------------------------------------ windows
    def windows(self, text: str) -> List[str]:
        sents = [s.strip() for s in _SENT.split(text) if s and s.strip()]
        n = len(sents)
        if n <= self.cfg.window_sents:
            return [text]
        size = max(self.cfg.window_sents, math.ceil(n / self.cfg.max_windows) + 1)
        stride = max(1, size - 1)
        out = [" ".join(sents[i:i + size]) for i in range(0, max(1, n - size + 1), stride)]
        if (n - size) % stride:                     # always cover the final sentence
            out.append(" ".join(sents[n - size:]))
        return out[: self.cfg.max_windows + 1]

    def length_bin(self, n_windows: int) -> int:
        for i, edge in enumerate(self.cfg.length_bins):
            if n_windows <= edge:
                return i
        return len(self.cfg.length_bins)

    def _use_windows(self, text: str, channel: str) -> bool:
        return channel == INDIRECT and len(text) > self.cfg.doc_len_threshold

    # ------------------------------------------------------------------ heads
    def _positive_windows(self, doc: str, needle: str, channel: str) -> List[str]:
        """The windows of a poisoned document that actually contain the injected instruction.

        At run time the head never sees a bare attack on this channel: it sees a window of a document, part
        document and part injection.  A positive example has to look the same, or the head is trained on one
        distribution and used on another -- which is what kept the indirect head weak on BIPIA and TaskTracker.
        Returns an empty list when no window holds the injection, so that no whole document is ever labelled
        positive by mistake.
        """
        wins = self.windows(doc) if self._use_windows(doc, channel) else [doc]
        hit = [w for w in wins if needle and needle in w]
        if not hit and len(needle) > 40:                 # the injection straddles a window boundary
            key = needle[:40]
            hit = [w for w in wins if key in w]
        return hit

    def fit_heads(self, data: Dict[str, Tuple[Sequence[object], Sequence[str]]], seed: int = 0) -> Dict[str, int]:
        """data = {channel: (positives, negative_texts)}  -- dev split only.
        A positive is either a text, or a pair (document, injected text): the pair is cut into the same windows
        the detector scores at run time and only the windows holding the injection are kept.
        Long negative texts are split into windows too, to match how they are scored at run time."""
        from sklearn.linear_model import LogisticRegression
        sizes = {}
        for ch, (pos, neg) in data.items():
            P, N = [], []
            for t in pos:
                if isinstance(t, (tuple, list)):
                    P.extend(self._positive_windows(t[0], t[1], ch))
                else:
                    P.append(t)
            for t in neg:
                N.extend(self.windows(t) if self._use_windows(t, ch) else [t])
            if not P or not N:
                raise ValueError(f"not enough training data for the {ch} head (pos={len(P)}, neg={len(N)})")
            X = self.emb.encode(P + N)
            y = np.array([1] * len(P) + [0] * len(N))
            clf = LogisticRegression(C=self.cfg.head_C, class_weight="balanced", max_iter=2000, random_state=seed)
            clf.fit(X, y)
            self.heads[ch] = clf
            sizes[ch] = len(y)
        self._cache.clear()
        return sizes

    def _head(self, channel: str):
        if channel not in self.heads:
            raise RuntimeError("the Stage 2 heads are not fitted yet (call fit_heads first)")
        return self.heads[channel]

    # ------------------------------------------------------------------ per-view scores
    def _q_and_top(self, text: str, channel: str, e: np.ndarray) -> Tuple[float, int, str, np.ndarray, List[float]]:
        head = self._head(channel)
        if not self._use_windows(text, channel):
            q = float(head.predict_proba(e[None, :])[0, 1])
            return q, 1, text, e, [q]
        wins = self.windows(text)
        E = self.emb.encode(wins)
        w = head.predict_proba(E)[:, 1]
        order = np.argsort(-w)
        top = int(order[0])
        if len(w) == 1:
            q = float(w[0])
        else:
            kap = self.kappa.get(self.length_bin(len(w)), math.inf)
            q = float(w[order[0]]) if w[order[0]] >= kap else float(w[order[1]])
        return q, len(wins), wins[top], E[top], [float(x) for x in w]

    def _rule_term(self, f: float, g: int) -> float:
        lam = self.cfg.fixed_lambda if self.cfg.fixed_lambda is not None else self.lam
        val = (lam * f if self.cfg.use_rule else 0.0) + (self.lam_g * g if self.cfg.use_obf else 0.0)
        return min(self.cfg.eta, val) if self.cfg.clip else val

    def _view(self, name: str, text: str, channel: str, f: float, g: int) -> ViewScore:
        e = self.emb.encode([text])[0]
        sims = self.bank.embs @ e if len(self.bank) else np.zeros(1)
        j = int(np.argmax(sims)) if len(self.bank) else -1
        m = float(sims[j]) if j >= 0 else 0.0
        q, nw, top_txt, e_top, _ = self._q_and_top(text, channel, e)
        base = self.w_q * q + self.w_m * m
        rt = self._rule_term(f, g)
        return ViewScore(name, text, f, g, m, q, base, rt, base + rt, nw, top_txt, e, e_top,
                         self.bank.texts[j] if j >= 0 else "")

    def process(self, text: str, channel: str) -> Stage2Out:
        key = (text, channel)
        if key in self._cache:
            return self._cache[key]
        f_raw = rule_flag(text)
        views = [self._view("raw", text, channel, f_raw, 0)]
        cres = canonicalize(text, self.canon_cfg) if self.cfg.use_canonical else None
        if cres is not None and cres.text and cres.text != text:
            views.append(self._view("canonical", cres.text, channel, rule_flag(cres.text), cres.g))
        fired = max(views, key=lambda v: v.s)
        fp = self.cfg.fast_path
        fast = False
        if fp != "off":
            fast = any(v.m > self.tau_f and (fp == "ungated" or v.f != 0.5) for v in views)
        hard = self.cfg.hard_reject and any(v.f > 0 for v in views)
        out = Stage2Out(fired.s, fast, hard, views, fired, views[0], cres.g if cres else 0,
                        cres.changes if cres else [])
        self._cache[key] = out
        return out

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ calibrate κ
    def calibrate_kappa(self, benign_indirect_texts: Sequence[str]) -> Dict[int, float]:
        """kappa per length bin = the (1 - nu/10) quantile of the top window score over benign documents"""
        head = self._head(INDIRECT)
        by_bin: Dict[int, List[float]] = {}
        for t in benign_indirect_texts:
            if not self._use_windows(t, INDIRECT):
                continue
            wins = self.windows(t)
            if len(wins) < 2:
                continue
            w = head.predict_proba(self.emb.encode(wins))[:, 1]
            by_bin.setdefault(self.length_bin(len(w)), []).append(float(w.max()))
        lvl = 1.0 - self.cfg.nu / 10.0
        self.kappa = {b: float(np.quantile(v, lvl, method="higher")) for b, v in by_bin.items() if v}
        self._cache.clear()
        return self.kappa

    # ------------------------------------------------------------------ fit fusion (Eq. 6)
    def fit_fusion(self, texts: Sequence[str], channels: Sequence[str], labels: Sequence[int]) -> Dict[str, float]:
        """Fits b0, w_q, w_m, lambda, lambda_g non-negative with L2, under the clip (not fit-then-clip)"""
        rows = []
        for t, ch, y in zip(texts, channels, labels):
            out = self.process(t, ch)
            for v in out.views:
                rows.append((v.q, v.m, v.f if self.cfg.use_rule else 0.0, v.g if self.cfg.use_obf else 0, y))
        A = np.array(rows, dtype=float)
        res = fit_fusion_arrays(*A.T, eta=self.cfg.eta, l2=self.cfg.fusion_l2, clip=self.cfg.clip,
                                fixed_lambda=self.cfg.fixed_lambda)
        self.b0, self.w_q, self.w_m, self.lam, self.lam_g = res["b0"], res["w_q"], res["w_m"], res["lam"], res["lam_g"]
        self._cache.clear()
        return res

    # ------------------------------------------------------------------ calibrate τ_f
    def calibrate_fast_path(self, benign_texts: Sequence[str], benign_channels: Sequence[str]) -> float:
        """tau_f = max(tau_f_min, the (1 - nu_fast) quantile of m over benign text)"""
        ms = []
        for t, ch in zip(benign_texts, benign_channels):
            out = self.process(t, ch)
            ms.extend(v.m for v in out.views)
        q = float(np.quantile(ms, 1.0 - self.cfg.nu_fast, method="higher")) if ms else 1.0
        self.tau_f = max(self.cfg.tau_f_min, q)
        self._cache.clear()
        return self.tau_f

    def params(self) -> Dict[str, float]:
        return dict(b0=self.b0, w_q=self.w_q, w_m=self.w_m, lam=self.lam, lam_g=self.lam_g, eta=self.cfg.eta,
                    tau_f=self.tau_f, **{f"kappa_bin{b}": k for b, k in self.kappa.items()})


def fit_fusion_arrays(q, m, f, g, y, eta: float = 0.3, l2: float = 0.05, clip: bool = True,
                      fixed_lambda: Optional[float] = None) -> Dict[str, float]:
    """Eq. 6 : logit = b0 + w_q q + w_m m + min(eta, lambda f + lambda_g g) -- non-negative with L2, class-balanced"""
    from scipy.optimize import minimize
    q, m, f, g, y = (np.asarray(a, dtype=float) for a in (q, m, f, g, y))
    if len(set(y.tolist())) < 2:
        raise ValueError("fit_fusion needs both attack and benign examples")
    sw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    def unpack(p):
        b0, wq, wm, lam, lamg = p
        return b0, wq, wm, (fixed_lambda if fixed_lambda is not None else lam), lamg

    def loss_grad(p):
        b0, wq, wm, lam, lamg = unpack(p)
        raw = lam * f + lamg * g
        active = (raw < eta) if clip else np.ones_like(raw, dtype=bool)
        z = b0 + wq * q + wm * m + np.where(active, raw, eta)
        pz = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        ll = -(sw * (y * np.log(pz + 1e-12) + (1 - y) * np.log(1 - pz + 1e-12))).sum() / sw.sum()
        reg = 0.5 * l2 * (wq ** 2 + wm ** 2 + lam ** 2 + lamg ** 2)
        r = sw * (pz - y) / sw.sum()
        grad = np.array([r.sum(), (r * q).sum() + l2 * wq, (r * m).sum() + l2 * wm,
                         0.0 if fixed_lambda is not None else (r * f * active).sum() + l2 * lam,
                         (r * g * active).sum() + l2 * lamg])
        return ll + reg, grad

    x0 = np.array([-2.0, 1.0, 1.0, 0.2 if fixed_lambda is None else fixed_lambda, 0.2])
    res = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                   bounds=[(None, None), (0, None), (0, None), (0, None), (0, None)])
    b0, wq, wm, lam, lamg = unpack(res.x)
    return dict(b0=float(b0), w_q=float(wq), w_m=float(wm), lam=float(lam), lam_g=float(lamg), loss=float(res.fun))


def fused_score(q, m, f, g, params: Dict[str, float], eta: float, clip: bool = True):
    q, m, f, g = (np.asarray(a, dtype=float) for a in (q, m, f, g))
    raw = params["lam"] * f + params["lam_g"] * g
    return params["w_q"] * q + params["w_m"] * m + (np.minimum(eta, raw) if clip else raw)
