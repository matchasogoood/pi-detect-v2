"""baselines.py : every comparison baseline

Each baseline gives a suspicion score per message (higher = more suspicious) with no notion of channel, session or R.
The threshold is then calibrated on validation data, at the same target FPR as our own pipeline.

  B1  keyword / regex filter           : any pattern match = attack (a fixed operating point, no threshold to tune)
  B2  banned terms + embedding         : in the style of Rahman 2024 -- our own reimplementation, not their code
  B3a Meta PromptGuard (86M)           : an off-the-shelf model, as used by PromptShield (gated on Hugging Face)
  B3b PIGuard (leolee99/PIGuard)       : an off-the-shelf model (needs trust_remote_code, i.e. it runs code from the model repo)

If a baseline cannot run (no access, no internet), the scripts skip it and record why, instead of failing the whole run.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .rules import BANNED_TERMS, find_matches


class BaselineUnavailable(RuntimeError):
    """This baseline cannot run on this machine (e.g. no model access, torch not installed)"""


class ScoreBaseline:
    name = "baseline"
    fixed_point = False        # True = no threshold to tune (use its own fixed verdict)
    reference_only = False

    def score(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def score_fresh(self, texts: Sequence[str]) -> np.ndarray:
        """Scores without using the cache -- used when measuring latency, so the comparison with the pipeline is fair
        (the pipeline is timed with the real embedding call included). Default = the normal score (baselines with no cache)."""
        return self.score(texts)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """For fixed_point baselines: returns 0/1"""
        return (self.score(texts) > 0.5).astype(int)


class RegexFilter(ScoreBaseline):
    """B1: keyword/regex -- the same pattern set as Stage 1, so the comparison is purely about how the rule output is used"""
    name = "B1-regex"
    fixed_point = True

    def score(self, texts):
        return np.array([1.0 if find_matches(t) else 0.0 for t in texts], dtype=float)


class RegexCount(ScoreBaseline):
    """B1t: the same patterns as B1, but the score is how many distinct patterns matched, so that a threshold
    can place it at the same false-positive target as the others (B1 itself has no threshold to set)"""
    name = "B1t-regex-count"

    def score(self, texts, fresh: bool = False):
        return np.array([float(len({m[0] for m in find_matches(t)})) for t in texts], dtype=float)


class SessionNormalized:
    """Any baseline, judged against the session it is in.

    A per-message detector sees one text at a time, while our pipeline reads the session, so comparing the two
    straight away gives our side the benefit of context.  This wrapper gives the baseline the cheapest form of the
    same thing: an event is scored by how far it stands above the events that came before it in the same session
    (causal, so no event is judged with information from its future).  Sessions are supplied by evaluate.py.
    """

    def __init__(self, base, name: Optional[str] = None):
        self.base = base
        self.name = name or (base.name + "+sessnorm")
        self.fixed_point = False
        self.session_norm = True

    def score(self, texts, fresh: bool = False):
        return self.base.score(texts, fresh=fresh) if hasattr(self.base, "score") else self.base.score(texts)

    def score_fresh(self, texts):
        return self.base.score_fresh(texts)

    def score_channels(self, texts, channels):
        if hasattr(self.base, "score_channels"):
            return self.base.score_channels(texts, channels)
        return self.base.score(texts)

    @staticmethod
    def transform(scores: Sequence[float]) -> np.ndarray:
        """scores of one session, in order -> each event minus the median of the events before it"""
        out, seen = [], []
        for v in scores:
            out.append(float(v) - (float(np.median(seen)) if seen else 0.0))
            seen.append(float(v))
        return np.array(out, dtype=float)


class BannedTermsEmbedding(ScoreBaseline):
    """B2: score = max(similarity to a banned phrase, 1.0 if a banned phrase appears verbatim)
    No channel, no reference, no soft flag (compare with Rahman 2024: banned terms + BERT similarity)"""
    name = "B2-banned+emb"

    def __init__(self, embedder, terms: Sequence[str] = BANNED_TERMS):
        self.emb = embedder
        self.terms = [t.lower() for t in terms]
        self.term_embs = embedder.encode(self.terms)

    def score(self, texts, fresh: bool = False):
        texts = list(texts)
        base = getattr(self.emb, "base", None)           # CachedEmbedder.base = the real encoder, without the cache
        E = base.encode(texts) if (fresh and base is not None) else self.emb.encode(texts)
        sim = (E @ self.term_embs.T).max(axis=1)
        lit = np.array([1.0 if any(t in x.lower() for t in self.terms) else 0.0 for x in texts])
        # a verbatim match adds to the similarity instead of replacing it with 1.0.  With max(sim, 1.0) every text
        # that quotes a banned phrase shares one score at the top of the scale, and since benign text does quote
        # them (a security article), more than the allowed share of benign items sits at that ceiling and no
        # threshold reaches the target false-positive rate: the baseline then blocks nothing and cannot be compared.
        return sim + 0.5 * lit

    def score_fresh(self, texts):
        return self.score(texts, fresh=True)


class HFClassifierBaseline(ScoreBaseline):
    """Wrapper around a Hugging Face text-classification model

    Score = 1 - P(benign), i.e. the probability that the text is an attack of any kind.

    Two details follow the model cards instead of our own convenience, so that the baseline is used the way its
    authors intend (otherwise the comparison would not be fair):

    * per channel.  Prompt-Guard exposes three labels (benign / injection / jailbreak) and its model card gives a
      different rule for each channel: for dialogue with the user the jailbreak probability alone
      ("get_jailbreak_score", because users are allowed to write instructions), and for third-party content the sum
      of the injection and jailbreak probabilities ("get_indirect_injection_score").  Applying the stricter
      third-party rule to user messages flags almost every benign message, so direct events use the jailbreak
      probability.  Models with only two labels (benign / injection) use 1 - P(benign) on both channels.
    * long inputs.  The context window is 512 tokens, and the model cards ask for longer inputs to be split into
      segments that are scanned one by one; we score every segment and keep the highest, so that a payload buried
      far down a document is still seen (truncation would hide it).
    """

    BENIGN_NAMES = {"benign", "safe", "legit", "legitimate", "clean", "normal", "label_0", "0", "negative"}
    MAX_SEGMENTS = 8                     # at most this many 512-token segments per text (bounds the cost)
    SEGMENT_STRIDE = 64                  # tokens of overlap, so a payload on a segment boundary is not cut in two

    @classmethod
    def pick_benign_index(cls, id2label: Dict[int, str]):
        """Picks the index of the label that means benign; returns (index, whether a known name was found).
        If no known name is found it assumes index 0 (and the caller should warn the user to check)."""
        ben = [i for i, n in id2label.items() if str(n).lower() in cls.BENIGN_NAMES]
        return (ben[0], True) if ben else (0, False)

    def __init__(self, model_id: str, name: str, trust_remote_code: bool = False, revision: Optional[str] = None,
                 device: Optional[str] = None, max_length: int = 512, batch_size: int = 16,
                 local_files_only: bool = False):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise BaselineUnavailable("needs torch and transformers  (pip install torch transformers)") from e
        self.name = name
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        kw = dict(trust_remote_code=trust_remote_code)
        if local_files_only:
            kw["local_files_only"] = True
        if revision:
            kw["revision"] = revision
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
        except Exception as e:  # noqa: BLE001
            raise BaselineUnavailable(
                f"could not load {model_id}: {type(e).__name__}: {str(e)[:300]}\n"
                "for a gated model (Prompt-Guard), accept the license on the model page and log in with "
                "`huggingface-cli login` or set the HF_TOKEN variable") from e
        from .device import best_device
        self.device = device or best_device()
        self.model.to(self.device).eval()
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.id2label = id2label
        self.benign_idx, found = self.pick_benign_index(id2label)
        if not found:
            print(f"[{name}] warning: no benign label found in {id2label} -> assuming index 0 is benign (please check)")
        self.jailbreak_idx = next((i for i, n in id2label.items() if "jailbreak" in str(n).lower()), None)
        self.injection_idx = next((i for i, n in id2label.items() if "injection" in str(n).lower()), None)
        # only a model that separates the two attack types can follow the per-channel rule of its model card
        self.channel_aware = self.jailbreak_idx is not None and self.injection_idx is not None
        print(f"[{name}] id2label = {id2label} ; benign index = {self.benign_idx} ; device = {self.device}"
              + (f" ; per-channel scoring: direct = P({id2label[self.jailbreak_idx]}), "
                 f"indirect = 1 - P({id2label[self.benign_idx]})" if self.channel_aware else ""))
        # never re-run the model on text it has already seen (texts repeat across sessions and seeds);
        # the key carries the scoring mode, because the same text scores differently on the two channels
        self._cache: Dict[tuple, float] = {}

    def _segments(self, text: str) -> List[List[int]]:
        """Splits one text into at most MAX_SEGMENTS windows of max_length tokens (the model card's advice)"""
        enc = self.tok(text, add_special_tokens=True, truncation=True, max_length=self.max_length,
                       return_overflowing_tokens=True, stride=self.SEGMENT_STRIDE)
        return enc["input_ids"][:self.MAX_SEGMENTS]

    def _score_uncached(self, texts, mode: str = "strict"):
        """mode = strict -> 1 - P(benign) ; mode = jailbreak -> P(jailbreak)  (see the class docstring)"""
        torch = self._torch
        pad = self.tok.pad_token_id or 0
        segs, owner = [], []                       # one row per segment, plus the index of the text it came from
        for i, t in enumerate(texts):
            ids = self._segments(t)
            if not ids:
                ids = [[]]
            for one in ids:
                segs.append(one)
                owner.append(i)
        out = np.zeros(len(texts), dtype=float)
        order = np.argsort([len(x) for x in segs])        # batch segments of similar length together
        with torch.no_grad():
            for i in range(0, len(segs), self.batch_size):
                idx = order[i:i + self.batch_size]
                width = max(len(segs[j]) for j in idx)
                ids = torch.tensor([segs[j] + [pad] * (width - len(segs[j])) for j in idx], device=self.device)
                att = torch.tensor([[1] * len(segs[j]) + [0] * (width - len(segs[j])) for j in idx],
                                   device=self.device)
                p = torch.softmax(self.model(input_ids=ids, attention_mask=att).logits, dim=-1).cpu().numpy()
                if mode == "jailbreak" and self.jailbreak_idx is not None:
                    v = p[:, self.jailbreak_idx]
                else:
                    v = 1.0 - p[:, self.benign_idx]
                for k, j in enumerate(idx):                # a text scores as high as its worst segment
                    out[owner[j]] = max(out[owner[j]], float(v[k]))
        return out

    def _mode_for(self, channel: Optional[str]) -> str:
        return "jailbreak" if (self.channel_aware and channel == "direct") else "strict"

    def _cached(self, texts, modes):
        todo = {(m, t) for t, m in zip(texts, modes) if (m, t) not in self._cache}
        for mode in sorted({m for m, _ in todo}):
            batch = [t for m, t in todo if m == mode]
            if batch:
                for t, v in zip(batch, self._score_uncached(batch, mode)):
                    self._cache[(mode, t)] = float(v)
        return np.array([self._cache[(m, t)] for t, m in zip(texts, modes)], dtype=float)

    def score(self, texts, fresh: bool = False):
        """Channel-blind scoring (1 - P(benign)); used where no channel is available, e.g. the latency measurement"""
        texts = [t if t.strip() else " " for t in texts]
        if fresh:                                          # used when measuring latency: never read the cache
            return self._score_uncached(texts, "strict")
        return self._cached(texts, ["strict"] * len(texts))

    def score_channels(self, texts, channels):
        """Scores each text with the rule its channel calls for (see the class docstring)"""
        texts = [t if t.strip() else " " for t in texts]
        return self._cached(texts, [self._mode_for(c) for c in channels])

    def score_fresh(self, texts):
        return self.score(texts, fresh=True)


# How a public guard model is used as Stage 2's content score.  Each mode names options for the wrapper below
# ("ext") and, where the mode needs it, overrides for the pipeline ("pipe").
#   weight 1.0 = its score alone, 0.5 = average with ours | combine "max" = the higher of the two
#   window = also read each window of a long third-party document | only = use it on this channel alone
#   logit  = give the decision model the log-odds of the probability instead of the probability itself
EXT_MODES = {
    "ext": dict(weight=1.0),
    "fused": dict(weight=0.5),
    "ext_win": dict(weight=1.0, window=True),
    "fused_win": dict(weight=0.5, window=True),
    "max_win": dict(combine="max", window=True),
    "ind_win": dict(weight=1.0, window=True, only="indirect"),
    # the same as ind_win, with the two corrections of Section IV-B: the log-odds scale, and a decision model that
    # fits a separate slope per channel for the content score and the two lexical counts
    "ind_win_logit": dict(weight=1.0, window=True, only="indirect", logit=True,
                          pipe=dict(interactions=("s", "n_imp", "n_ai"))),
    # the log-odds scale on its own.  The per-channel slopes above cost detection on the direct channel in the
    # development run, including in the variant that uses them without any external model, so the two corrections
    # are separated here rather than kept as one change.
    "ind_win_logit_only": dict(weight=1.0, window=True, only="indirect", logit=True),
    # the log-odds scale with one extra parameter instead of three: a separate slope for the content score alone,
    # which is the only feature whose scale actually differs between the channels once the score comes from
    # another model.  The three-feature version cost direct detection in the development run.
    "ind_win_logit_s": dict(weight=1.0, window=True, only="indirect", logit=True,
                            pipe=dict(interactions=("s",))),
    # Two candidates added after the first reported run (seeds 10-14), which showed that on the direct channel the
    # content score is still ours while PIGuard's is used on the indirect one.  In the run above the direct score
    # lives on a 0-1 scale and the indirect one on a log-odds scale, so one weight in the decision model has to read
    # both.  These two put both channels on the log-odds scale.  Seeds 10-14 are development seeds from here on.
    #   ext_win_logit : the guard model's log-odds is the content score on both channels
    #   ens_win_logit : the mean of the guard model's log-odds and the log-odds of our own Stage 2 score, on both
    #                   channels (a fixed equal-weight rule, no parameter fitted on any outcome)
    "ext_win_logit": dict(weight=1.0, window=True, logit=True),
    "ens_win_logit": dict(combine="mean_logit", window=True, logit=True),
}


def ext_pipe_overrides(tag: str) -> Dict:
    """The pipeline options a mode needs (empty for most of them)"""
    return dict(EXT_MODES[tag].get("pipe", {}))


class ExternalContentScore:
    """A public guard model used as the content signal of Stage 2, so that the session layer can be tested on top
    of a baseline's own classifier.

    The point of the comparison is fairness: if the same per-message model scores the text in both systems, any
    remaining difference belongs to the session layer and not to a better content model.  Scores are cached per
    (channel, text), so a text that repeats across sessions and seeds is encoded once.

    Note on the log-odds option.  A probability that saturates near 1 compresses the whole top of the scale, which
    is exactly the region a low false-positive threshold reads, while a logistic decision model is linear in the
    log-odds; we therefore pass log(v / (1 - v)) rather than v.  The transform is strictly monotone, so it cannot
    change the baseline's own numbers -- its AUC and its detection rate at any false-positive rate are invariant --
    and it changes only how our decision model reads the same score.
    """

    # The guard models' probabilities come from a float32 softmax, which resolves about 1e-7 away from 0 and 1
    # (log-odds +-16).  Clipping at 20 therefore ties no two values the model itself can tell apart, and keeps a
    # probability that rounded to exactly 0 or 1 from becoming infinite.
    LOGIT_CLIP = 20.0

    def __init__(self, bl, weight: float = 1.0, window_scan: bool = False, combine: str = "avg",
                 only_channel: Optional[str] = None, logit: bool = False):
        self.bl = bl
        self.weight = float(weight)            # avg: 1.0 = use the external score alone; 0.5 = average with ours
        self.window_scan = bool(window_scan)  # long third-party text: also score each Stage 3 window, keep the worst
        self.combine = combine                 # "avg" or "max" (the higher of our score and the external one)
        self.only_channel = only_channel       # use the external score on this channel only; ours elsewhere
        self.logit = bool(logit)               # pass the log-odds instead of the probability
        self.name = getattr(bl, "name", "external")

    @classmethod
    def from_mode(cls, bl, tag: str):
        o = EXT_MODES[tag]
        return cls(bl, weight=o.get("weight", 1.0), window_scan=o.get("window", False),
                   combine=o.get("combine", "avg"), only_channel=o.get("only"), logit=o.get("logit", False))

    def score_many(self, texts, channel: str):
        if hasattr(self.bl, "score_channels"):
            return np.asarray(self.bl.score_channels(list(texts), [channel] * len(texts)), dtype=float)
        return np.asarray(self.bl.score(list(texts)), dtype=float)

    def score_one(self, text: str, channel: str) -> float:
        return float(self.score_many([text], channel)[0])

    def _scale(self, v: float) -> float:
        if not self.logit:
            return v
        eps = 1.0 / (1.0 + math.exp(self.LOGIT_CLIP))
        v = min(max(v, eps), 1.0 - eps)
        return math.log(v / (1.0 - v))

    def _ours_log_odds(self, s_ours: float, offset: float) -> float:
        """Our Stage 2 score is the linear part of a logistic model (Eq. 6, logit = b0 + w_q q + w_m m + rule term, with
        the bias b0 added when a probability is wanted), so it is already on the log-odds scale up to the bias b0;
        offset passes that bias, and the result is clipped like the guard model's."""
        return max(-self.LOGIT_CLIP, min(self.LOGIT_CLIP, float(s_ours) + float(offset)))

    def fuse(self, s_ours: float, text: str, channel: str, windows=None, ours_offset: float = 0.0) -> float:
        """windows = the Stage 3 windows of a long third-party text (None for everything else).  The guard model
        then also reads each window on its own and the text counts as high as its worst window, the same rule the
        guard models already follow for their 512-token segments, only at the finer grain of the window."""
        if self.only_channel is not None and channel != self.only_channel:
            return float(s_ours)
        v = self.score_one(text, channel)
        if self.window_scan and windows is not None and len(windows) > 1:
            v = max(v, float(self.score_many(windows, channel).max()))
        v = self._scale(v)                     # monotone: the ordering of the guard model's own scores is kept
        if self.combine == "mean_logit":
            return 0.5 * (self._ours_log_odds(s_ours, ours_offset) + v)
        if self.combine == "max":
            return max(float(s_ours), v)
        return v if self.weight >= 1.0 else (1.0 - self.weight) * float(s_ours) + self.weight * v


def make_baselines(cfg: Dict, embedder, names: Iterable[str]):
    """Builds baselines by name (B1 / B2 / B3a / B3b / B3c); returns (built, skipped)
    built = {name: baseline} ; skipped = {name: reason}"""
    built: Dict[str, ScoreBaseline] = {}
    skipped: Dict[str, str] = {}
    bc = cfg.get("baselines", {})
    for n in names:
        try:
            if n == "B1":
                built["B1-regex"] = RegexFilter()
            elif n == "B1t":
                built["B1t-regex-count"] = RegexCount()
            elif n == "B2":
                built["B2-banned+emb"] = BannedTermsEmbedding(embedder)
            elif n == "B3a":
                c = bc.get("promptguard", {})
                if not c.get("enabled", True):
                    raise BaselineUnavailable("disabled in config.yaml (baselines.promptguard.enabled)")
                built["B3a-PromptGuard"] = HFClassifierBaseline(c.get("model_id", "meta-llama/Prompt-Guard-86M"),
                                                                 "B3a-PromptGuard")
            elif n == "B3b":
                c = bc.get("piguard", {})
                if not c.get("enabled", False):
                    raise BaselineUnavailable("disabled by default because PIGuard needs trust_remote_code=True, "
                                              "which runs code from the model repo -- enable it in config.yaml "
                                              "(baselines.piguard.enabled: true) after reading that code")
                built["B3b-PIGuard"] = HFClassifierBaseline(c.get("model_id", "leolee99/PIGuard"), "B3b-PIGuard",
                                                             trust_remote_code=True, revision=c.get("revision"))
            elif n == "B3c":
                c = bc.get("protectai", {})
                if not c.get("enabled", True):
                    raise BaselineUnavailable("disabled in config.yaml (baselines.protectai.enabled)")
                # public (Apache-2.0, not gated), a plain DeBERTa-v3 classifier: no remote code is run
                built["B3c-ProtectAIv2"] = HFClassifierBaseline(
                    c.get("model_id", "protectai/deberta-v3-base-prompt-injection-v2"), "B3c-ProtectAIv2",
                    revision=c.get("revision"))
            elif n.endswith("+sn"):                      # e.g. "B3c+sn": the same baseline, judged within its session
                inner, _ = make_baselines(cfg, embedder, [n[:-3]])
                if not inner:
                    raise BaselineUnavailable(f"the baseline {n[:-3]} it wraps is unavailable")
                base = list(inner.values())[0]
                built[base.name + "+sessnorm"] = SessionNormalized(base)
            else:
                skipped[n] = "unknown baseline name"
        except BaselineUnavailable as e:
            skipped[n] = str(e)
            print(f"[baseline {n}] skipped: {e}")
    return built, skipped
