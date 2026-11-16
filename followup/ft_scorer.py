"""Fine-tuned MiniLM as a per-message scorer for the pipeline (baseline B4 / external content score).

The model is retrained for every seed on that seed's dev split only (head_training_data, the same data our Stage 2
head uses) plus Alpaca hard negatives, so it never sees a validation or test event of the seed it is scored on.
Settings are fixed in followup/PREREG.md and must not be changed after the run starts.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence

import numpy as np

from pidetect.baselines import ScoreBaseline
from pidetect.lexical import n_address_ai, n_imperative
from pidetect.runtime import log

BASE = "sentence-transformers/all-MiniLM-L6-v2"
HP = dict(epochs=4, batch=16, lr_enc=3e-5, lr_head=1e-3, wd=0.01, warmup=0.1, max_len=256)
SEG_STRIDE = 64          # token overlap between segments of a long text
MAX_SEGMENTS = 8         # same bound the pipeline uses for PIGuard
N_HARD = 600             # Alpaca hard negatives per seed (fixed in PREREG.md)
N_INJ = 600              # experiment G only: dev documents with a dev attack inserted (fixed in PREREG_G.md)
_SENT_END = __import__("re").compile(r"(?<=[.!?])\s+")


def _norm(t: str) -> str:
    return " ".join(str(t).lower().split())


def alpaca_hard_negatives(n: int, seed: int, banned: set) -> List[str]:
    """Benign Alpaca instructions that our own instruction-ness counts react to (n_imp > 0 or n_ai > 0)."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    pool = []
    for ins, inp in zip(ds["instruction"], ds["input"]):
        t = (ins + (" " + inp if inp else "")).strip()
        if 20 <= len(t) <= 600 and (n_imperative(t) > 0 or n_address_ai(t) > 0) and _norm(t) not in banned:
            pool.append(t)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def injected_documents(sp, seed: int, n: int) -> List[str]:
    """Experiment G: positives in which the attack sits INSIDE a document, built from dev data only.

    Half are real poisoned TaskTracker dev paragraphs (label 1). The other half are BIPIA dev contexts with one dev
    attack (BIPIA dev attack or dev direct attack) inserted at a random sentence boundary. The clean version of each
    context stays a negative, so the model has to find the instruction rather than judge length or topic."""
    rng = np.random.default_rng(seed + 1000)
    tt = [r["text"] for r in sp.tt.get("dev", []) if int(r["label"]) == 1]
    tt = [tt[i] for i in rng.permutation(len(tt))[: n // 2]]
    atks = [a for v in sp.bipia["dev"]["attacks"].values() for a in v] + [d["text"] for d in sp.att["dev"]]
    ctxs = [c["context"] for c in sp.bipia["dev"]["contexts"]]
    out = []
    for j in rng.permutation(len(ctxs))[: n - len(tt)]:
        sents = _SENT_END.split(str(ctxs[j]))
        k = int(rng.integers(0, len(sents) + 1))
        out.append(" ".join(sents[:k] + [str(atks[int(rng.integers(len(atks)))])] + sents[k:]))
    return tt + out


def _build(device):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    enc = AutoModel.from_pretrained(BASE)
    tok = AutoTokenizer.from_pretrained(BASE)

    class Classifier(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(enc.config.hidden_size, 2)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state
            m = attention_mask.unsqueeze(-1).float()
            return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1e-9)))

    return Classifier(enc).to(device), tok


def _device():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from pidetect.device import best_device   # cuda > mps > cpu; PI_DEVICE overrides
    return best_device()


class FineTunedMiniLM(ScoreBaseline):
    """Name starts with 'B4-' so that the h1 script's ext_content lookup ('B4') finds it."""
    name = "B4-OursFT"

    def __init__(self, inject: bool = False, name: str | None = None):
        self.inject = inject                 # experiment G: also train on documents with an injection inside
        if name:
            self.name = name
        self.device = _device()
        self.model = self.tok = None
        self.seed = None
        self._cache: Dict[str, float] = {}
        self.no_cache = False                # True while cost is measured: every text is scored fresh
        self._last = (None, 0.0)             # (text, score) of the last single text scored while no_cache
        self.train_info: Dict[int, dict] = {}

    # ---------------------------------------------------------------- training
    def fit_for_seed(self, seed: int, sp, hand_attacks: Sequence[str]):
        import torch
        from torch import nn
        from pidetect.sessions import head_training_data

        data = head_training_data(sp, hand_attacks)
        pos, neg = set(), set()
        for p, n in data.values():
            pos.update(map(str, p))
            neg.update(map(str, n))
        conflict = pos & neg
        pos, neg = sorted(pos - conflict), sorted(neg - conflict)
        banned = {_norm(t) for t in pos + neg}
        hard = alpaca_hard_negatives(N_HARD, seed, banned)
        inj = injected_documents(sp, seed, N_INJ) if self.inject else []
        texts = pos + inj + neg + hard
        y = np.array([1] * (len(pos) + len(inj)) + [0] * (len(neg) + len(hard)))
        log(f"[B4] seed {seed}: training on {len(pos)} attacks, {len(neg)} benign dev texts, "
            f"{len(hard)} Alpaca hard negatives, {len(inj)} injected documents "
            f"({len(conflict)} label conflicts dropped)")

        t0 = time.time()
        torch.manual_seed(seed)
        model, tok = _build(self.device)
        w = len(y) / (2 * np.bincount(y, minlength=2).clip(min=1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=self.device))
        opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": HP["lr_enc"]},
                                 {"params": model.head.parameters(), "lr": HP["lr_head"]}], weight_decay=HP["wd"])
        steps = HP["epochs"] * math.ceil(len(texts) / HP["batch"])
        warm = max(1, int(HP["warmup"] * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
        rng = np.random.default_rng(seed)
        model.train()
        for _ in range(HP["epochs"]):
            order = rng.permutation(len(texts))
            for i in range(0, len(order), HP["batch"]):
                idx = order[i:i + HP["batch"]]
                enc = tok([texts[j] for j in idx], truncation=True, padding=True,
                          max_length=HP["max_len"], return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                loss = loss_fn(model(**enc), torch.tensor(y[idx], device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        self.model, self.tok, self.seed = model, tok, seed
        self._cache = {}
        self.train_info[seed] = dict(n_pos=len(pos), n_inj=len(inj), n_neg=len(neg), n_hard=len(hard),
                                     n_conflicts=len(conflict), train_seconds=round(time.time() - t0, 1))
        log(f"[B4] seed {seed}: trained in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- scoring
    def _score_uncached(self, texts: List[str]) -> np.ndarray:
        import torch
        if not texts:
            return np.zeros(0)
        enc = self.tok(texts, truncation=True, max_length=HP["max_len"], stride=SEG_STRIDE,
                       return_overflowing_tokens=True, padding=True, return_tensors="pt")
        owner = enc.pop("overflow_to_sample_mapping").numpy()
        keep, seen = [], {}
        for r, o in enumerate(owner):                       # at most MAX_SEGMENTS segments per text
            seen[o] = seen.get(o, 0) + 1
            if seen[o] <= MAX_SEGMENTS:
                keep.append(r)
        keep = np.array(keep)
        enc = {k: v[keep] for k, v in enc.items()}
        owner = owner[keep]
        probs = []
        with torch.inference_mode():
            for i in range(0, len(keep), 32):
                b = {k: v[i:i + 32].to(self.device) for k, v in enc.items()}
                probs.append(torch.softmax(self.model(**b).float(), -1)[:, 1].cpu().numpy())
        probs = np.concatenate(probs)
        out = np.zeros(len(texts))
        for p, o in zip(probs, owner):                      # a text scores as high as its worst segment
            out[o] = max(out[o], p)
        return out

    def score(self, texts: Sequence[str], fresh: bool = False) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("B4-OursFT has not been trained for this seed")
        texts = [str(t) for t in texts]
        if self.no_cache and not fresh and len(texts) == 1 and texts[0] == self._last[0]:
            return np.array([self._last[1]])          # same event asked twice in one step: score it once
        if fresh or self.no_cache:
            out = self._score_uncached(texts)
            if len(texts) == 1:
                self._last = (texts[0], float(out[0]))
            return out
        todo = sorted({t for t in texts if t not in self._cache})
        for i in range(0, len(todo), 64):
            part = todo[i:i + 64]
            for t, s in zip(part, self._score_uncached(part)):
                self._cache[t] = float(s)
        return np.array([self._cache[t] for t in texts])

    def score_fresh(self, texts):
        return self.score(texts, fresh=True)
