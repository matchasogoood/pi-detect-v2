"""embedder.py : turns text into a semantic vector (an embedding)

Two implementations
  * SBERTEmbedder   : Sentence-BERT (all-MiniLM-L6-v2), the real one used in the paper  <-- for real runs
  * HashingEmbedder : a fallback that needs no model download, only for checking that the code runs
                      (it measures word overlap, not meaning -> its results must never go into the paper)

Every embedder normalizes its vectors to length 1, so cosine similarity is a dot product.
"""
from __future__ import annotations

import re
import warnings
import zlib
from typing import Iterable, List

import numpy as np

DEFAULT_SBERT = "sentence-transformers/all-MiniLM-L6-v2"


class BaseEmbedder:
    name = "base"
    dim = 0
    is_fallback = False  # True = results from this embedder must not be reported

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SBERTEmbedder(BaseEmbedder):
    """Sentence-BERT from sentence-transformers (needs internet on the first load)"""

    def __init__(self, model_name: str = DEFAULT_SBERT, device: str | None = None,
                 max_seq_length: int = 256):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers not found. Install it with:  pip install sentence-transformers"
            ) from e
        self.name = model_name
        from .device import best_device
        self.model = SentenceTransformer(model_name, device=device or best_device())
        self.model.max_seq_length = max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if len(texts) <= 4 * batch_size:                      # small calls stay silent
            vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        # a big call (a whole dataset) can take minutes on a CPU: encode in chunks and report when it is slow
        from .progress import track
        chunk = batch_size * 8
        parts = [self.model.encode(texts[i:i + chunk], batch_size=batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
                 for i in track(range(0, len(texts), chunk), total=(len(texts) + chunk - 1) // chunk,
                                desc=f"embedding {len(texts)} texts (in chunks)")]
        return np.asarray(np.concatenate(parts, axis=0), dtype=np.float32)


_TOKEN = re.compile(r"[a-z0-9äöüß]+")


class HashingEmbedder(BaseEmbedder):
    """Fallback: feature hashing of words (unigram + bigram) and 4-character shingles"""

    is_fallback = True

    def __init__(self, dim: int = 384):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        low = text.lower()
        words = _TOKEN.findall(low)
        feats = list(words) + [a + "_" + b for a, b in zip(words, words[1:])]
        joined = " ".join(words)
        feats += ["#" + joined[i:i + 4] for i in range(max(0, len(joined) - 3))]
        for f in feats:
            h = zlib.crc32(f.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 20) & 1 else -1.0
            v[idx] += sign
        # sublinear tf, then normalize
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(t) for t in texts]).astype(np.float32)


class CachedEmbedder(BaseEmbedder):
    """Wraps any embedder in a cache (keyed by the text) so nothing is encoded twice

    When measuring latency, call .base.encode(...) directly or call .clear() first
    """

    def __init__(self, base: BaseEmbedder):
        self.base = base
        self.name = base.name
        self.dim = base.dim
        self.is_fallback = base.is_fallback
        self._cache: dict[str, np.ndarray] = {}

    def clear(self) -> None:
        self._cache.clear()

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        missing: List[str] = []
        seen = set()
        for t in texts:
            if t not in self._cache and t not in seen:
                missing.append(t)
                seen.add(t)
        if missing:
            vecs = self.base.encode(missing, batch_size=batch_size)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder(kind: str = "sbert", model_name: str = DEFAULT_SBERT,
                 allow_fallback: bool = False, device: str | None = None) -> CachedEmbedder:
    """Builds the embedder named in the config

    kind = "sbert"   -> Sentence-BERT (if it cannot be loaded and allow_fallback=False, this raises a clear error)
    kind = "hashing" -> the fallback (for testing the code only)
    """
    if kind == "hashing":
        warnings.warn("Using HashingEmbedder (fallback) -> results are for testing the code only, not for the paper",
                      stacklevel=2)
        return CachedEmbedder(HashingEmbedder())
    try:
        return CachedEmbedder(SBERTEmbedder(model_name, device=device))
    except Exception as e:  # noqa: BLE001
        if allow_fallback:
            warnings.warn(f"Could not load {model_name} ({e}) -> falling back to HashingEmbedder (not for the paper)",
                          stacklevel=2)
            return CachedEmbedder(HashingEmbedder())
        raise RuntimeError(
            f"Could not load the embedding model '{model_name}': {e}\n"
            "Check that (1) sentence-transformers is installed and (2) the internet is reachable on the first load.\n"
            "To just check that the code runs, pass  --embedder hashing"
        ) from e


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of normalized vectors (= dot product)"""
    return float(np.dot(a, b))
