"""
Turning text into vectors.

FastEmbedEmbedder   a small local model (BAAI/bge-small-en-v1.5 by default) run with
                    ONNX. No API key, no cost. The model is downloaded on first use.
HashingEmbedder     no model at all: words and word pairs hashed into a fixed-size
                    vector. It matches on shared wording, not on meaning, but it is
                    deterministic, instant and works offline. It is the fallback.
AutoEmbedder        tries FastEmbed once, and if that cannot start (not installed, or
                    no network for the first download) it switches to hashing and
                    records why. The app never fails to start because of embeddings.

Every vector is length 1, so a dot product is a cosine similarity. `floor` is the
similarity below which a vector match is treated as unrelated. Hybrid search also
uses keyword matching, so a weak floor on one side does not decide the result.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from ..logging_setup import get_logger
from ..retrieval import tokenize

log = get_logger("embeddings")


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class Embedder(ABC):
    name: str = "base"
    model: str = ""
    dim: int = 0
    floor: float = 0.0

    @property
    def key(self) -> str:
        """Identifies the vector space. An index built with one key cannot be searched with another."""
        return f"{self.name}:{self.model}:{self.dim}"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    name = "hashing"
    model = "signed-hashing-1gram-2gram"
    floor = 0.12

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")  # stable across runs
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = tokenize(text)
            features = Counter(words + [f"{a}_{b}" for a, b in zip(words, words[1:])])
            for feature, count in features.items():
                column, sign = self._bucket(feature)
                out[row, column] += sign * (1.0 + math.log(count))
        return _normalise(out)


class FastEmbedEmbedder(Embedder):
    name = "fastembed"
    floor = 0.45

    def __init__(self, model: str):
        from fastembed import TextEmbedding   # raises ImportError if not installed
        self.model = model
        self._engine = TextEmbedding(model_name=model)   # downloads the model on first use
        self.dim = int(np.asarray(list(self._engine.embed(["dimension probe"]))).shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.asarray(list(self._engine.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        try:   # BGE models work better when the query gets their search prefix, which this applies
            return _normalise(np.asarray(list(self._engine.query_embed(text)), dtype=np.float32))[0]
        except AttributeError:
            return super().embed_query(text)


class AutoEmbedder(Embedder):
    """Resolves lazily, on first use, so app start-up never waits for a model download."""

    def __init__(self, backend: str = "auto", model: str = "BAAI/bge-small-en-v1.5"):
        self.backend, self.requested_model = backend, model
        self._active: Embedder | None = None
        self.fallback_reason: str | None = None

    def _resolve(self) -> Embedder:
        if self._active is None:
            if self.backend in ("auto", "fastembed"):
                try:
                    self._active = FastEmbedEmbedder(self.requested_model)
                except Exception as exc:
                    self.fallback_reason = (f"fastembed is not available ({type(exc).__name__}). "
                                            "Using offline hashing embeddings, which match on wording and not meaning.")
                    log.warning(self.fallback_reason)
            if self._active is None:
                self._active = HashingEmbedder()
            log.info("Embeddings: %s", self._active.key)
        return self._active

    @property
    def active(self) -> Embedder:
        return self._resolve()

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._resolve().name

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._resolve().model

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._resolve().dim

    @property
    def floor(self) -> float:  # type: ignore[override]
        return self._resolve().floor

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._resolve().embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._resolve().embed_query(text)

    def describe(self) -> str:
        """For display. Does not trigger a model download."""
        if self._active is None:
            return f"not loaded yet (backend: {self.backend}, model: {self.requested_model})"
        return f"{self._active.name} ({self._active.model}, {self._active.dim} dimensions)"


def build_embedder(backend: str, model: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    return AutoEmbedder(backend, model)
