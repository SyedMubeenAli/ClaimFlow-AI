"""
The vector store: finds the stored vectors closest to a query vector.

FAISS (IndexFlatIP, exact inner-product search) is used when faiss-cpu is installed.
Otherwise a NumPy matrix product gives the same answer. For one policy of a few dozen
chunks the two are equally fast, so the fallback costs nothing. Vectors are length 1,
so the inner product is the cosine similarity.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # not installed, or a platform without a wheel
    faiss = None


class VectorStore:
    def __init__(self, vectors: np.ndarray, *, force_numpy: bool = False):
        array = np.ascontiguousarray(vectors, dtype=np.float32)
        if array.ndim != 2:   # an empty index arrives as a flat or zero-size array
            array = array.reshape(len(array), -1) if array.size else np.zeros((0, 0), dtype=np.float32)
        self.vectors = array
        self.count = len(self.vectors)
        self.dim = self.vectors.shape[1] if self.count else 0
        self._index = None
        if faiss is not None and not force_numpy and self.count:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self.vectors)
        self.backend = "faiss" if self._index is not None else "numpy"

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The `k` closest vectors as (position, similarity), best first."""
        if not self.count or k <= 0:
            return []
        k = min(k, self.count)
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        similarities = self.vectors @ q[0]
        order = np.argsort(-similarities, kind="stable")[:k]
        return [(int(i), float(similarities[i])) for i in order]
