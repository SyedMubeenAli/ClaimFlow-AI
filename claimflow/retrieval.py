"""
Policy search: split pages into short chunks, then rank chunks against a query
with BM25, the standard keyword-ranking formula.

Why not embeddings and a vector index: one policy is a few dozen pages, keyword
ranking handles that well, it needs no model download, and it is easy to test.
If retrieval quality turns out to be the weak point, a vector retriever can sit
behind the same `search` method.

Every chunk carries an id, its source file and its page, so an answer can cite
exactly where it came from.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .extract import Page

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "we", "will", "with", "you", "your", "any", "if",
}


@dataclass(frozen=True)
class Chunk:
    """
    A piece of a document with the metadata needed to cite it. `text` is the page's
    own words (whitespace collapsed), so a quote can be checked against it exactly.
    """
    id: str
    source: str
    page: int
    text: str
    section: str = ""     # the nearest heading above this text, for example "Section 4. Excess"
    doc_ref: str = ""     # "d<document id>", so a chunk can be traced to its stored file

    @property
    def search_text(self) -> str:
        """What gets embedded and keyword-searched: the heading helps a short clause find its topic."""
        missing = [part for part in self.section.split(" / ") if part and part not in self.text]
        return f"{'. '.join(missing)}. {self.text}" if missing else self.text


def _pieces(sentence: str, limit: int) -> list[str]:
    return [sentence[i:i + limit] for i in range(0, len(sentence), limit)]


def chunk_pages(doc_ref: str, source: str, pages: list[Page], max_chars: int = 700) -> list[Chunk]:
    """Pack sentences into chunks of at most `max_chars`. A chunk never crosses a page."""
    chunks: list[Chunk] = []
    for page in pages:
        buffer, count = "", 0

        def flush() -> None:
            nonlocal buffer, count
            if buffer:
                chunks.append(Chunk(f"{doc_ref}-p{page.number}-{count}", source, page.number, buffer))
                buffer, count = "", count + 1

        for raw in re.split(r"(?<=[.;:!?])\s+|\n{2,}", page.text.strip()):
            sentence = " ".join(raw.split())
            for piece in _pieces(sentence, max_chars) if sentence else []:
                if buffer and len(buffer) + 1 + len(piece) > max_chars:
                    flush()
                buffer = f"{buffer} {piece}".strip()
        flush()
    return chunks


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Crude plural folding is enough here: "exclusions" should match "exclusion".
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in tokens if t not in _STOP]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tokens = [Counter(tokenize(c.search_text)) for c in chunks]
        self._lengths = [sum(t.values()) for t in self._tokens]
        self._avg = (sum(self._lengths) / len(chunks)) if chunks else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._tokens:
            document_frequency.update(counts.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in document_frequency.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, counts, length in zip(self.chunks, self._tokens, self._lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                    score += self._idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def search_many(self, queries: list[str], per_query: int = 4, total: int = 10) -> list[Chunk]:
        """Best chunks across several queries, without duplicates, in reading order."""
        best: dict[str, tuple[Chunk, float]] = {}
        for query in queries:
            for chunk, score in self.search(query, per_query):
                if chunk.id not in best or score > best[chunk.id][1]:
                    best[chunk.id] = (chunk, score)
        top = sorted(best.values(), key=lambda pair: -pair[1])[:total]
        return sorted((chunk for chunk, _ in top), key=lambda c: (c.source, c.page, c.id))
