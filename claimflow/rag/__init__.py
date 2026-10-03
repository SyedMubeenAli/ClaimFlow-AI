"""The policy RAG subsystem: read, chunk, embed, store, retrieve, answer with citations."""

from .answer import NOT_FOUND, Citation, PolicyAnswer, answer_question, highlight_quote, split_on_quote
from .chunker import chunk_policy
from .embeddings import AutoEmbedder, Embedder, FastEmbedEmbedder, HashingEmbedder, build_embedder
from .index import Hit, PolicyIndex, PolicyIndexService
from .store import VectorStore

__all__ = [
    "AutoEmbedder", "Citation", "Embedder", "FastEmbedEmbedder", "HashingEmbedder", "Hit", "NOT_FOUND",
    "PolicyAnswer", "PolicyIndex", "PolicyIndexService", "VectorStore", "answer_question", "build_embedder",
    "chunk_policy", "highlight_quote", "split_on_quote",
]
