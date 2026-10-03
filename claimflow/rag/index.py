"""
The policy index: chunks, their vectors, and hybrid search over both.

Build once per uploaded policy and save to disk, so a Streamlit rerun or a restart does
not re-embed anything. The saved index is reused only if it is still valid: same file
(SHA-256), same index format, same embedding model. Otherwise it is rebuilt.

Search is hybrid. Vector similarity finds clauses that mean the same thing, keyword
ranking (BM25) finds exact terms such as "deductible" or "racing", and Reciprocal Rank
Fusion merges the two lists. A chunk counts as a match only if it clears the keyword
test or the embedder's similarity floor, so an unrelated question returns nothing
instead of the least-bad chunk.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..errors import IndexingError
from ..extract import has_text, read_pdf
from ..logging_setup import get_logger
from ..models import DocumentRecord, utc_now
from ..retrieval import BM25Retriever, Chunk
from ..uploads import resolve_stored_path
from .chunker import chunk_policy
from .embeddings import Embedder, HashingEmbedder
from .store import VectorStore

log = get_logger("rag.index")

SCHEMA = 1
RRF_K = 60
CANDIDATES = 20


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float                 # the fused rank score. Only the order is meaningful.
    vector_score: float | None   # cosine similarity, if the vector side matched
    lexical_score: float         # BM25 score, 0 if the keyword side did not match


class PolicyIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder,
                 manifests: list[dict], *, force_numpy: bool = False):
        self.chunks = chunks
        self.embedder = embedder
        self.manifests = manifests
        self.store = VectorStore(vectors, force_numpy=force_numpy)
        self.bm25 = BM25Retriever(chunks)
        self._position = {c.id: i for i, c in enumerate(chunks)}

    def search(self, query: str, k: int = 6) -> list[Hit]:
        if not self.chunks:
            return []
        vector = {i: s for i, s in self.store.search(self.embedder.embed_query(query), CANDIDATES)
                  if s >= self.embedder.floor}
        lexical = {self._position[c.id]: s for c, s in self.bm25.search(query, CANDIDATES)}
        fused: dict[int, float] = {}
        for ranking in (vector, lexical):
            for rank, i in enumerate(sorted(ranking, key=lambda j: -ranking[j])):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], fused[i], vector.get(i), lexical.get(i, 0.0)) for i in top]

    def search_many(self, queries: list[str], per_query: int = 4, total: int = 10) -> list[Chunk]:
        """The best chunks across several queries, without duplicates, in reading order."""
        best: dict[str, Hit] = {}
        for query in queries:
            for hit in self.search(query, per_query):
                if hit.chunk.id not in best or hit.score > best[hit.chunk.id].score:
                    best[hit.chunk.id] = hit
        top = sorted(best.values(), key=lambda h: -h.score)[:total]
        return sorted((h.chunk for h in top), key=lambda c: (c.source, c.page, c.id))

    @property
    def backend(self) -> str:
        return self.store.backend


@dataclass
class _Indexed:
    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict


class PolicyIndexService:
    """Builds, saves, loads and combines policy indexes. `index_dir=None` keeps everything in memory."""

    def __init__(self, upload_dir: Path, embedder: Embedder, index_dir: Path | None = None):
        self.upload_dir = Path(upload_dir)
        self.embedder = embedder
        self.index_dir = Path(index_dir) if index_dir is not None else None

    @classmethod
    def in_memory(cls, upload_dir: Path) -> "PolicyIndexService":
        return cls(upload_dir, HashingEmbedder(), None)

    # ------------------------------------------------------------ public

    def info(self, doc: DocumentRecord) -> dict | None:
        """The saved manifest for this document, or None if it is not indexed. Never builds anything."""
        manifest = self._read_manifest(self._folder(doc))
        return manifest if manifest and manifest.get("sha256") == doc.sha256 else None

    def ensure(self, doc: DocumentRecord) -> dict:
        """Make sure the document is indexed and current. Returns its manifest."""
        return self._ensure(doc).manifest

    def rebuild(self, doc: DocumentRecord) -> dict:
        return self._build(doc).manifest

    def corpus(self, docs: list[DocumentRecord], *, force_numpy: bool = False) -> PolicyIndex:
        """One searchable index over several policy documents, reusing each saved index."""
        if not docs:
            raise IndexingError("There is no policy document to search.")
        parts = [self._ensure(d) for d in docs]
        vectors = np.vstack([p.vectors for p in parts]) if len(parts) > 1 else parts[0].vectors
        return PolicyIndex([c for p in parts for c in p.chunks], vectors, self.embedder,
                           [p.manifest for p in parts], force_numpy=force_numpy)

    # ------------------------------------------------------------ build and load

    def _folder(self, doc: DocumentRecord) -> Path | None:
        return None if self.index_dir is None else self.index_dir / f"doc{doc.id}-{doc.sha256[:12]}"

    def _ensure(self, doc: DocumentRecord) -> _Indexed:
        loaded = self._load(doc)
        return loaded if loaded is not None else self._build(doc)

    def _build(self, doc: DocumentRecord) -> _Indexed:
        try:
            path = resolve_stored_path(self.upload_dir, doc.stored_path)
            data = path.read_bytes()
        except Exception as exc:
            raise IndexingError("The stored policy file could not be found.",
                                hint="Upload the policy again.") from exc
        pdf = read_pdf(data)
        if pdf is None:
            raise IndexingError("The uploaded policy could not be opened as a PDF.", hint="It may be damaged or password protected.")
        if not any(has_text(p.text) for p in pdf.pages):
            raise IndexingError("The uploaded policy could not be parsed as readable text.",
                                hint="It may be a scanned image. Upload a text-based PDF.")
        chunks = chunk_policy(f"d{doc.id}", doc.original_name, pdf.pages)
        if not chunks:
            raise IndexingError("The uploaded policy could not be parsed as readable text.",
                                hint="It may be a scanned image. Upload a text-based PDF.")
        try:
            vectors = self.embedder.embed([c.search_text for c in chunks])
        except Exception as exc:
            raise IndexingError(f"The policy could not be embedded ({type(exc).__name__}).") from exc

        sections: list[dict] = []
        for chunk in chunks:
            if chunk.section and chunk.section not in {s["title"] for s in sections}:
                sections.append({"title": chunk.section, "page": chunk.page})
        manifest = {
            "schema": SCHEMA, "doc_id": doc.id, "file": doc.original_name, "sha256": doc.sha256,
            "pages": pdf.page_count, "pages_indexed": len(pdf.pages),
            "pages_with_text": sum(has_text(p.text) for p in pdf.pages), "chunks": len(chunks),
            "chars": sum(len(c.text) for c in chunks), "sections": sections,
            "extractor": pdf.extractor, "pdf_metadata": pdf.metadata,
            "embedder": {"name": self.embedder.name, "model": self.embedder.model, "dim": self.embedder.dim,
                         "key": self.embedder.key},
            "vector_store": VectorStore(vectors).backend, "indexed_at": utc_now(),
        }
        self._save(self._folder(doc), manifest, chunks, vectors)
        log.info("Indexed %s: %d pages, %d chunks, %d sections (%s)", doc.original_name, pdf.page_count,
                 len(chunks), len(sections), self.embedder.key)
        return _Indexed(chunks, vectors, manifest)

    def _load(self, doc: DocumentRecord) -> _Indexed | None:
        folder = self._folder(doc)
        manifest = self._read_manifest(folder)
        if (manifest is None or manifest.get("schema") != SCHEMA or manifest.get("sha256") != doc.sha256
                or manifest.get("embedder", {}).get("key") != self.embedder.key):
            return None
        try:
            chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
            vectors = np.load(folder / "vectors.npy", allow_pickle=False)
        except (OSError, ValueError, TypeError):
            log.warning("Saved index for %s is unreadable; rebuilding", doc.original_name)
            return None
        if len(chunks) != len(vectors) or len(chunks) != manifest.get("chunks"):
            return None
        return _Indexed(chunks, vectors, manifest)

    @staticmethod
    def _read_manifest(folder: Path | None) -> dict | None:
        if folder is None:
            return None
        try:
            return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _save(folder: Path | None, manifest: dict, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if folder is None:
            return
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8")
            np.save(folder / "vectors.npy", vectors, allow_pickle=False)
            tmp = folder / "manifest.tmp"
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, folder / "manifest.json")   # written last: its presence means the index is complete
        except OSError as exc:
            log.warning("Could not save the policy index: %s", exc)   # an in-memory index still works
