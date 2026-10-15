"""Persistent FAISS vector store with document fingerprint caching and safe rebuilding."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from autospec.chunker import SpecificationChunk
from autospec.config import settings


class FAISSVectorStore:
    """FAISS-based vector index for semantic retrieval of specification chunks."""

    def __init__(
        self,
        index_dir: Path | None = None,
        embedding_dim: int = 384,
    ) -> None:
        self.index_dir = index_dir or settings.get_index_dir()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.index: faiss.Index | None = None
        self.chunks: list[SpecificationChunk] = []
        self.manifest: dict[str, Any] = {}

    def _index_path(self, fingerprint: str) -> Path:
        return self.index_dir / f"{fingerprint}.faiss"

    def _metadata_path(self, fingerprint: str) -> Path:
        return self.index_dir / f"{fingerprint}_chunks.json"

    def _manifest_path(self, fingerprint: str) -> Path:
        return self.index_dir / f"{fingerprint}_manifest.json"

    def is_fresh(self, fingerprint: str) -> bool:
        """Check if index artifacts exist and are consistent for the given document fingerprint."""
        idx_p = self._index_path(fingerprint)
        meta_p = self._metadata_path(fingerprint)
        man_p = self._manifest_path(fingerprint)

        if not (idx_p.exists() and meta_p.exists() and man_p.exists()):
            return False

        try:
            with man_p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("fingerprint") == fingerprint and data.get("chunk_count", 0) > 0
        except Exception:
            return False

    def build_index(
        self,
        fingerprint: str,
        document_filename: str,
        chunks: list[SpecificationChunk],
        embeddings: np.ndarray,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Build and atomically persist a new FAISS index and chunk store."""
        if len(chunks) == 0:
            raise ValueError("Cannot build FAISS index with 0 specification chunks.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})."
            )

        dim = embeddings.shape[1]
        self.embedding_dim = dim

        # Build IndexFlatIP with normalized vectors for cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        # Atomic file persistence
        tmp_idx_path = self.index_dir / f"{fingerprint}.faiss.tmp"
        tmp_meta_path = self.index_dir / f"{fingerprint}_chunks.json.tmp"
        tmp_man_path = self.index_dir / f"{fingerprint}_manifest.json.tmp"

        faiss.write_index(index, str(tmp_idx_path))

        # Save chunk objects as json
        chunks_data = [ch.to_dict() for ch in chunks]
        with tmp_meta_path.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        manifest = {
            "fingerprint": fingerprint,
            "document_filename": document_filename,
            "chunk_count": len(chunks),
            "embedding_model": model_name,
            "dimension": dim,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with tmp_man_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Atomic replacement
        tmp_idx_path.replace(self._index_path(fingerprint))
        tmp_meta_path.replace(self._metadata_path(fingerprint))
        tmp_man_path.replace(self._manifest_path(fingerprint))

        self.index = index
        self.chunks = chunks
        self.manifest = manifest

    def load_index(self, fingerprint: str) -> bool:
        """Load an existing FAISS index and chunk metadata from disk."""
        if not self.is_fresh(fingerprint):
            return False

        idx_path = self._index_path(fingerprint)
        meta_path = self._metadata_path(fingerprint)
        man_path = self._manifest_path(fingerprint)

        self.index = faiss.read_index(str(idx_path))

        with meta_path.open("r", encoding="utf-8") as f:
            chunks_raw = json.load(f)
            self.chunks = [SpecificationChunk(**item) for item in chunks_raw]

        with man_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        subsystem_filter: str | None = None,
    ) -> list[tuple[SpecificationChunk, float]]:
        """Search vector index for top-k closest chunks, optionally filtering by subsystem."""
        if self.index is None or len(self.chunks) == 0:
            return []

        # Request more items if filtering by subsystem
        k_request = min(len(self.chunks), top_k * 3 if subsystem_filter else top_k)
        if k_request <= 0:
            return []

        scores, indices = self.index.search(query_vector.astype(np.float32), k_request)

        results: list[tuple[SpecificationChunk, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if subsystem_filter and chunk.subsystem != subsystem_filter:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results
