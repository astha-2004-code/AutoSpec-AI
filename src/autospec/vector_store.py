"""Persistent FAISS vector store with atomic persistence, locking, and page-text restoration."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from autospec.chunker import SpecificationChunk
from autospec.config import settings


class FAISSVectorStore:
    """Thread-safe FAISS vector index with document fingerprint caching and page-evidence persistence."""

    _io_locks: dict[str, threading.Lock] = {}
    _registry_lock = threading.Lock()

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
        self.page_text_map: dict[int, str] = {}
        self.page_meta_map: dict[int, dict[str, Any]] = {}

    @classmethod
    def _get_lock(cls, key: str) -> threading.Lock:
        with cls._registry_lock:
            if key not in cls._io_locks:
                cls._io_locks[key] = threading.Lock()
            return cls._io_locks[key]

    def _index_path(self, key: str) -> Path:
        return self.index_dir / f"{key}.faiss"

    def _metadata_path(self, key: str) -> Path:
        return self.index_dir / f"{key}_chunks.json"

    def _manifest_path(self, key: str) -> Path:
        return self.index_dir / f"{key}_manifest.json"

    def _pages_path(self, key: str) -> Path:
        return self.index_dir / f"{key}_pages.json"

    def is_fresh(self, fingerprint: str) -> bool:
        """Check if index artifacts exist and are consistent for the given key/fingerprint."""
        idx_p = self._index_path(fingerprint)
        meta_p = self._metadata_path(fingerprint)
        man_p = self._manifest_path(fingerprint)

        if not (idx_p.exists() and meta_p.exists() and man_p.exists()):
            return False

        try:
            with man_p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return (
                    data.get("fingerprint") == fingerprint
                    and data.get("chunk_count", 0) > 0
                )
        except Exception:
            return False

    def build_index(
        self,
        fingerprint: str,
        document_filename: str,
        chunks: list[SpecificationChunk],
        embeddings: np.ndarray,
        model_name: str = "all-MiniLM-L6-v2",
        total_pages: int | None = None,
        page_text_map: dict[int, str] | None = None,
        page_meta_map: dict[int, dict[str, Any]] | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> None:
        """Build and atomically persist a FAISS index, chunk store, manifest, and page text evidence."""
        if len(chunks) == 0:
            raise ValueError("Cannot build FAISS index with 0 specification chunks.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})."
            )

        dim = embeddings.shape[1]
        self.embedding_dim = dim

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        manifest: dict[str, Any] = {
            "fingerprint": fingerprint,
            "document_filename": document_filename,
            "chunk_count": len(chunks),
            "total_pages": total_pages if total_pages is not None else len(page_text_map or {}),
            "embedding_model": model_name,
            "dimension": dim,
            "created_at": datetime.now(UTC).isoformat(),
        }
        if extra_manifest:
            manifest.update(extra_manifest)

        lock = self._get_lock(str(self.index_dir / fingerprint))
        with lock:
            tmp_idx_path = self.index_dir / f"{fingerprint}.faiss.tmp"
            tmp_meta_path = self.index_dir / f"{fingerprint}_chunks.json.tmp"
            tmp_man_path = self.index_dir / f"{fingerprint}_manifest.json.tmp"
            tmp_pages_path = self.index_dir / f"{fingerprint}_pages.json.tmp"

            faiss.write_index(index, str(tmp_idx_path))

            chunks_data = [ch.to_dict() for ch in chunks]
            with tmp_meta_path.open("w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2)

            with tmp_man_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            pages_payload = {
                "page_text_map": {str(k): v for k, v in (page_text_map or {}).items()},
                "page_meta_map": {str(k): v for k, v in (page_meta_map or {}).items()},
            }
            with tmp_pages_path.open("w", encoding="utf-8") as f:
                json.dump(pages_payload, f)

            tmp_idx_path.replace(self._index_path(fingerprint))
            tmp_meta_path.replace(self._metadata_path(fingerprint))
            tmp_man_path.replace(self._manifest_path(fingerprint))
            tmp_pages_path.replace(self._pages_path(fingerprint))

        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        self.page_text_map = dict(page_text_map or {})
        self.page_meta_map = dict(page_meta_map or {})

    def load_index(self, fingerprint: str) -> bool:
        """Load an existing FAISS index, chunk metadata, manifest, and page texts from disk."""
        if not self.is_fresh(fingerprint):
            return False

        lock = self._get_lock(str(self.index_dir / fingerprint))
        with lock:
            idx_path = self._index_path(fingerprint)
            meta_path = self._metadata_path(fingerprint)
            man_path = self._manifest_path(fingerprint)
            pages_path = self._pages_path(fingerprint)

            self.index = faiss.read_index(str(idx_path))

            with meta_path.open("r", encoding="utf-8") as f:
                chunks_raw = json.load(f)
                self.chunks = [SpecificationChunk(**item) for item in chunks_raw]

            with man_path.open("r", encoding="utf-8") as f:
                self.manifest = json.load(f)

            self.page_text_map = {}
            self.page_meta_map = {}
            if pages_path.exists():
                with pages_path.open("r", encoding="utf-8") as f:
                    p_data = json.load(f)
                    self.page_text_map = {
                        int(k): str(v) for k, v in p_data.get("page_text_map", {}).items()
                    }
                    self.page_meta_map = {
                        int(k): v for k, v in p_data.get("page_meta_map", {}).items()
                    }

        return True

    def delete_index(self, fingerprint: str) -> None:
        """Remove persisted index artifacts for a document fingerprint."""
        lock = self._get_lock(str(self.index_dir / fingerprint))
        with lock:
            for path in (
                self._index_path(fingerprint),
                self._metadata_path(fingerprint),
                self._manifest_path(fingerprint),
                self._pages_path(fingerprint),
                self.index_dir / f"{fingerprint}.bm25",
            ):
                if path.exists():
                    path.unlink()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        subsystem_filter: str | None = None,
        document_ids: set[str] | None = None,
        manufacturer_filter: str | None = None,
        model_filter: str | None = None,
        year_filter: int | None = None,
    ) -> list[tuple[SpecificationChunk, float]]:
        """Search vector index for top-k closest chunks, applying metadata filters."""
        if self.index is None or len(self.chunks) == 0:
            return []

        has_filters = any([
            subsystem_filter,
            document_ids is not None,
            manufacturer_filter,
            model_filter,
            year_filter is not None,
        ])
        k_request = min(len(self.chunks), max(top_k * 5, 50) if has_filters else top_k)
        if k_request <= 0:
            return []

        scores, indices = self.index.search(query_vector.astype(np.float32), k_request)

        results: list[tuple[SpecificationChunk, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if document_ids is not None and chunk.document_id not in document_ids:
                continue
            if subsystem_filter and chunk.subsystem != subsystem_filter:
                continue
            if manufacturer_filter and manufacturer_filter.lower() not in (chunk.vehicle_make or "").lower():
                continue
            if model_filter and model_filter.lower() not in (chunk.vehicle_model or "").lower():
                continue
            if year_filter is not None and chunk.vehicle_year != year_filter:
                continue

            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results
