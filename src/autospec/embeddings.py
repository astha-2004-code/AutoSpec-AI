"""Local semantic embeddings, query embedding cache, and CPU-friendly cross-encoder reranking."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Protocol

import numpy as np

from autospec.config import settings

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """Protocol for embedding generators."""

    @property
    def dimension(self) -> int:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Thread-safe SentenceTransformer embedder with lazy loading, L2 normalization, and query cache."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_size: int = 512,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._dimension = settings.embedding_dim
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    @property
    def model(self):
        """Lazy load model once on first inference call."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    local_only = bool(
                        settings.offline_models or os.environ.get("HF_HUB_OFFLINE") == "1"
                    )
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=local_only,
                    )
                    if hasattr(self._model, "get_embedding_dimension"):
                        self._dimension = self._model.get_embedding_dimension()
                    elif hasattr(self._model, "get_sentence_embedding_dimension"):
                        self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimension of the embedding model."""
        return int(self._dimension or settings.embedding_dim)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a collection of texts into normalized float32 embeddings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a 2D float32 normalized vector with LRU cache."""
        q = (query or "").strip()
        if not q:
            return np.zeros((1, self.dimension), dtype=np.float32)

        with self._model_lock:
            if q in self._query_cache:
                self._query_cache.move_to_end(q)
                return self._query_cache[q]

        embedding = self.model.encode(
            [q],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        with self._model_lock:
            self._query_cache[q] = embedding
            if len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)

        return embedding


class DeterministicMockEmbedder:
    """High-speed deterministic mock embedder for offline testing without weight downloads."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_to_vector(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self._dimension, dtype=np.float32)

        hex_digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(hex_digest[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.array([self._hash_to_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._hash_to_vector(query.strip())
        return vec.reshape(1, -1)


class CandidateReranker:
    """Second-stage CPU reranker supporting optional CrossEncoder weights + deterministic feature scoring."""

    def __init__(
        self,
        model_name: str | None = None,
        use_neural: bool | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.use_neural = settings.use_cross_encoder_model if use_neural is None else use_neural
        self._cross_encoder: Any = None
        self._lock = threading.Lock()
        self._neural_failed = False

    def _get_cross_encoder(self):
        if not self.use_neural or self._neural_failed:
            return None
        if self._cross_encoder is None:
            with self._lock:
                if self._cross_encoder is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        local_only = bool(
                            settings.offline_models or os.environ.get("HF_HUB_OFFLINE") == "1"
                        )
                        self._cross_encoder = CrossEncoder(
                            self.model_name,
                            device=settings.device,
                            local_files_only=local_only,
                        )
                    except Exception as err:
                        logger.warning("CrossEncoder unavailable (%s); using feature reranker.", err)
                        self._neural_failed = True
                        return None
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        """Rerank candidate (chunk, rrf_score) pairs using CrossEncoder or deterministic domain features."""
        if not candidates:
            return []

        ce = self._get_cross_encoder()
        if ce is not None:
            try:
                pairs = [(query, c.content) for c, _ in candidates]
                ce_scores = ce.predict(pairs)
                reranked = []
                for (chunk, base_score), ce_score in zip(candidates, ce_scores, strict=False):
                    # Sigmoid-normalized neural logit combined with domain-weighted RRF score
                    prob = 1.0 / (1.0 + float(np.exp(-float(ce_score))))
                    combined = (base_score * 10.0) * 0.45 + prob * 0.55
                    reranked.append((chunk, round(combined, 6)))
                reranked.sort(key=lambda x: x[1], reverse=True)
                return reranked
            except Exception as err:
                logger.warning("CrossEncoder inference failed (%s); falling back to feature reranker.", err)

        return self._feature_rerank(query, candidates)

    @staticmethod
    def _feature_rerank(query: str, candidates: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
        """Deterministic CPU-zero-overhead reranker evaluating phrase, component, and numerical alignment."""
        q_tokens = set(re.findall(r"[a-z0-9\-]{2,}", query.lower()))
        q_clean = re.sub(r"\s+", " ", query.lower()).strip()

        reranked: list[tuple[Any, float]] = []
        for chunk, base_score in candidates:
            comp_low = (chunk.component or "").lower()
            comp_tokens = set(re.findall(r"[a-z0-9\-]{2,}", comp_low))
            content_tokens = set(re.findall(r"[a-z0-9\-]{2,}", (chunk.content or "").lower()))

            # Component token Jaccard / precision
            comp_overlap = len(q_tokens & comp_tokens) / max(1, len(comp_tokens)) if comp_tokens else 0.0
            query_cov = len(q_tokens & content_tokens) / max(1, len(q_tokens))
            exact_phrase_bonus = 0.25 if comp_low and len(comp_low) > 5 and comp_low in q_clean else 0.0
            row_bonus = 0.08 if chunk.chunk_type == "spec_row" else 0.0

            final_score = base_score * (1.0 + 0.65 * comp_overlap + 0.35 * query_cov + exact_phrase_bonus + row_bonus)
            reranked.append((chunk, round(final_score, 6)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


# Shared process-wide embedder cache so multiple isolated document pipelines share model weights
_EMBEDDER_CACHE: dict[str, EmbedderProtocol] = {}
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(model_name: str | None = None) -> EmbedderProtocol:
    """Factory returning a cached embedder instance to eliminate cold-start reloads."""
    name = (model_name or settings.embedding_model).strip()
    with _EMBEDDER_LOCK:
        if name in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[name]
        if name.lower() in ("mock", "test", "dummy"):
            inst: EmbedderProtocol = DeterministicMockEmbedder(dimension=settings.embedding_dim)
        else:
            inst = SentenceTransformerEmbedder(model_name=name)
        _EMBEDDER_CACHE[name] = inst
        return inst
