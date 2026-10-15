"""Local semantic embeddings using SentenceTransformers with lazy loading and normalization."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from autospec.config import settings


class EmbedderProtocol(Protocol):
    """Protocol for embedding generators."""

    def embed_query(self, query: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Local SentenceTransformer embedder with L2 normalization for cosine similarity."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        self._model = None
        self._dimension = settings.embedding_dim

    @property
    def model(self):
        """Lazy load model on first inference call."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimension of the embedding model."""
        return self._dimension

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
        """Encode a single search query into a 2D float32 normalized vector."""
        if not query or not query.strip():
            # Return zero vector if query is blank
            vec = np.zeros((1, self.dimension), dtype=np.float32)
            return vec

        embedding = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)


class DeterministicMockEmbedder:
    """High-speed deterministic mock embedder for offline testing without weight downloads."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> np.ndarray:
        """Convert text into pseudo-random deterministic normalized vector using md5 seed."""
        if not text:
            return np.zeros(self.dimension, dtype=np.float32)

        # Generate seed from md5
        hex_digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(hex_digest[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
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
        """Generate deterministic mock vectors for a list of texts."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        matrix = np.array([self._hash_to_vector(t) for t in texts], dtype=np.float32)
        return matrix

    def embed_query(self, query: str) -> np.ndarray:
        """Generate deterministic mock vector for a query."""
        vec = self._hash_to_vector(query.strip())
        return vec.reshape(1, -1)


def get_embedder(model_name: str | None = None) -> EmbedderProtocol:
    """Factory function to acquire configured embedder instance."""
    name = model_name or settings.embedding_model
    if name.lower() in ("mock", "test", "dummy"):
        return DeterministicMockEmbedder(dimension=settings.embedding_dim)
    return SentenceTransformerEmbedder(model_name=name)
