"""Lightweight ONNX, external Gemini API, and optional local semantic embeddings + CPU reranking."""

from __future__ import annotations

import gc
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

_ONNX_MODEL_ALIASES = {
    "all-minilm-l6-v2": "Xenova/all-MiniLM-L6-v2",
    "sentence-transformers/all-minilm-l6-v2": "Xenova/all-MiniLM-L6-v2",
    "xenova/all-minilm-l6-v2": "Xenova/all-MiniLM-L6-v2",
}


class EmbedderProtocol(Protocol):
    """Protocol for embedding generators."""

    model_name: str

    @property
    def dimension(self) -> int:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        ...


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (arr / norms).astype(np.float32)


class OnnxEmbeddingEmbedder:
    """Memory-efficient CPU ONNX embedder (no PyTorch/CUDA/transformers required)."""

    def __init__(
        self,
        model_name: str | None = None,
        onnx_file: str = "onnx/model_quantized.onnx",
        cache_size: int | None = None,
        max_length: int = 192,
    ) -> None:
        raw_name = (model_name or settings.embedding_model).strip()
        self.repo_id = _ONNX_MODEL_ALIASES.get(raw_name.lower(), raw_name)
        self.model_name = f"onnx:{self.repo_id}:qint8"
        self.onnx_file = onnx_file
        self._dimension = settings.embedding_dim
        self._max_length = max_length
        self._cache_size = cache_size or settings.embedding_cache_size
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()

    def _ensure_loaded(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            local_only = bool(settings.offline_models or os.environ.get("HF_HUB_OFFLINE") == "1")
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.onnx_file,
                local_files_only=local_only,
            )
            tok_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="tokenizer.json",
                local_files_only=local_only,
            )

            tok = Tokenizer.from_file(tok_path)
            tok.enable_truncation(max_length=self._max_length)
            tok.enable_padding(length=None)

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess_opts.enable_mem_pattern = False
            sess_opts.enable_cpu_mem_arena = False

            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = {inp.name for inp in sess.get_inputs()}
            self._tokenizer = tok
            self._session = sess
            logger.info("Loaded ONNX quantized embedder %s", self.model_name)

    @property
    def dimension(self) -> int:
        return int(self._dimension or settings.embedding_dim)

    def _encode_batch(self, batch: list[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._session is not None
        encoded = self._tokenizer.encode_batch(batch)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        token_embeddings = self._session.run(None, feed)[0]
        mask_expanded = attention_mask[..., None].astype(np.float32)
        summed = (token_embeddings * mask_expanded).sum(axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        self._dimension = int(pooled.shape[1])
        return _l2_normalize(pooled)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        eff_batch = min(batch_size or settings.embedding_batch_size, 8)
        batches: list[np.ndarray] = []
        for idx in range(0, len(texts), eff_batch):
            chunk = [t if t.strip() else " " for t in texts[idx : idx + eff_batch]]
            batches.append(self._encode_batch(chunk))
        result = np.vstack(batches).astype(np.float32)
        gc.collect()
        return result

    def embed_query(self, query: str) -> np.ndarray:
        q = (query or "").strip()
        if not q:
            return np.zeros((1, self.dimension), dtype=np.float32)
        with self._lock:
            if q in self._query_cache:
                self._query_cache.move_to_end(q)
                return self._query_cache[q]
        vec = self._encode_batch([q]).astype(np.float32)
        with self._lock:
            self._query_cache[q] = vec
            if len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)
        return vec


class GeminiEmbeddingEmbedder:
    """Zero-local-RAM semantic embedder calling Google Gemini Embeddings REST API."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
        cache_size: int | None = None,
    ) -> None:
        raw_model = (model_name or settings.embedding_model or "gemini-embedding-001").strip()
        if "minilm" in raw_model.lower():
            raw_model = "gemini-embedding-001"
        self._api_model = raw_model.replace("models/", "")
        self.model_name = f"gemini:{self._api_model}:{dimension or settings.embedding_dim}"
        self._api_key = api_key or settings.llm_api_key
        self._dimension = dimension or settings.embedding_dim
        self._cache_size = cache_size or settings.embedding_cache_size
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        if not self._api_key:
            raise ValueError("AUTOSPEC_LLM_API_KEY is required when embedding_provider='gemini'.")

    @property
    def dimension(self) -> int:
        return int(self._dimension)

    def _call_batch_api(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        import httpx

        base = (settings.llm_base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base}/v1beta/models/{self._api_model}:batchEmbedContents"
        requests_payload = [
            {
                "model": f"models/{self._api_model}",
                "content": {"parts": [{"text": t[:4000]}]},
                "taskType": task_type,
                "outputDimensionality": self._dimension,
            }
            for t in texts
        ]
        with httpx.Client(timeout=float(settings.request_timeout_seconds)) as client:
            resp = client.post(
                url,
                headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
                json={"requests": requests_payload},
            )
            resp.raise_for_status()
            data = resp.json()
        vectors = [item["values"] for item in data.get("embeddings", [])]
        return _l2_normalize(np.array(vectors, dtype=np.float32))

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        eff_batch = min(batch_size or settings.embedding_batch_size, 8)
        out: list[np.ndarray] = []
        for idx in range(0, len(texts), eff_batch):
            chunk = [t if t.strip() else " " for t in texts[idx : idx + eff_batch]]
            out.append(self._call_batch_api(chunk, task_type="RETRIEVAL_DOCUMENT"))
        return np.vstack(out).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        q = (query or "").strip()
        if not q:
            return np.zeros((1, self.dimension), dtype=np.float32)
        with self._lock:
            if q in self._query_cache:
                self._query_cache.move_to_end(q)
                return self._query_cache[q]
        vec = self._call_batch_api([q], task_type="RETRIEVAL_QUERY").astype(np.float32)
        with self._lock:
            self._query_cache[q] = vec
            if len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)
        return vec


class SentenceTransformerEmbedder:
    """Optional PyTorch SentenceTransformer embedder for local GPU/dev environments only."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._dimension = settings.embedding_dim
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size or settings.embedding_cache_size

    @property
    def model(self):
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
        return int(self._dimension or settings.embedding_dim)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or settings.embedding_batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
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
    """High-speed deterministic mock embedder for offline unit tests ONLY."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self.model_name = f"mock:{dimension}"

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
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.array([self._hash_to_vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._hash_to_vector(query.strip())
        return vec.reshape(1, -1)


class CandidateReranker:
    """Second-stage CPU reranker defaulting to zero-RAM feature reranking on free-tier hosts."""

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
        """Rerank candidate (chunk, rrf_score) pairs using domain features or optional CrossEncoder."""
        if not candidates:
            return []

        ce = self._get_cross_encoder()
        if ce is not None:
            try:
                pairs = [(query, c.content) for c, _ in candidates]
                ce_scores = ce.predict(pairs)
                reranked = []
                for (chunk, base_score), ce_score in zip(candidates, ce_scores, strict=False):
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

            comp_overlap = len(q_tokens & comp_tokens) / max(1, len(comp_tokens)) if comp_tokens else 0.0
            query_cov = len(q_tokens & content_tokens) / max(1, len(q_tokens))
            exact_phrase_bonus = 0.25 if comp_low and len(comp_low) > 5 and comp_low in q_clean else 0.0
            row_bonus = 0.08 if chunk.chunk_type == "spec_row" else 0.0

            final_score = base_score * (
                1.0 + 0.65 * comp_overlap + 0.35 * query_cov + exact_phrase_bonus + row_bonus
            )
            reranked.append((chunk, round(final_score, 6)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


# Bounded shared embedder cache so multiple isolated document pipelines share one model instance
_EMBEDDER_CACHE: dict[str, EmbedderProtocol] = {}
_EMBEDDER_LOCK = threading.Lock()


def resolve_embedding_provider(model_name: str | None = None) -> str:
    """Determine active embedding backend without importing PyTorch on production hosts."""
    name = (model_name or settings.embedding_model).strip().lower()
    explicit = settings.embedding_provider.lower()
    if name in ("mock", "test", "dummy") or explicit == "mock":
        if settings.env == "production":
            raise RuntimeError("Mock embeddings are forbidden when AUTOSPEC_ENV=production.")
        return "mock"
    if explicit != "auto":
        return explicit
    return "onnx"


def get_embedder(model_name: str | None = None) -> EmbedderProtocol:
    """Factory returning a cached lightweight embedder instance."""
    name = (model_name or settings.embedding_model).strip()
    provider = resolve_embedding_provider(name)
    cache_key = f"{provider}:{name}:{settings.embedding_dim}"

    with _EMBEDDER_LOCK:
        if cache_key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[cache_key]

        inst: EmbedderProtocol
        if provider == "mock":
            inst = DeterministicMockEmbedder(dimension=settings.embedding_dim)
        elif provider == "gemini":
            inst = GeminiEmbeddingEmbedder(model_name=name, dimension=settings.embedding_dim)
        elif provider == "sentence_transformers":
            inst = SentenceTransformerEmbedder(model_name=name)
        else:
            inst = OnnxEmbeddingEmbedder(model_name=name)

        _EMBEDDER_CACHE[cache_key] = inst
        return inst
