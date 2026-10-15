"""Hybrid retrieval engine combining dense vector search, sparse BM25, and query routing."""

from __future__ import annotations

from autospec.chunker import SpecificationChunk
from autospec.config import settings
from autospec.embeddings import EmbedderProtocol, get_embedder
from autospec.keyword_retriever import BM25KeywordRetriever
from autospec.query_router import ParsedQuery, QueryRouter
from autospec.vector_store import FAISSVectorStore


class HybridRetriever:
    """Combines semantic search and BM25 lexical search using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        keyword_retriever: BM25KeywordRetriever,
        embedder: EmbedderProtocol | None = None,
        query_router: QueryRouter | None = None,
        rrf_k: int | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever
        self.embedder = embedder or get_embedder()
        self.query_router = query_router or QueryRouter()
        self.rrf_k = rrf_k or settings.rrf_k
        self.semantic_weight = semantic_weight if semantic_weight is not None else settings.semantic_weight
        self.keyword_weight = keyword_weight if keyword_weight is not None else settings.keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        subsystem_filter: str | None = None,
    ) -> list[tuple[SpecificationChunk, float]]:
        """Perform hybrid retrieval using RRF and specification-aware re-ranking."""
        if not query or not query.strip():
            return []

        # 1. Parse and interpret query
        parsed = self.query_router.parse_query(query)
        effective_subsystem = subsystem_filter or parsed.target_subsystem

        # Retrieve a broader candidate pool (top_k * 4) for fusion
        candidate_k = max(20, top_k * 4)

        # 2. Dense Semantic Search (FAISS)
        dense_results: list[tuple[SpecificationChunk, float]] = []
        try:
            query_vec = self.embedder.embed_query(query)
            dense_results = self.vector_store.search(
                query_vec,
                top_k=candidate_k,
                subsystem_filter=effective_subsystem if parsed.is_exact else None,
            )
        except Exception:
            dense_results = []

        # 3. Sparse Lexical Search (BM25)
        lexical_results: list[tuple[SpecificationChunk, float]] = []
        try:
            lexical_results = self.keyword_retriever.search(
                query,
                top_k=candidate_k,
                subsystem_filter=effective_subsystem if parsed.is_exact else None,
            )
        except Exception:
            lexical_results = []

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF score = w_dense / (k + rank_dense) + w_lexical / (k + rank_lexical)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SpecificationChunk] = {}

        for rank, (chunk, _) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            score = self.semantic_weight / (self.rrf_k + rank)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + score

        for rank, (chunk, _) in enumerate(lexical_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            score = self.keyword_weight / (self.rrf_k + rank)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + score

        if not rrf_scores:
            return []

        # 5. Specification-aware boosting and filtering
        final_scores: list[tuple[SpecificationChunk, float]] = []
        for chunk_id, base_score in rrf_scores.items():
            chunk = chunk_map[chunk_id]
            multiplier = 1.0

            # (A) Subsystem alignment boost/penalty
            if effective_subsystem:
                if chunk.subsystem == effective_subsystem:
                    multiplier *= 1.4  # Strong boost for matching target subsystem
                elif (
                    ("front" in effective_subsystem and "rear" in chunk.subsystem)
                    or ("rear" in effective_subsystem and "front" in chunk.subsystem)
                ):
                    multiplier *= 0.1  # Severe penalty for cross-brake confusion!

            # (B) Component hint boost
            for hint in parsed.component_hints:
                if hint.lower() in chunk.component.lower() or hint.lower() in chunk.content.lower():
                    multiplier *= 1.3

            # (C) Spec-type match boost
            if parsed.spec_type and chunk.metadata.get("spec_type") == parsed.spec_type:
                multiplier *= 1.2

            # (D) Structured spec_row priority boost
            if chunk.chunk_type == "spec_row":
                multiplier *= 1.4

            final_scores.append((chunk, base_score * multiplier))

        # 6. Sort and Deduplicate
        final_scores.sort(key=lambda x: x[1], reverse=True)

        seen_keys: set[str] = set()
        deduped_results: list[tuple[SpecificationChunk, float]] = []

        for chunk, score in final_scores:
            # Deduplicate by component + section + primary_value
            dedup_key = f"{chunk.subsystem}_{chunk.section_id}_{chunk.component.lower().strip()}_{chunk.primary_value}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped_results.append((chunk, score))

            if len(deduped_results) >= top_k:
                break

        return deduped_results
