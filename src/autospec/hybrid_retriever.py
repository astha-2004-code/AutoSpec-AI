"""Hybrid retrieval engine combining dense vector search, sparse BM25, RRF, and reranking."""

from __future__ import annotations

import logging
from typing import Literal

from autospec.chunker import SpecificationChunk
from autospec.config import settings
from autospec.embeddings import CandidateReranker, EmbedderProtocol, get_embedder
from autospec.keyword_retriever import BM25KeywordRetriever
from autospec.query_router import QueryRouter
from autospec.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class HybridRetriever:
    """Combines dense FAISS search, sparse BM25, RRF fusion, metadata filtering, and second-stage reranking."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        keyword_retriever: BM25KeywordRetriever,
        embedder: EmbedderProtocol | None = None,
        query_router: QueryRouter | None = None,
        reranker: CandidateReranker | None = None,
        rrf_k: int | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever
        self.embedder = embedder or get_embedder()
        self.query_router = query_router or QueryRouter()
        self.reranker = reranker or CandidateReranker()
        self.rrf_k = rrf_k or settings.rrf_k
        self.semantic_weight = semantic_weight if semantic_weight is not None else settings.semantic_weight
        self.keyword_weight = keyword_weight if keyword_weight is not None else settings.keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        subsystem_filter: str | None = None,
        document_ids: set[str] | None = None,
        manufacturer_filter: str | None = None,
        model_filter: str | None = None,
        year_filter: int | None = None,
        retrieval_mode: RetrievalMode = "hybrid_rerank",
        min_score: float | None = None,
    ) -> list[tuple[SpecificationChunk, float]]:
        """Perform retrieval using the configured strategy with filtering, thresholds, and deduplication."""
        if not query or not query.strip():
            return []

        parsed = self.query_router.parse_query(query)
        norm_query = parsed.normalized_query or query.strip()

        effective_subsystem = subsystem_filter or parsed.target_subsystem
        eff_make = manufacturer_filter or parsed.vehicle_make_hint
        eff_model = model_filter or parsed.vehicle_model_hint
        eff_year = year_filter if year_filter is not None else parsed.vehicle_year_hint

        # Adaptive top-K: broaden candidate pool when procedural or multi-variant queries need context
        effective_k = top_k
        if parsed.is_comparison or parsed.is_procedural:
            effective_k = min(20, max(top_k, top_k * 2))
        candidate_k = max(50, effective_k * 8)

        # Single-strategy modes for ablation and baseline benchmarking
        if retrieval_mode == "dense":
            try:
                query_vec = self.embedder.embed_query(norm_query)
                dense_only = self.vector_store.search(
                    query_vec,
                    top_k=candidate_k,
                    subsystem_filter=effective_subsystem if parsed.is_exact else subsystem_filter,
                    document_ids=document_ids,
                    manufacturer_filter=eff_make,
                    model_filter=eff_model,
                    year_filter=eff_year,
                )
                return self._deduplicate(dense_only, effective_k, min_score=0.0)
            except Exception as err:
                logger.warning("Dense retrieval failed: %s", err)
                return []

        if retrieval_mode == "bm25":
            try:
                bm25_only = self.keyword_retriever.search(
                    norm_query,
                    top_k=candidate_k,
                    subsystem_filter=effective_subsystem if parsed.is_exact else subsystem_filter,
                    document_ids=document_ids,
                    manufacturer_filter=eff_make,
                    model_filter=eff_model,
                    year_filter=eff_year,
                )
                return self._deduplicate(bm25_only, effective_k, min_score=0.0)
            except Exception as err:
                logger.warning("BM25 retrieval failed: %s", err)
                return []

        # Hybrid Retrieval (Dense + BM25)
        dense_results: list[tuple[SpecificationChunk, float]] = []
        try:
            query_vec = self.embedder.embed_query(norm_query)
            dense_results = self.vector_store.search(
                query_vec,
                top_k=candidate_k,
                subsystem_filter=effective_subsystem if parsed.is_exact else subsystem_filter,
                document_ids=document_ids,
                manufacturer_filter=eff_make,
                model_filter=eff_model,
                year_filter=eff_year,
            )
        except Exception as err:
            logger.warning("Dense vector search failed during hybrid retrieval: %s", err)
            dense_results = []

        lexical_results: list[tuple[SpecificationChunk, float]] = []
        try:
            lexical_results = self.keyword_retriever.search(
                norm_query,
                top_k=candidate_k,
                subsystem_filter=effective_subsystem if parsed.is_exact else subsystem_filter,
                document_ids=document_ids,
                manufacturer_filter=eff_make,
                model_filter=eff_model,
                year_filter=eff_year,
            )
        except Exception as err:
            logger.warning("BM25 search failed during hybrid retrieval: %s", err)
            lexical_results = []

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SpecificationChunk] = {}

        for rank, (chunk, _) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                self.semantic_weight / (self.rrf_k + rank)
            )

        for rank, (chunk, _) in enumerate(lexical_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                self.keyword_weight / (self.rrf_k + rank)
            )

        if not rrf_scores:
            return []

        # Domain-aware boosting and cross-assembly penalty
        scored: list[tuple[SpecificationChunk, float]] = []
        for chunk_id, base_score in rrf_scores.items():
            chunk = chunk_map[chunk_id]
            multiplier = 1.0

            if effective_subsystem:
                if chunk.subsystem == effective_subsystem:
                    multiplier *= 1.45
                elif (
                    ("front" in effective_subsystem and "rear" in chunk.subsystem)
                    or ("rear" in effective_subsystem and "front" in chunk.subsystem)
                    or ("4wd" in effective_subsystem and "rwd" in chunk.subsystem)
                    or ("rwd" in effective_subsystem and "4wd" in chunk.subsystem)
                ):
                    multiplier *= 0.08

            for hint in parsed.component_hints:
                if hint.lower() in chunk.component.lower():
                    multiplier *= 1.5
                elif hint.lower() in chunk.content.lower():
                    multiplier *= 1.2

            if parsed.spec_type:
                chunk_spec_type = chunk.metadata.get("spec_type")
                if chunk_spec_type == parsed.spec_type:
                    multiplier *= 1.6
                elif chunk_spec_type and chunk_spec_type != parsed.spec_type:
                    # Normalize legacy "dimension" vs "dimension_clearance"
                    if not (
                        {chunk_spec_type, parsed.spec_type} <= {"dimension", "dimension_clearance"}
                    ):
                        multiplier *= 0.6

            if parsed.is_procedural and chunk.chunk_type == "text_passage":
                multiplier *= 1.5
            elif not parsed.is_procedural and chunk.chunk_type == "spec_row":
                multiplier *= 1.4

            scored.append((chunk, base_score * multiplier))

        scored.sort(key=lambda x: x[1], reverse=True)

        if retrieval_mode == "hybrid_rerank" and settings.enable_reranker:
            rerank_pool = scored[: max(20, effective_k * 4)]
            scored = self.reranker.rerank(norm_query, rerank_pool)

        eff_threshold = min_score if min_score is not None else settings.min_retrieval_score
        return self._deduplicate(scored, effective_k, min_score=eff_threshold)

    @staticmethod
    def _deduplicate(
        candidates: list[tuple[SpecificationChunk, float]],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[tuple[SpecificationChunk, float]]:
        seen_keys: set[str] = set()
        deduped: list[tuple[SpecificationChunk, float]] = []

        for chunk, score in candidates:
            if score < min_score:
                continue
            dedup_key = (
                f"{chunk.document_id}_{chunk.document_name}_{chunk.subsystem}_{chunk.section_id}_"
                f"{chunk.component.lower().strip()}_{chunk.primary_value}_{chunk.primary_unit}"
            )
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped.append((chunk, round(float(score), 6)))
            if len(deduped) >= top_k:
                break

        return deduped
