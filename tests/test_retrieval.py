"""Unit tests for BM25 lexical search, FAISS vector search, query routing, and hybrid fusion."""

from __future__ import annotations

from pathlib import Path

from autospec.chunker import SpecificationChunk
from autospec.embeddings import DeterministicMockEmbedder
from autospec.hybrid_retriever import HybridRetriever
from autospec.keyword_retriever import AutomotiveTokenizer, BM25KeywordRetriever
from autospec.query_router import QueryRouter
from autospec.vector_store import FAISSVectorStore


def test_automotive_tokenizer():
    """Test AutomotiveTokenizer preserves hyphenated part numbers and units."""
    tokens = AutomotiveTokenizer.tokenize("Tighten the WSS-M6C62-A fluid to 35 Nm (26 lb-ft)")
    assert "wss-m6c62-a" in tokens
    assert "wss" in tokens
    assert "35" in tokens
    assert "nm" in tokens
    assert "lb-ft" in tokens
    assert "the" not in tokens  # Stopword removed


def test_bm25_retriever(sample_chunks: list[SpecificationChunk]):
    """Test BM25KeywordRetriever retrieves exact component match."""
    retriever = BM25KeywordRetriever()
    retriever.build_index(sample_chunks)

    results = retriever.search("brake caliper anchor plate bolts", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert "anchor plate" in top_chunk.component.lower()
    assert top_chunk.primary_value == "250"
    assert score > 0.0


def test_faiss_vector_store_persistence(temp_dir: Path, sample_chunks: list[SpecificationChunk], mock_embedder: DeterministicMockEmbedder):
    """Test FAISSVectorStore building, saving, loading, and searching."""
    vs = FAISSVectorStore(index_dir=temp_dir, embedding_dim=mock_embedder.dimension)
    texts = [c.content for c in sample_chunks]
    embeddings = mock_embedder.embed_texts(texts)

    fingerprint = "test_fp_123"
    vs.build_index(fingerprint, "test_manual.pdf", sample_chunks, embeddings)

    assert vs.is_fresh(fingerprint) is True
    assert vs.is_fresh("different_fp") is False

    # Search
    q_vec = mock_embedder.embed_query("anchor plate bolts")
    hits = vs.search(q_vec, top_k=2)
    assert len(hits) == 2

    # Reload from disk
    vs_reloaded = FAISSVectorStore(index_dir=temp_dir, embedding_dim=mock_embedder.dimension)
    loaded = vs_reloaded.load_index(fingerprint)
    assert loaded is True
    assert len(vs_reloaded.chunks) == 3


def test_query_router():
    """Test QueryRouter accurately extracts subsystem, component, and spec types."""
    router = QueryRouter()

    # Front brake query
    q1 = router.parse_query("What is the torque for front brake caliper guide pin bolts?")
    assert q1.target_subsystem == "front_brakes"
    assert q1.spec_type == "torque"
    assert any("guide pin" in h.lower() for h in q1.component_hints)

    # Rear brake query
    q2 = router.parse_query("Torque for rear caliper guide pin bolts in lb-ft")
    assert q2.target_subsystem == "rear_brakes"
    assert q2.spec_type == "torque"
    assert q2.unit_hint == "lb-ft"

    # Capacity query
    q3 = router.parse_query("What is the brake fluid fill capacity?")
    assert q3.spec_type in ("capacity", "fluid_capacity")


def test_hybrid_retriever_front_vs_rear_isolation(temp_dir: Path, sample_chunks: list[SpecificationChunk], mock_embedder: DeterministicMockEmbedder):
    """Test that HybridRetriever isolates front and rear brake guide pin bolts."""
    vs = FAISSVectorStore(index_dir=temp_dir, embedding_dim=mock_embedder.dimension)
    bm25 = BM25KeywordRetriever()

    texts = [c.content for c in sample_chunks]
    embeddings = mock_embedder.embed_texts(texts)
    vs.build_index("test_fp", "test.pdf", sample_chunks, embeddings)
    bm25.build_index(sample_chunks)

    hybrid = HybridRetriever(
        vector_store=vs,
        keyword_retriever=bm25,
        embedder=mock_embedder,
    )

    # Query for rear guide pin must return 33 Nm from rear_brakes
    rear_hits = hybrid.retrieve("Torque for rear brake caliper guide pin bolts", top_k=1)
    assert len(rear_hits) == 1
    top_rear, _ = rear_hits[0]
    assert top_rear.subsystem == "rear_brakes"
    assert top_rear.primary_value == "33"
    assert top_rear.page_number == 652

    # Query for front guide pin must return 37 Nm from front_brakes
    front_hits = hybrid.retrieve("Torque for front brake caliper guide pin bolts", top_k=1)
    assert len(front_hits) == 1
    top_front, _ = front_hits[0]
    assert top_front.subsystem == "front_brakes"
    assert top_front.primary_value == "37"
    assert top_front.page_number == 636
