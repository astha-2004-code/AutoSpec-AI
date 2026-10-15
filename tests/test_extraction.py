"""Unit tests for specification extraction, LLM parsing, and deterministic fallback."""

from __future__ import annotations

from autospec.chunker import SpecificationChunk
from autospec.schemas import ExtractionMethod
from autospec.spec_extractor import SpecificationExtractor


def test_deterministic_fallback_spec_row(sample_chunks: list[SpecificationChunk]):
    """Test deterministic fallback extracts structured specifications from spec_row chunks."""
    extractor = SpecificationExtractor(provider="none")
    retrieved = [(sample_chunks[0], 0.95)]  # front anchor plate

    specs = extractor.extract("torque for front brake caliper anchor plate bolts", retrieved)
    assert len(specs) == 1
    spec = specs[0]

    assert spec.component == "Brake caliper anchor plate bolts"
    assert spec.value == "250"
    assert spec.unit == "Nm"
    assert spec.alternate_value == "184"
    assert spec.alternate_unit == "lb-ft"
    assert spec.page_number == 636
    assert spec.extraction_method == ExtractionMethod.DETERMINISTIC_FALLBACK.value


def test_deterministic_fallback_empty_retrieval():
    """Test extractor returns empty list when no chunks are retrieved (zero hallucination)."""
    extractor = SpecificationExtractor(provider="none")
    specs = extractor.extract("unknown component torque", [])
    assert specs == []


def test_deterministic_fallback_passage_parsing():
    """Test fallback regex parsing on narrative text passages."""
    extractor = SpecificationExtractor(provider="none")
    passage_chunk = SpecificationChunk(
        chunk_id="p_test",
        document_name="test.pdf",
        page_number=640,
        section_id="206-03",
        section_title="Front Disc Brake",
        subsystem="front_brakes",
        chunk_type="text_passage",
        content="To install, tighten the new bolts to 250 Nm (184 lb-ft).",
        raw_evidence="To install, tighten the new bolts to 250 Nm (184 lb-ft).",
    )

    specs = extractor.extract("torque", [(passage_chunk, 0.8)])
    assert len(specs) == 1
    assert specs[0].value == "250"
    assert specs[0].unit == "Nm"
    assert specs[0].alternate_value == "184"
    assert specs[0].alternate_unit == "lb-ft"
