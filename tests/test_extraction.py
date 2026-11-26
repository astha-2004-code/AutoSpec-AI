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


def test_gemini_extraction_provider_parses_structured_json(
    monkeypatch, sample_chunks: list[SpecificationChunk]
):
    """Test native Gemini REST JSON response parses through strict Pydantic validation."""
    import json

    class _MockResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            payload = {
                "specifications": [
                    {
                        "component": "Brake caliper anchor plate bolts",
                        "spec_type": "torque",
                        "value": "250",
                        "unit": "Nm",
                        "alternate_value": "184",
                        "alternate_unit": "lb-ft",
                        "section": "SECTION 206-03: Front Disc Brake",
                        "page_number": 636,
                        "source_quote": sample_chunks[0].raw_evidence,
                    }
                ]
            }
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}

    class _MockClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, headers: dict, json: dict) -> _MockResponse:
            assert "gemini-2.5-flash:generateContent" in url
            assert headers.get("x-goog-api-key") == "test-gemini-key"
            return _MockResponse()

    import httpx

    monkeypatch.setattr(httpx, "Client", _MockClient)
    extractor = SpecificationExtractor(
        provider="gemini", model_name="gemini-2.5-flash", api_key="test-gemini-key"
    )
    specs = extractor.extract("anchor plate torque", [(sample_chunks[0], 0.95)])
    assert len(specs) == 1
    assert specs[0].value == "250"
    assert specs[0].unit == "Nm"
    assert specs[0].extraction_method == ExtractionMethod.LLM_EXTRACTION.value

