"""Unit tests for text cleaning, section detection, and table-aware specification chunking."""

from __future__ import annotations

from autospec.chunker import TableAwareChunker
from autospec.cleaner import TextCleaner
from autospec.pdf_parser import ExtractedPage
from autospec.section_detector import SectionDetector


def test_cleaner_removes_footers_preserves_numbers(sample_front_brake_text: str):
    """Test TextCleaner removes URLs and page markers while preserving technical values."""
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(sample_front_brake_text)

    # Footers and URLs removed
    assert "file:///C:/TSO" not in cleaned
    assert "Page 1 sur 2" not in cleaned

    # Essential specifications preserved
    assert "250 184" in cleaned
    assert "1,420 ml" in cleaned
    assert "32 mm (1.259 in)" in cleaned
    assert "SECTION 206-03: Front Disc Brake" in cleaned


def test_section_detector(sample_front_brake_text: str, sample_rear_brake_text: str):
    """Test SectionDetector accurately classifies front vs rear disc brake subsystems."""
    detector = SectionDetector()

    sec_front = detector.detect_in_page(sample_front_brake_text, page_number=636)
    assert sec_front is not None
    assert sec_front.section_id == "206-03"
    assert sec_front.subsystem == "front_brakes"
    assert sec_front.is_spec_section is True

    sec_rear = detector.detect_in_page(sample_rear_brake_text, page_number=652)
    assert sec_rear is not None
    assert sec_rear.section_id == "206-04"
    assert sec_rear.subsystem == "rear_brakes"
    assert sec_rear.is_spec_section is True


def test_table_aware_chunking_front_brakes(sample_front_brake_text: str):
    """Test TableAwareChunker creates structured spec_row chunks with correct units and components."""
    chunker = TableAwareChunker()
    page = ExtractedPage(
        page_number=636,
        raw_text=sample_front_brake_text,
        char_count=len(sample_front_brake_text),
        word_count=len(sample_front_brake_text.split()),
        is_empty=False,
        is_text_poor=False,
    )

    chunks = chunker.chunk_page(page, document_name="test_manual.pdf")
    spec_rows = [c for c in chunks if c.chunk_type == "spec_row"]

    # Verify anchor plate bolt
    anchor_chunk = next((c for c in spec_rows if "anchor plate" in c.component.lower()), None)
    assert anchor_chunk is not None
    assert anchor_chunk.primary_value == "250"
    assert anchor_chunk.primary_unit == "Nm"
    assert anchor_chunk.alternate_value == "184"
    assert anchor_chunk.alternate_unit == "lb-ft"
    assert anchor_chunk.subsystem == "front_brakes"
    assert anchor_chunk.page_number == 636

    # Verify guide pin bolt (front: 37 Nm)
    guide_chunk = next((c for c in spec_rows if "guide pin" in c.component.lower()), None)
    assert guide_chunk is not None
    assert guide_chunk.primary_value == "37"
    assert guide_chunk.primary_unit == "Nm"
    assert guide_chunk.alternate_value == "27"
    assert guide_chunk.alternate_unit == "lb-ft"


def test_table_aware_chunking_rear_brakes(sample_rear_brake_text: str):
    """Test TableAwareChunker preserves rear brake specifications (guide pin: 33 Nm)."""
    chunker = TableAwareChunker()
    page = ExtractedPage(
        page_number=652,
        raw_text=sample_rear_brake_text,
        char_count=len(sample_rear_brake_text),
        word_count=len(sample_rear_brake_text.split()),
        is_empty=False,
        is_text_poor=False,
    )

    chunks = chunker.chunk_page(page, document_name="test_manual.pdf")
    spec_rows = [c for c in chunks if c.chunk_type == "spec_row"]

    # Rear guide pin must be 33 Nm, distinct from front (37 Nm)
    rear_guide = next((c for c in spec_rows if "guide pin" in c.component.lower()), None)
    assert rear_guide is not None
    assert rear_guide.primary_value == "33"
    assert rear_guide.primary_unit == "Nm"
    assert rear_guide.alternate_value == "24"
    assert rear_guide.subsystem == "rear_brakes"
    assert rear_guide.page_number == 652
