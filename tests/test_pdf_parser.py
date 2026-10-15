"""Unit tests for PDF parsing, metadata extraction, and runtime document validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from autospec.config import DocumentValidationError, validate_pdf_bytes, validate_pdf_path
from autospec.pdf_parser import PDFParser


def test_validate_pdf_path_success(sample_synthetic_pdf: Path):
    """Test successful path validation on a synthetic PDF."""
    resolved = validate_pdf_path(sample_synthetic_pdf)
    assert resolved.exists()
    assert resolved.is_file()
    assert resolved.suffix.lower() == ".pdf"


def test_validate_pdf_path_nonexistent(temp_dir: Path):
    """Test rejection of non-existent document path."""
    with pytest.raises(DocumentValidationError, match="Document not found"):
        validate_pdf_path(temp_dir / "nonexistent.pdf")


def test_validate_pdf_path_invalid_header(temp_dir: Path):
    """Test rejection of file lacking %PDF- magic header."""
    fake_file = temp_dir / "fake.pdf"
    fake_file.write_text("This is not a real PDF file content.")
    with pytest.raises(DocumentValidationError, match="missing %PDF- header"):
        validate_pdf_path(fake_file)


def test_validate_pdf_bytes_empty():
    """Test rejection of zero-byte data."""
    with pytest.raises(DocumentValidationError, match="empty"):
        validate_pdf_bytes(b"")


def test_validate_pdf_bytes_oversize():
    """Test rejection of data exceeding maximum permitted size."""
    large_data = b"%PDF-1.4 " + (b"0" * (2 * 1024 * 1024))
    with pytest.raises(DocumentValidationError, match="exceeds maximum allowed size"):
        validate_pdf_bytes(large_data, max_size_mb=1)


def test_pdf_parser_extracts_pages_and_fingerprint(sample_synthetic_pdf: Path):
    """Test that PDFParser correctly extracts pages, word counts, and computes fingerprint."""
    parser = PDFParser()
    doc_meta = parser.parse_file(sample_synthetic_pdf)

    assert doc_meta.total_pages == 2
    assert doc_meta.extracted_page_count == 2
    assert len(doc_meta.fingerprint) == 64  # SHA-256 length

    p1 = doc_meta.pages[0]
    assert p1.page_number == 1
    assert "SECTION 206-03: Front Disc Brake" in p1.raw_text
    assert p1.char_count > 100
    assert not p1.is_empty
    assert not p1.is_text_poor

    p2 = doc_meta.pages[1]
    assert p2.page_number == 2
    assert "SECTION 206-04: Rear Disc Brake" in p2.raw_text


def test_pdf_parser_page_range(sample_synthetic_pdf: Path):
    """Test bounded page parsing (start_page and end_page)."""
    parser = PDFParser()
    doc_meta = parser.parse_file(sample_synthetic_pdf, start_page=2, end_page=2)

    assert doc_meta.total_pages == 2
    assert doc_meta.extracted_page_count == 1
    assert doc_meta.pages[0].page_number == 2
    assert "SECTION 206-04" in doc_meta.pages[0].raw_text
