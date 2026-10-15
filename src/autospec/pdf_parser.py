"""PDF text extraction with exact page metadata and document fingerprinting."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pymupdf  # PyMuPDF official import

from autospec.config import DocumentValidationError, validate_pdf_bytes, validate_pdf_path


@dataclass
class TextBlock:
    """Represents an individual text block with bounding box coordinates."""

    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    block_type: int  # 0 for text, 1 for image
    block_no: int


@dataclass
class ExtractedPage:
    """Structured representation of a single extracted PDF page."""

    page_number: int  # 1-indexed original PDF page number
    raw_text: str
    char_count: int
    word_count: int
    is_empty: bool
    is_text_poor: bool
    blocks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert page object to dictionary."""
        return asdict(self)


@dataclass
class DocumentMetadata:
    """Document-level metadata and container of extracted pages."""

    filename: str
    fingerprint: str  # SHA-256 hex digest
    total_pages: int
    extracted_page_count: int
    file_size_bytes: int
    title: str = ""
    author: str = ""
    creation_date: str = ""
    pages: list[ExtractedPage] = field(default_factory=list)

    def to_dict(self, include_pages: bool = True) -> dict[str, Any]:
        """Convert document metadata to serializable dictionary."""
        d = {
            "filename": self.filename,
            "fingerprint": self.fingerprint,
            "total_pages": self.total_pages,
            "extracted_page_count": self.extracted_page_count,
            "file_size_bytes": self.file_size_bytes,
            "title": self.title,
            "author": self.author,
            "creation_date": self.creation_date,
        }
        if include_pages:
            d["pages"] = [p.to_dict() for p in self.pages]
        return d


class PDFParser:
    """Robust, streaming-compatible PDF text extractor using PyMuPDF."""

    def __init__(self, text_poor_threshold: int = 50) -> None:
        self.text_poor_threshold = text_poor_threshold

    @staticmethod
    def compute_fingerprint_file(path: Path) -> str:
        """Compute SHA-256 fingerprint for a local PDF file."""
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_fingerprint_bytes(data: bytes) -> str:
        """Compute SHA-256 fingerprint for in-memory PDF bytes."""
        return hashlib.sha256(data).hexdigest()

    def parse_file(
        self,
        path_str: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DocumentMetadata:
        """Parse a local PDF file, preserving original 1-indexed page numbers."""
        pdf_path = validate_pdf_path(path_str)
        fingerprint = self.compute_fingerprint_file(pdf_path)
        file_size = pdf_path.stat().st_size

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as err:
            raise DocumentValidationError(f"PyMuPDF failed to open '{pdf_path.name}': {err}") from err

        return self._extract_from_doc(
            doc=doc,
            filename=pdf_path.name,
            fingerprint=fingerprint,
            file_size=file_size,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
        )

    def parse_bytes(
        self,
        data: bytes,
        filename: str = "upload.pdf",
        start_page: int | None = None,
        end_page: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DocumentMetadata:
        """Parse raw in-memory PDF bytes, preserving 1-indexed page numbers."""
        validate_pdf_bytes(data, filename=filename)
        fingerprint = self.compute_fingerprint_bytes(data)
        file_size = len(data)

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as err:
            raise DocumentValidationError(f"PyMuPDF failed to open stream for '{filename}': {err}") from err

        return self._extract_from_doc(
            doc=doc,
            filename=filename,
            fingerprint=fingerprint,
            file_size=file_size,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
        )

    def _extract_from_doc(
        self,
        doc: pymupdf.Document,
        filename: str,
        fingerprint: str,
        file_size: int,
        start_page: int | None = None,
        end_page: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DocumentMetadata:
        """Internal worker to iterate through PDF pages and collect structured metadata."""
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise DocumentValidationError(f"PDF document '{filename}' contains 0 pages.")

        # Determine 1-indexed page bounds
        s_page = max(1, start_page or 1)
        e_page = min(total_pages, end_page or total_pages)
        if s_page > e_page:
            doc.close()
            raise DocumentValidationError(
                f"Invalid page bounds: start_page ({s_page}) > end_page ({e_page})."
            )

        pdf_meta = doc.metadata or {}
        doc_info = DocumentMetadata(
            filename=filename,
            fingerprint=fingerprint,
            total_pages=total_pages,
            extracted_page_count=0,
            file_size_bytes=file_size,
            title=str(pdf_meta.get("title") or ""),
            author=str(pdf_meta.get("author") or ""),
            creation_date=str(pdf_meta.get("creationDate") or ""),
        )

        extracted_pages: list[ExtractedPage] = []
        total_to_process = e_page - s_page + 1

        try:
            for idx, page_num in enumerate(range(s_page, e_page + 1)):
                page = doc.load_page(page_num - 1)
                text = cast(str, page.get_text("text") or "")
                stripped = text.strip()
                char_count = len(stripped)
                words = stripped.split()
                word_count = len(words)

                # Extract block details for layout awareness
                blocks_raw = cast(list[Any], page.get_text("blocks") or [])
                block_dicts: list[dict[str, Any]] = []
                for b_idx, b in enumerate(blocks_raw):
                    if len(b) >= 5:
                        block_dicts.append({
                            "bbox": (b[0], b[1], b[2], b[3]),
                            "text": b[4],
                            "block_no": b_idx,
                            "type": b[5] if len(b) > 5 else 0,
                        })

                is_empty = char_count == 0
                is_text_poor = char_count < self.text_poor_threshold and not is_empty

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        raw_text=text,
                        char_count=char_count,
                        word_count=word_count,
                        is_empty=is_empty,
                        is_text_poor=is_text_poor,
                        blocks=block_dicts,
                        metadata={
                            "rect": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1],
                            "rotation": page.rotation,
                        },
                    )
                )

                if progress_callback:
                    progress_callback(idx + 1, total_to_process)

        finally:
            doc.close()

        doc_info.pages = extracted_pages
        doc_info.extracted_page_count = len(extracted_pages)
        return doc_info
