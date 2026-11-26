"""PDF extraction with layout bounding boxes, table detection, OCR fallback, and fingerprinting."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pymupdf  # PyMuPDF official import

from autospec.config import DocumentValidationError, settings, validate_pdf_bytes, validate_pdf_path

logger = logging.getLogger(__name__)


def is_ocr_available() -> bool:
    """Return True if pytesseract + PIL and a working tesseract binary are available."""
    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]

        _ = Image
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@dataclass
class TextBlock:
    """Represents an individual text block with bounding box coordinates."""

    text: str
    bbox: tuple[float, float, float, float]
    block_type: int
    block_no: int


@dataclass
class ExtractedPage:
    """Structured representation of a single extracted PDF page with coordinates and labels."""

    page_number: int  # 1-indexed physical PDF page number
    raw_text: str
    char_count: int
    word_count: int
    is_empty: bool
    is_text_poor: bool
    printed_page_label: str = ""
    page_width: float = 612.0
    page_height: float = 792.0
    rotation: int = 0
    ocr_used: bool = False
    is_duplicate_content: bool = False
    content_hash: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
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
    ocr_page_count: int = 0
    duplicate_page_count: int = 0
    pages: list[ExtractedPage] = field(default_factory=list)

    def to_dict(self, include_pages: bool = True) -> dict[str, Any]:
        """Convert document metadata to serializable dictionary."""
        d: dict[str, Any] = {
            "filename": self.filename,
            "fingerprint": self.fingerprint,
            "total_pages": self.total_pages,
            "extracted_page_count": self.extracted_page_count,
            "file_size_bytes": self.file_size_bytes,
            "title": self.title,
            "author": self.author,
            "creation_date": self.creation_date,
            "ocr_page_count": self.ocr_page_count,
            "duplicate_page_count": self.duplicate_page_count,
        }
        if include_pages:
            d["pages"] = [p.to_dict() for p in self.pages]
        return d


_PRINTED_PAGE_RE = re.compile(
    r"(?:^|\n)\s*(?:Page\s+([A-Za-z0-9\-]+)(?:\s+(?:of|sur)\s+\d+)?|([0-9]{3}-[0-9]{2}-[0-9]+))\s*$",
    re.IGNORECASE,
)


class PDFParser:
    """Streaming-compatible PDF extractor with coordinate capture, table detection, and optional OCR."""

    def __init__(self, text_poor_threshold: int = 40, enable_ocr: bool | None = None) -> None:
        self.text_poor_threshold = text_poor_threshold
        self.enable_ocr = settings.enable_ocr if enable_ocr is None else enable_ocr

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
        cancel_check: Callable[[], bool] | None = None,
    ) -> DocumentMetadata:
        """Parse a local PDF file, preserving original 1-indexed physical page numbers."""
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
            cancel_check=cancel_check,
        )

    def parse_bytes(
        self,
        data: bytes,
        filename: str = "upload.pdf",
        start_page: int | None = None,
        end_page: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> DocumentMetadata:
        """Parse raw in-memory PDF bytes, preserving 1-indexed physical page numbers."""
        validate_pdf_bytes(data, filename=filename, verify_structure=False)
        fingerprint = self.compute_fingerprint_bytes(data)
        file_size = len(data)

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as err:
            raise DocumentValidationError(f"PyMuPDF failed to open stream for '{filename}': {err}") from err

        if doc.is_encrypted:
            doc.close()
            raise DocumentValidationError(f"File '{filename}' is password-encrypted.")
        if len(doc) > settings.max_pdf_pages:
            total = len(doc)
            doc.close()
            raise DocumentValidationError(
                f"File '{filename}' has {total} pages, exceeding limit ({settings.max_pdf_pages})."
            )

        return self._extract_from_doc(
            doc=doc,
            filename=filename,
            fingerprint=fingerprint,
            file_size=file_size,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _run_ocr_on_page(self, page: pymupdf.Page) -> tuple[str, list[dict[str, Any]]]:
        """Run optional pytesseract OCR on a rendered page image when text layer is missing."""
        if not self.enable_ocr:
            return "", []
        try:
            import io

            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]

            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img) or ""
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            scale_x = page.rect.width / float(pix.width or 1)
            scale_y = page.rect.height / float(pix.height or 1)

            line_groups: dict[tuple[int, int, int], list[tuple[str, float, float, float, float]]] = {}
            n = len(data.get("text", []))
            for idx in range(n):
                w = str(data["text"][idx]).strip()
                if not w:
                    continue
                key = (
                    int(data["block_num"][idx]),
                    int(data["par_num"][idx]),
                    int(data["line_num"][idx]),
                )
                x0 = data["left"][idx] * scale_x
                y0 = data["top"][idx] * scale_y
                x1 = (data["left"][idx] + data["width"][idx]) * scale_x
                y1 = (data["top"][idx] + data["height"][idx]) * scale_y
                line_groups.setdefault(key, []).append((w, x0, y0, x1, y1))

            lines: list[dict[str, Any]] = []
            for items in line_groups.values():
                line_txt = " ".join(it[0] for it in items)
                x0 = min(it[1] for it in items)
                y0 = min(it[2] for it in items)
                x1 = max(it[3] for it in items)
                y1 = max(it[4] for it in items)
                lines.append({"text": line_txt, "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]})

            return text, lines
        except Exception as err:
            logger.debug("OCR fallback skipped/unavailable: %s", err)
            return "", []

    def _extract_page_lines(self, page: pymupdf.Page) -> list[dict[str, Any]]:
        """Extract line-level text and exact PDF bounding boxes."""
        lines: list[dict[str, Any]] = []
        try:
            text_dict = cast(dict[str, Any], page.get_text("dict") or {})
            for block in text_dict.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(str(s.get("text", "")) for s in spans).strip()
                    bbox = line.get("bbox")
                    if line_text and bbox and len(bbox) == 4:
                        lines.append({
                            "text": line_text,
                            "bbox": [round(float(c), 2) for c in bbox],
                        })
        except Exception as err:
            logger.debug("Line bbox extraction fallback: %s", err)
        return lines

    def _extract_page_tables(self, page: pymupdf.Page) -> list[dict[str, Any]]:
        """Detect structured tables and row bounding boxes using PyMuPDF find_tables."""
        tables_out: list[dict[str, Any]] = []
        try:
            finder = page.find_tables()
            for t_idx, table in enumerate(finder.tables):
                rows_raw = table.extract() or []
                if not rows_raw:
                    continue
                cleaned_rows: list[list[str]] = []
                # Forward-fill merged cells horizontally/vertically only for header or label columns
                for row in rows_raw:
                    cleaned_rows.append([
                        re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row
                    ])
                row_bboxes: list[list[float] | None] = []
                if hasattr(table, "rows") and table.rows:
                    for r_obj in table.rows:
                        if hasattr(r_obj, "bbox") and r_obj.bbox:
                            row_bboxes.append([round(float(c), 2) for c in r_obj.bbox])
                        else:
                            row_bboxes.append(None)

                tables_out.append({
                    "table_index": t_idx,
                    "bbox": [round(float(c), 2) for c in table.bbox] if table.bbox else None,
                    "rows": cleaned_rows,
                    "row_bboxes": row_bboxes,
                })
        except Exception as err:
            logger.debug("Table detection skipped on page: %s", err)
        return tables_out

    def _detect_printed_label(self, page: pymupdf.Page, text: str, page_num: int) -> str:
        """Determine printed manual page label vs physical PDF page index."""
        try:
            label = (page.get_label() or "").strip()
            if label and label != str(page_num):
                return label
        except Exception:
            pass

        # Inspect top/bottom lines for manual printed page stamps (e.g. 'Page M-14', '206-03-2')
        m = _PRINTED_PAGE_RE.search(text.strip())
        if m:
            found = (m.group(1) or m.group(2) or "").strip()
            if found and found != str(page_num):
                return found
        return str(page_num)

    def _extract_from_doc(
        self,
        doc: pymupdf.Document,
        filename: str,
        fingerprint: str,
        file_size: int,
        start_page: int | None = None,
        end_page: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> DocumentMetadata:
        """Iterate through PDF pages and collect structured text, tables, and bounding boxes."""
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise DocumentValidationError(f"PDF document '{filename}' contains 0 pages.")

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
        seen_hashes: set[str] = set()
        total_to_process = e_page - s_page + 1

        try:
            for idx, page_num in enumerate(range(s_page, e_page + 1)):
                if cancel_check and cancel_check():
                    raise InterruptedError("PDF ingestion cancelled by user.")

                page = doc.load_page(page_num - 1)
                text = cast(str, page.get_text("text") or "")
                lines = self._extract_page_lines(page)
                ocr_used = False

                if len(text.strip()) < self.text_poor_threshold and self.enable_ocr:
                    ocr_text, ocr_lines = self._run_ocr_on_page(page)
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        lines = ocr_lines or lines
                        ocr_used = True
                        doc_info.ocr_page_count += 1

                stripped = text.strip()
                char_count = len(stripped)
                word_count = len(stripped.split())

                norm_for_hash = re.sub(r"\s+", " ", stripped.lower())
                content_hash = hashlib.sha256(norm_for_hash.encode("utf-8")).hexdigest()[:16] if norm_for_hash else ""
                is_dup = bool(content_hash and len(norm_for_hash) > 80 and content_hash in seen_hashes)
                if content_hash and len(norm_for_hash) > 80:
                    seen_hashes.add(content_hash)
                if is_dup:
                    doc_info.duplicate_page_count += 1

                blocks_raw = cast(list[Any], page.get_text("blocks") or [])
                block_dicts: list[dict[str, Any]] = []
                for b_idx, b in enumerate(blocks_raw):
                    if len(b) >= 5:
                        block_dicts.append({
                            "bbox": [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2), round(float(b[3]), 2)],
                            "text": b[4],
                            "block_no": b_idx,
                            "type": b[5] if len(b) > 5 else 0,
                        })

                tables = self._extract_page_tables(page)
                printed_label = self._detect_printed_label(page, text, page_num)

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
                        printed_page_label=printed_label,
                        page_width=round(float(page.rect.width), 2),
                        page_height=round(float(page.rect.height), 2),
                        rotation=int(page.rotation or 0),
                        ocr_used=ocr_used,
                        is_duplicate_content=is_dup,
                        content_hash=content_hash,
                        blocks=block_dicts,
                        lines=lines,
                        tables=tables,
                        metadata={
                            "rect": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1],
                            "rotation": page.rotation,
                            "printed_label": printed_label,
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

    @staticmethod
    def find_evidence_bboxes(
        page: ExtractedPage | dict[str, Any],
        component: str,
        value: str,
        quote: str = "",
    ) -> list[dict[str, Any]]:
        """Locate real PDF bounding boxes matching a component/value/quote on a page. Never fabricates."""
        lines = page.lines if isinstance(page, ExtractedPage) else page.get("lines", [])
        pw = page.page_width if isinstance(page, ExtractedPage) else float(page.get("page_width", 612.0))
        ph = page.page_height if isinstance(page, ExtractedPage) else float(page.get("page_height", 792.0))

        if not lines:
            return []

        comp_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9\-]{3,}", component or "")]
        val_clean = (value or "").replace(",", "").strip().lower()
        quote_norm = re.sub(r"\s+", " ", (quote or "").lower()).strip()

        matched: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            ltxt = str(line.get("text", ""))
            l_low = ltxt.lower()
            l_clean = l_low.replace(",", "")
            bbox = line.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            # Exact or strong token overlap with component or quote
            token_hits = sum(1 for t in comp_tokens if t in l_low)
            has_val = bool(val_clean and val_clean in l_clean)
            in_quote = bool(quote_norm and len(l_low) > 6 and l_low in quote_norm)

            if (comp_tokens and token_hits >= max(1, len(comp_tokens) // 2) and (has_val or token_hits >= 2)) or (in_quote and has_val):
                matched.append({
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "page_width": pw,
                    "page_height": ph,
                })
                # If value sits on the immediately following line in multi-line tables, include it too
                if not has_val and val_clean and idx + 1 < len(lines):
                    nxt = lines[idx + 1]
                    nxt_bbox = nxt.get("bbox")
                    if nxt_bbox and val_clean in str(nxt.get("text", "")).replace(",", "").lower():
                        matched.append({
                            "x0": nxt_bbox[0],
                            "y0": nxt_bbox[1],
                            "x1": nxt_bbox[2],
                            "y1": nxt_bbox[3],
                            "page_width": pw,
                            "page_height": ph,
                        })

        # Deduplicate identical boxes
        seen: set[tuple[float, float, float, float]] = set()
        unique: list[dict[str, Any]] = []
        for b in matched:
            key = (b["x0"], b["y0"], b["x1"], b["y1"])
            if key not in seen:
                seen.add(key)
                unique.append(b)
        return unique[:6]
