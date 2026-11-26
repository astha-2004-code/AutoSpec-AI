"""Table-aware specification chunking for automotive workshop manuals.

Supports:
1. PyMuPDF structured tables (including multi-page tables & merged cells).
2. Multi-line OEM torque, material/fluid, dimension/clearance, electrical, and pressure tables.
3. Overlapping procedural text passages with preserved section hierarchy and real PDF bounding boxes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from autospec.cleaner import TextCleaner
from autospec.pdf_parser import ExtractedPage, PDFParser
from autospec.section_detector import SectionDetector
from autospec.units import normalize_unit_symbol, parse_numeric_or_range


def _deterministic_id(*parts: Any) -> str:
    """Create a stable, process-independent short digest for chunk IDs."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


@dataclass
class SpecificationChunk:
    """A semantically enriched, evidence-grounded chunk for hybrid retrieval."""

    chunk_id: str
    document_name: str
    page_number: int
    section_id: str
    section_title: str
    subsystem: str
    chunk_type: str  # 'spec_row', 'spec_table', 'text_passage'
    content: str
    raw_evidence: str
    document_id: str = ""
    printed_page_label: str = ""
    table_name: str = ""
    component: str = ""
    primary_value: str = ""
    primary_unit: str = ""
    alternate_value: str = ""
    alternate_unit: str = ""
    range_min: str = ""
    range_max: str = ""
    tolerance: str = ""
    conditions: str = ""
    part_number: str = ""
    vehicle_make: str = ""
    vehicle_model: str = ""
    vehicle_year: int | None = None
    evidence_bboxes: list[dict[str, Any]] = field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return asdict(self)


class TableAwareChunker:
    """Extracts specification rows, multi-page tables, and narrative passages from manual pages."""

    def __init__(self, cleaner: TextCleaner | None = None, section_detector: SectionDetector | None = None) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.section_detector = section_detector or SectionDetector()

    def chunk_document(
        self,
        pages: list[ExtractedPage],
        document_name: str,
        document_id: str = "",
        vehicle_make: str = "",
        vehicle_model: str = "",
        vehicle_year: int | None = None,
    ) -> list[SpecificationChunk]:
        """Process all extracted document pages into structured chunks with multi-page table continuity."""
        # Reset detector for document-level isolation
        detector = SectionDetector()
        all_chunks: list[SpecificationChunk] = []
        pending_table_headers: list[str] | None = None

        for page in pages:
            page_chunks, pending_table_headers = self._chunk_page_internal(
                page=page,
                document_name=document_name,
                document_id=document_id,
                detector=detector,
                pending_headers=pending_table_headers,
                vehicle_make=vehicle_make,
                vehicle_model=vehicle_model,
                vehicle_year=vehicle_year,
            )
            all_chunks.extend(page_chunks)

        return all_chunks

    def chunk_page(
        self,
        page: ExtractedPage,
        document_name: str,
        document_id: str = "",
    ) -> list[SpecificationChunk]:
        """Chunk a single page using table-aware extraction (backward-compatible signature)."""
        chunks, _ = self._chunk_page_internal(
            page=page,
            document_name=document_name,
            document_id=document_id,
            detector=self.section_detector,
            pending_headers=None,
        )
        return chunks

    def _chunk_page_internal(
        self,
        page: ExtractedPage,
        document_name: str,
        document_id: str,
        detector: SectionDetector,
        pending_headers: list[str] | None,
        vehicle_make: str = "",
        vehicle_model: str = "",
        vehicle_year: int | None = None,
    ) -> tuple[list[SpecificationChunk], list[str] | None]:
        cleaned_text = self.cleaner.clean_text(page.raw_text)
        if not cleaned_text or page.is_empty:
            return [], pending_headers

        section_info = detector.detect_in_page(cleaned_text, page.page_number)
        sec_id = section_info.section_id if section_info else "GENERAL"
        sec_title = section_info.title if section_info else "General Information"
        subsystem = section_info.subsystem if section_info else "general"
        printed_label = page.printed_page_label or str(page.page_number)

        chunks: list[SpecificationChunk] = []
        seen_row_sigs: set[tuple[str, str, str]] = set()
        next_pending_headers: list[str] | None = None

        # 1. Extract from PyMuPDF structured tables first (supports complex/merged/multi-page tables)
        if page.tables:
            t_chunks, next_pending_headers = self._parse_structured_pdf_tables(
                page=page,
                doc_name=document_name,
                doc_id=document_id,
                sec_id=sec_id,
                sec_title=sec_title,
                subsystem=subsystem,
                printed_label=printed_label,
                pending_headers=pending_headers,
                vehicle_make=vehicle_make,
                vehicle_model=vehicle_model,
                vehicle_year=vehicle_year,
            )
            for ch in t_chunks:
                sig = (ch.component.lower(), ch.primary_value.lower(), ch.primary_unit.lower())
                if sig not in seen_row_sigs:
                    seen_row_sigs.add(sig)
                    chunks.append(ch)

        # 2. Text-layout table extractors (fallback when PyMuPDF structured tables didn't cover the page)
        if (section_info and section_info.is_spec_section or self._looks_like_spec_page(cleaned_text)) and not page.tables:
            text_chunks: list[SpecificationChunk] = []
            text_chunks.extend(
                self._parse_torque_tables(cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem)
            )
            text_chunks.extend(
                self._parse_material_tables(cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem)
            )
            text_chunks.extend(
                self._parse_dimension_tables(cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem)
            )
            text_chunks.extend(
                self._parse_general_kv_specs(cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem)
            )

            for ch in text_chunks:
                sig = (ch.component.lower(), ch.primary_value.lower(), ch.primary_unit.lower())
                if sig in seen_row_sigs:
                    continue
                seen_row_sigs.add(sig)
                ch.document_id = document_id
                ch.printed_page_label = printed_label
                ch.vehicle_make = vehicle_make
                ch.vehicle_model = vehicle_model
                ch.vehicle_year = vehicle_year
                self._enrich_chunk_ranges_and_bboxes(ch, page)
                chunks.append(ch)
        elif (section_info and section_info.is_spec_section or self._looks_like_spec_page(cleaned_text)) and page.tables:
            # Still run general kv extractor for non-table colon lines on pages with tables
            for ch in self._parse_general_kv_specs(cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem):
                sig = (ch.component.lower(), ch.primary_value.lower(), ch.primary_unit.lower())
                if sig in seen_row_sigs:
                    continue
                seen_row_sigs.add(sig)
                ch.document_id = document_id
                ch.printed_page_label = printed_label
                ch.vehicle_make = vehicle_make
                ch.vehicle_model = vehicle_model
                ch.vehicle_year = vehicle_year
                self._enrich_chunk_ranges_and_bboxes(ch, page)
                chunks.append(ch)

        # 3. Generate overlapping procedural/context text passages (full page coverage)
        passages = self._generate_page_passages(
            cleaned_text,
            page,
            document_name,
            document_id,
            sec_id,
            sec_title,
            subsystem,
            printed_label,
            vehicle_make,
            vehicle_model,
            vehicle_year,
        )
        chunks.extend(passages)

        return chunks, next_pending_headers

    @staticmethod
    def _looks_like_spec_page(text: str) -> bool:
        low = text.lower()
        return any(
            m in low
            for m in ("torque", "specifications", "capacity", "clearance", "minimum thickness", "voltage", "pressure")
        )

    def _enrich_chunk_ranges_and_bboxes(self, ch: SpecificationChunk, page: ExtractedPage) -> None:
        """Populate range_min/range_max, normalized units, and real PDF coordinate bounding boxes."""
        if ch.primary_value and not ch.range_min:
            parsed = parse_numeric_or_range(ch.primary_value)
            if parsed.is_range:
                ch.range_min = f"{parsed.range_min:g}" if parsed.range_min is not None else ""
                ch.range_max = f"{parsed.range_max:g}" if parsed.range_max is not None else ""
                if parsed.tolerance is not None:
                    ch.tolerance = f"±{parsed.tolerance:g}"
        if not ch.evidence_bboxes and page.lines:
            ch.evidence_bboxes = PDFParser.find_evidence_bboxes(
                page, component=ch.component, value=ch.primary_value, quote=ch.raw_evidence
            )

    def _parse_structured_pdf_tables(
        self,
        page: ExtractedPage,
        doc_name: str,
        doc_id: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
        printed_label: str,
        pending_headers: list[str] | None,
        vehicle_make: str,
        vehicle_model: str,
        vehicle_year: int | None,
    ) -> tuple[list[SpecificationChunk], list[str] | None]:
        """Parse PyMuPDF-detected tables, propagating merged cells and multi-page headers."""
        chunks: list[SpecificationChunk] = []
        last_headers: list[str] | None = None

        for t_idx, table in enumerate(page.tables):
            rows: list[list[str]] = table.get("rows", [])
            row_bboxes: list[list[float] | None] = table.get("row_bboxes", [])
            if not rows or len(rows[0]) < 2:
                continue

            header_row = [c.strip() for c in rows[0]]
            header_joined = " ".join(header_row).lower()

            # Skip Table of Contents / Document Control index tables
            if "printed page" in header_joined or "section code" in header_joined or "manual section" in header_joined:
                continue

            is_header = any(
                k in header_joined
                for k in ("component", "description", "item", "parameter", "specification", "nm", "lb-ft", "value", "unit", "capacity")
            )

            data_start_idx = 1 if is_header else 0
            active_headers = header_row if is_header else (pending_headers or [])
            if active_headers:
                last_headers = active_headers

            # Carry forward merged category/variant cell in column 0 if needed
            prev_col0 = ""
            for r_i in range(data_start_idx, len(rows)):
                row = rows[r_i]
                if not any(cell.strip() for cell in row):
                    continue
                # Skip repeated subheader rows inside multi-page tables
                row_join_low = " ".join(row).lower()
                if active_headers and row_join_low == " ".join(active_headers).lower():
                    continue

                comp = row[0].strip()
                if not comp and prev_col0:
                    comp = prev_col0
                elif comp:
                    prev_col0 = comp

                if not comp or len(comp) < 2:
                    continue

                # Determine primary / alternate values and units from row + headers
                spec_type, val, unit, alt_val, alt_unit, part_no, material, conditions = self._interpret_table_row(
                    row, active_headers
                )
                if not val and not part_no:
                    continue

                raw_ev = " | ".join(c for c in row if c)
                bboxes: list[dict[str, Any]] = []
                if r_i < len(row_bboxes) and row_bboxes[r_i]:
                    rb = row_bboxes[r_i]
                    if rb and len(rb) == 4:
                        bboxes.append({
                            "x0": rb[0],
                            "y0": rb[1],
                            "x1": rb[2],
                            "y1": rb[3],
                            "page_width": page.page_width,
                            "page_height": page.page_height,
                        })
                if not bboxes and page.lines:
                    bboxes = PDFParser.find_evidence_bboxes(page, comp, val or part_no, raw_ev)

                parsed_q = parse_numeric_or_range(val)
                alt_str = f" / {alt_val} {alt_unit}".strip() if alt_val else ""
                vehicle_prefix = " ".join(str(x) for x in (vehicle_year, vehicle_make, vehicle_model) if x).strip()
                content = (
                    f"[{sec_title} (Section {sec_id}, Page {page.page_number}) | {spec_type.replace('_', ' ').title()}] "
                    f"{f'Vehicle: {vehicle_prefix} | ' if vehicle_prefix else ''}"
                    f"Component: {comp} | Value: {val} {unit}{alt_str} | Subsystem: {subsystem}"
                    f"{f' | Conditions: {conditions}' if conditions else ''}"
                )

                chunk_id = f"{doc_name}_p{page.page_number}_tbl{t_idx}_{_deterministic_id(comp, val, unit, r_i)}"
                chunks.append(
                    SpecificationChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        document_name=doc_name,
                        page_number=page.page_number,
                        printed_page_label=printed_label,
                        section_id=sec_id,
                        section_title=sec_title,
                        subsystem=subsystem,
                        chunk_type="spec_row",
                        content=content,
                        raw_evidence=raw_ev,
                        table_name=" | ".join(h for h in active_headers if h) or "Structured Table",
                        component=comp,
                        primary_value=val or part_no,
                        primary_unit=unit or ("part_number" if part_no else ""),
                        alternate_value=alt_val,
                        alternate_unit=alt_unit,
                        range_min=f"{parsed_q.range_min:g}" if parsed_q.range_min is not None else "",
                        range_max=f"{parsed_q.range_max:g}" if parsed_q.range_max is not None else "",
                        tolerance=f"±{parsed_q.tolerance:g}" if parsed_q.tolerance is not None else "",
                        conditions=conditions,
                        part_number=part_no,
                        vehicle_make=vehicle_make,
                        vehicle_model=vehicle_model,
                        vehicle_year=vehicle_year,
                        evidence_bboxes=bboxes,
                        metadata={
                            "spec_type": spec_type,
                            "ford_spec": material,
                            "part_number": part_no,
                            "table_headers": active_headers,
                        },
                    )
                )

        # If a table reached near bottom of page, carry its headers to next page
        carry = last_headers if page.tables and (page.tables[-1].get("bbox") or [0, 0, 0, 0])[3] > page.page_height * 0.72 else None
        return chunks, carry

    @staticmethod
    def _interpret_table_row(
        row: list[str],
        headers: list[str],
    ) -> tuple[str, str, str, str, str, str, str, str]:
        """Infer spec_type, primary/alternate value+unit, part_number, material, and conditions from a row."""
        h_low = [h.lower() for h in headers]
        val = ""
        unit = ""
        alt_val = ""
        alt_unit = ""
        part_no = ""
        material = ""
        conditions = ""
        spec_type = "general"

        # Check explicit torque column layout (e.g. Description | Nm | lb-ft | lb-in)
        if any("nm" in h for h in h_low):
            spec_type = "torque"
            for idx in range(1, len(row)):
                cell = row[idx].strip()
                if not cell or cell in ("—", "-", "N/A"):
                    continue
                hdr = h_low[idx] if idx < len(h_low) else ""
                if "nm" in hdr and not val:
                    val, unit = re.sub(r"\s*nm$", "", cell, flags=re.I).strip(), "Nm"
                elif ("lb-ft" in hdr or "ft-lb" in hdr) and not alt_val:
                    alt_val, alt_unit = re.sub(r"\s*(?:lb-ft|ft-lb)$", "", cell, flags=re.I).strip(), "lb-ft"
                elif ("lb-in" in hdr or "in-lb" in hdr) and not alt_val:
                    alt_val, alt_unit = re.sub(r"\s*(?:lb-in|in-lb)$", "", cell, flags=re.I).strip(), "lb-in"
                elif "condition" in hdr or "note" in hdr or "variant" in hdr:
                    conditions = cell

        # Generic column parsing
        if not val:
            for idx in range(1, len(row)):
                cell = row[idx].strip()
                if not cell or cell in ("—", "-"):
                    continue
                hdr = h_low[idx] if idx < len(h_low) else ""

                if "condition" in hdr or "variant" in hdr or "application" in hdr or "note" in hdr:
                    conditions = cell
                    continue
                if "part" in hdr:
                    part_no = cell
                    if not val:
                        spec_type = "part_number"
                    continue
                if "fluid" in hdr or "grade" in hdr or "material" in hdr or "standard" in hdr:
                    material = cell
                    continue

                # Extract value + unit embedded in cell
                m = re.match(
                    r"^([-+]?[0-9,\.]+(?:\s*(?:-|to|±|\+/-)\s*[0-9,\.]+)?)\s*([A-Za-z°%]+(?:-[A-Za-z]+)?)?(?:\s*\(([^)]+)\))?$",
                    cell,
                    re.I,
                )
                if m and not val:
                    val = m.group(1).strip()
                    unit = normalize_unit_symbol(m.group(2) or "")
                    if not unit and hdr:
                        for token in ("nm", "lb-ft", "lb-in", "mm", "in", "ml", "l", "qt", "pt", "kpa", "psi", "bar", "v", "a", "cca", "ah", "rpm", "°c", "°f"):
                            if re.search(rf"\b{re.escape(token)}\b", hdr):
                                unit = normalize_unit_symbol(token)
                                break
                    if m.group(3):
                        m_alt = re.match(r"([-+]?[0-9,\.]+(?:\s*-\s*[0-9,\.]+)?)\s*([A-Za-z°%/-]+)", m.group(3).strip())
                        if m_alt:
                            alt_val = m_alt.group(1).strip()
                            alt_unit = normalize_unit_symbol(m_alt.group(2))
                elif m and not alt_val:
                    alt_val = m.group(1).strip()
                    alt_unit = normalize_unit_symbol(m.group(2) or "")
                elif not val and len(cell) <= 80:
                    val = cell

        if spec_type == "general":
            u_low = unit.lower()
            if u_low in ("nm", "lb-ft", "lb-in", "kgf-m"):
                spec_type = "torque"
            elif u_low in ("ml", "l", "pt", "qt", "gal", "fl-oz"):
                spec_type = "fluid_capacity"
            elif u_low in ("mm", "cm", "in", "mil"):
                spec_type = "dimension_clearance"
            elif u_low in ("v", "mv", "a", "ma", "cca", "ah", "ohm", "kohm"):
                spec_type = "electrical"
            elif u_low in ("kpa", "bar", "psi", "mpa"):
                spec_type = "pressure"
            elif part_no and not val:
                spec_type = "part_number"

        return spec_type, val, unit, alt_val, alt_unit, part_no, material, conditions

    def _parse_torque_tables(
        self,
        text: str,
        page_num: int,
        doc_name: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
    ) -> list[SpecificationChunk]:
        """Extract individual torque rows with primary Nm and alternate lb-ft / lb-in values."""
        chunks: list[SpecificationChunk] = []

        if "nm" not in text.lower() or ("description" not in text.lower() and "torque" not in text.lower()):
            return chunks

        lines = text.splitlines()
        in_torque_table = False
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if re.search(r"^\s*Description\s*$", line, re.I) or re.search(r"Nm\s+lb-ft", line, re.I):
                in_torque_table = True
                i += 1
                while i < len(lines) and any(u in lines[i].lower() for u in ("nm", "lb-ft", "lb-in")):
                    i += 1
                continue

            if in_torque_table:
                if re.match(r"^(SECTION|Material|Item|Procedure revision date|General Specifications)", line, re.I):
                    in_torque_table = False
                    i += 1
                    continue

                if not line or line in ("—", "-"):
                    i += 1
                    continue

                desc = line
                i += 1
                tokens: list[str] = []

                multi_match = re.search(r"^(.*?)\s+(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s+(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?|—|-)(?:\s+(\d+(?:\.\d+)?|—|-))?$", desc)
                single_trailing_num = re.search(r"^(.*?\D)\s+(\d+(?:\.\d+)?)$", desc)

                if multi_match:
                    desc = multi_match.group(1).strip()
                    tokens = [multi_match.group(2)]
                    if multi_match.group(3):
                        tokens.append(multi_match.group(3))
                    if multi_match.group(4):
                        tokens.append(multi_match.group(4))
                elif single_trailing_num:
                    desc = single_trailing_num.group(1).strip()
                    tokens = [single_trailing_num.group(2)]
                    while i < len(lines):
                        nxt = lines[i].strip()
                        if not nxt:
                            i += 1
                            continue
                        num_parts = nxt.split()
                        if all(re.match(r"^(\d+(?:\.\d+)?|—|-)$", p) for p in num_parts):
                            tokens.extend(num_parts)
                            i += 1
                            if len(tokens) >= 3:
                                break
                        else:
                            break
                else:
                    while i < len(lines):
                        nxt = lines[i].strip()
                        if not nxt:
                            i += 1
                            continue
                        num_parts = nxt.split()
                        if all(re.match(r"^(\d+(?:\.\d+)?|—|-)$", p) for p in num_parts):
                            tokens.extend(num_parts)
                            i += 1
                            if len(tokens) >= 2:
                                break
                        else:
                            break

                if tokens and len(tokens) >= 1:
                    primary_nm = tokens[0] if tokens[0] not in ("—", "-") else ""
                    alt_lbft = tokens[1] if len(tokens) > 1 and tokens[1] not in ("—", "-") else ""
                    alt_lbin = tokens[2] if len(tokens) > 2 and tokens[2] not in ("—", "-") else ""

                    val_str_parts = []
                    if primary_nm:
                        val_str_parts.append(f"{primary_nm} Nm")
                    if alt_lbft:
                        val_str_parts.append(f"{alt_lbft} lb-ft")
                    if alt_lbin:
                        val_str_parts.append(f"{alt_lbin} lb-in")

                    val_str = " / ".join(val_str_parts) if val_str_parts else "N/A"
                    content = (
                        f"[{sec_title} (Section {sec_id}, Page {page_num}) | Torque Specification] "
                        f"Component: {desc} | Torque: {val_str} | Subsystem: {subsystem}"
                    )
                    evidence = f"{desc} -> {val_str}"
                    chunk_id = f"{doc_name}_p{page_num}_tr_{_deterministic_id(desc, val_str)}"
                    chunks.append(
                        SpecificationChunk(
                            chunk_id=chunk_id,
                            document_name=doc_name,
                            page_number=page_num,
                            section_id=sec_id,
                            section_title=sec_title,
                            subsystem=subsystem,
                            chunk_type="spec_row",
                            content=content,
                            raw_evidence=evidence,
                            table_name="Torque Specifications",
                            component=desc,
                            primary_value=primary_nm,
                            primary_unit="Nm" if primary_nm else "",
                            alternate_value=alt_lbft or alt_lbin,
                            alternate_unit="lb-ft" if alt_lbft else ("lb-in" if alt_lbin else ""),
                            metadata={
                                "spec_type": "torque",
                                "nm": primary_nm,
                                "lb_ft": alt_lbft,
                                "lb_in": alt_lbin,
                            },
                        )
                    )
                continue

            i += 1

        return chunks

    def _parse_material_tables(
        self,
        text: str,
        page_num: int,
        doc_name: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
    ) -> list[SpecificationChunk]:
        """Extract fluid capacities, lubricants, and part numbers."""
        chunks: list[SpecificationChunk] = []

        if "material" not in text.lower() and "fill capacity" not in text.lower() and "capacity" not in text.lower():
            return chunks

        mat_match = re.search(
            r"Material\s*\n\s*Item.*?(?=(?:Item\s*\n\s*Specification|Description\s*\n\s*Nm|SECTION|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if mat_match:
            block = mat_match.group(0)
            raw_lines = [line.strip() for line in block.splitlines() if line.strip()]
            headers = {"material", "item", "specification", "fill", "capacity"}
            content_lines = [line for line in raw_lines if line.lower() not in headers and line not in ("—", "-", "")]

            entries: list[dict[str, str]] = []
            curr_item: list[str] = []
            curr_parts: list[str] = []
            curr_specs: list[str] = []
            curr_caps: list[str] = []
            state = "item"

            for line in content_lines:
                if re.search(r"\b(?:PM-[0-9]-[A-Z]|CPM-[0-9]|XL-[0-9]|XT-[0-9]|XG-[0-9])", line, re.I):
                    state = "part"
                    curr_parts.append(line)
                elif re.search(r"\b(?:WSS-[A-Z0-9\-]+|WSK-[A-Z0-9\-]+|ESB-[A-Z0-9\-]+)", line, re.I):
                    state = "spec"
                    curr_specs.append(line)
                elif re.search(r"[0-9,]+(?:\.[0-9]+)?\s*(?:ml|l|pt|qt|gal)", line, re.I):
                    state = "cap"
                    curr_caps.append(line)
                elif line.startswith("(") and line.endswith(")") and state == "part":
                    curr_parts.append(line)
                elif line.startswith("(") and line.endswith(")") and state == "cap":
                    curr_caps.append(line)
                elif state in ("spec", "cap") and ("WSS-" in line or "or" in line):
                    curr_specs.append(line)
                else:
                    if state != "item":
                        entries.append({
                            "item": " ".join(curr_item),
                            "part_no": " ".join(curr_parts),
                            "spec": " ".join(curr_specs),
                            "capacity": " ".join(curr_caps),
                        })
                        curr_item = [line]
                        curr_parts = []
                        curr_specs = []
                        curr_caps = []
                        state = "item"
                    else:
                        curr_item.append(line)

            if curr_item:
                entries.append({
                    "item": " ".join(curr_item),
                    "part_no": " ".join(curr_parts),
                    "spec": " ".join(curr_specs),
                    "capacity": " ".join(curr_caps),
                })

            for entry in entries:
                item_name = entry["item"].strip()
                p_no = entry["part_no"].strip()
                s_code = entry["spec"].strip()
                cap_val = entry["capacity"].strip()

                if not item_name or item_name in ("Item", "Material"):
                    continue

                if cap_val:
                    c_content = (
                        f"[{sec_title} (Section {sec_id}, Page {page_num}) | Fluid Capacity] "
                        f"Item: {item_name} | Fill Capacity: {cap_val} | Part Number: {p_no} | Spec: {s_code} | Subsystem: {subsystem}"
                    )
                    c_raw = f"{item_name} | {cap_val} | {p_no} | {s_code}"
                    chunk_id = f"{doc_name}_p{page_num}_matcap_{_deterministic_id(item_name, cap_val)}"
                    chunks.append(
                        SpecificationChunk(
                            chunk_id=chunk_id,
                            document_name=doc_name,
                            page_number=page_num,
                            section_id=sec_id,
                            section_title=sec_title,
                            subsystem=subsystem,
                            chunk_type="spec_row",
                            content=c_content,
                            raw_evidence=c_raw,
                            table_name="Material Specifications",
                            component=item_name,
                            part_number=p_no,
                            primary_value=cap_val,
                            primary_unit="capacity",
                            metadata={
                                "spec_type": "fluid_capacity",
                                "ford_spec": s_code,
                                "part_number": p_no,
                                "capacity": cap_val,
                            },
                        )
                    )

                if p_no:
                    p_content = (
                        f"[{sec_title} (Section {sec_id}, Page {page_num}) | Part Number Specification] "
                        f"Item: {item_name} | Part Number: {p_no} | Spec: {s_code} | Fill Capacity: {cap_val} | Subsystem: {subsystem}"
                    )
                    p_raw = f"{item_name} | {p_no} | {s_code} | {cap_val}"
                    p_id = f"{doc_name}_p{page_num}_matpart_{_deterministic_id(item_name, p_no)}"
                    chunks.append(
                        SpecificationChunk(
                            chunk_id=p_id,
                            document_name=doc_name,
                            page_number=page_num,
                            section_id=sec_id,
                            section_title=sec_title,
                            subsystem=subsystem,
                            chunk_type="spec_row",
                            content=p_content,
                            raw_evidence=p_raw,
                            table_name="Material Specifications",
                            component=item_name,
                            part_number=p_no,
                            primary_value=p_no,
                            primary_unit="part_number",
                            metadata={
                                "spec_type": "part_number",
                                "ford_spec": s_code,
                                "part_number": p_no,
                                "capacity": cap_val,
                            },
                        )
                    )

        fluid_capacity_re = re.compile(
            r"([A-Za-z0-9\s®\-\(\)]+?)\s+([A-Z0-9]{2,4}-[A-Z0-9\-]+(?:\s*\([A-Za-z]+\))?)\s+([A-Z0-9\-]+(?: or [A-Z0-9\-]+)?)\s+([0-9\.,]+\s*(?:ml|l|pt|qt|gal)[^\n\r]*)",
            re.IGNORECASE,
        )
        for match in fluid_capacity_re.finditer(text):
            item = match.group(1).strip()
            part_no = match.group(2).strip()
            spec_code = match.group(3).strip()
            capacity = match.group(4).strip()
            content = (
                f"[{sec_title} (Section {sec_id}, Page {page_num}) | Fluid & Material Specification] "
                f"Item: {item} | Part Number: {part_no} | Spec: {spec_code} | Fill Capacity: {capacity} | Subsystem: {subsystem}"
            )
            raw_ev = f"{item} | {part_no} | {spec_code} | {capacity}"
            chunk_id = f"{doc_name}_p{page_num}_mat_{_deterministic_id(item, capacity)}"
            chunks.append(
                SpecificationChunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    page_number=page_num,
                    section_id=sec_id,
                    section_title=sec_title,
                    subsystem=subsystem,
                    chunk_type="spec_row",
                    content=content,
                    raw_evidence=raw_ev,
                    table_name="Material Specifications",
                    component=item,
                    part_number=part_no,
                    primary_value=capacity,
                    primary_unit="capacity",
                    metadata={
                        "spec_type": "fluid_capacity",
                        "ford_spec": spec_code,
                        "part_number": part_no,
                        "capacity": capacity,
                    },
                )
            )

        cap_block_re = re.compile(
            r"([0-9\.,]+\s*(?:ml|l|liters?|pt|qt|gal)(?:\s*\([^\)]+\))?)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            if any(term in line.lower() for term in ("brake fluid", "lubricant", "fluid", "oil", "coolant", "grease")) and any(
                unit in line.lower() for unit in ("ml", " pt", " qt", "gal", " l ", " l(")
            ):
                cap_match = cap_block_re.search(line)
                if cap_match:
                    cap_val = cap_match.group(1)
                    comp = line.split(cap_val)[0].strip(" :|-") or "Vehicle Fluid"
                    content = (
                        f"[{sec_title} (Section {sec_id}, Page {page_num}) | Capacity] "
                        f"Line: {line.strip()} | Capacity: {cap_val} | Subsystem: {subsystem}"
                    )
                    chunk_id = f"{doc_name}_p{page_num}_cap_{_deterministic_id(line)}"
                    chunks.append(
                        SpecificationChunk(
                            chunk_id=chunk_id,
                            document_name=doc_name,
                            page_number=page_num,
                            section_id=sec_id,
                            section_title=sec_title,
                            subsystem=subsystem,
                            chunk_type="spec_row",
                            content=content,
                            raw_evidence=line.strip(),
                            table_name="Capacities",
                            component=comp,
                            primary_value=cap_val,
                            primary_unit="volume",
                            metadata={"spec_type": "fluid_capacity"},
                        )
                    )

        return chunks

    def _parse_dimension_tables(
        self,
        text: str,
        page_num: int,
        doc_name: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
    ) -> list[SpecificationChunk]:
        """Extract dimensions, minimum thicknesses, runout, tolerances, and backlash specifications."""
        chunks: list[SpecificationChunk] = []
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        i = 0

        while i < len(raw_lines):
            line = raw_lines[i]
            m_same = re.match(
                r"^(.*?)\s+([0-9\.\-]+(?:\s*-\s*[0-9\.]+)?(?:\s*(?:±|\+/-)\s*[0-9\.]+)?)\s*mm(?:\s*\(([0-9\.\-]+(?:\s*-\s*[0-9\.]+)?)\s*in\))?$",
                line,
                re.IGNORECASE,
            )
            if m_same and m_same.group(1):
                comp = m_same.group(1).strip(" :|-")
                val_mm = m_same.group(2).strip()
                val_in = m_same.group(3).strip() if m_same.group(3) else ""

                content = (
                    f"[{sec_title} (Section {sec_id}, Page {page_num}) | Dimension Specification] "
                    f"Component: {comp} | Minimum/Clearance: {val_mm} mm"
                    f"{f' ({val_in} in)' if val_in else ''} | Subsystem: {subsystem}"
                )
                evidence = f"{comp} {val_mm} mm" + (f" ({val_in} in)" if val_in else "")
                chunk_id = f"{doc_name}_p{page_num}_dim_{_deterministic_id(comp, val_mm)}"
                chunks.append(
                    SpecificationChunk(
                        chunk_id=chunk_id,
                        document_name=doc_name,
                        page_number=page_num,
                        section_id=sec_id,
                        section_title=sec_title,
                        subsystem=subsystem,
                        chunk_type="spec_row",
                        content=content,
                        raw_evidence=evidence,
                        table_name="Dimension & Clearance Specifications",
                        component=comp,
                        primary_value=val_mm,
                        primary_unit="mm",
                        alternate_value=val_in,
                        alternate_unit="in" if val_in else "",
                        metadata={"spec_type": "dimension_clearance", "mm": val_mm, "in": val_in},
                    )
                )
                i += 1
                continue

            if i + 1 < len(raw_lines):
                next_line = raw_lines[i + 1]
                m_next = re.match(
                    r"^([0-9\.\-]+(?:\s*-\s*[0-9\.]+)?)\s*mm(?:\s*\(([0-9\.\-]+(?:\s*-\s*[0-9\.]+)?)\s*in\))?$",
                    next_line,
                    re.IGNORECASE,
                )
                if m_next and not re.match(r"^(Item|Specification|Description|SECTION|Procedure|General)", line, re.I):
                    comp = line
                    val_mm = m_next.group(1).strip()
                    val_in = m_next.group(2).strip() if m_next.group(2) else ""

                    content = (
                        f"[{sec_title} (Section {sec_id}, Page {page_num}) | Dimension Specification] "
                        f"Component: {comp} | Minimum/Clearance: {val_mm} mm"
                        f"{f' ({val_in} in)' if val_in else ''} | Subsystem: {subsystem}"
                    )
                    evidence = f"{comp} {val_mm} mm" + (f" ({val_in} in)" if val_in else "")
                    chunk_id = f"{doc_name}_p{page_num}_dim_{_deterministic_id(comp, val_mm)}"
                    chunks.append(
                        SpecificationChunk(
                            chunk_id=chunk_id,
                            document_name=doc_name,
                            page_number=page_num,
                            section_id=sec_id,
                            section_title=sec_title,
                            subsystem=subsystem,
                            chunk_type="spec_row",
                            content=content,
                            raw_evidence=evidence,
                            table_name="Dimension & Clearance Specifications",
                            component=comp,
                            primary_value=val_mm,
                            primary_unit="mm",
                            alternate_value=val_in,
                            alternate_unit="in" if val_in else "",
                            metadata={"spec_type": "dimension_clearance", "mm": val_mm, "in": val_in},
                        )
                    )
                    i += 2
                    continue

            i += 1

        return chunks

    def _parse_general_kv_specs(
        self,
        text: str,
        page_num: int,
        doc_name: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
    ) -> list[SpecificationChunk]:
        """Extract general colon/pipe-delimited electrical, pressure, and maintenance specifications."""
        chunks: list[SpecificationChunk] = []
        kv_re = re.compile(
            r"^([A-Za-z0-9\s\-/()]{4,70}?)\s*(?::|\|)\s*"
            r"([-+]?[0-9,\.]+(?:\s*(?:-|to|±|\+/-)\s*[0-9,\.]+)?)\s*"
            r"(kPa|bar|psi|MPa|V|mV|A|mA|CCA|Ah|Ohm|kOhm|rpm|deg|°C|°F|L|ml|qt|mm|Nm|lb-ft)\b"
            r"(?:\s*\(([^)]+)\))?",
            re.I,
        )
        for line in text.splitlines():
            m = kv_re.match(line.strip())
            if not m:
                continue
            comp = m.group(1).strip()
            if comp.lower() in ("vehicle identity", "engine code", "transmission", "transaxle", "high-voltage system"):
                continue
            val = m.group(2).strip()
            unit = normalize_unit_symbol(m.group(3))
            alt_raw = (m.group(4) or "").strip()
            alt_val, alt_unit = "", ""
            if alt_raw:
                m_alt = re.match(r"([-+]?[0-9,\.]+(?:\s*-\s*[0-9,\.]+)?)\s*([A-Za-z°%/-]+)", alt_raw)
                if m_alt:
                    alt_val = m_alt.group(1).strip()
                    alt_unit = normalize_unit_symbol(m_alt.group(2))

            if unit in ("kPa", "bar", "psi", "MPa"):
                stype = "pressure"
            elif unit in ("V", "mV", "A", "mA", "CCA", "Ah", "Ohm", "kOhm"):
                stype = "electrical"
            elif unit in ("Nm", "lb-ft"):
                stype = "torque"
            elif unit in ("L", "ml", "qt"):
                stype = "fluid_capacity"
            elif unit in ("mm",):
                stype = "dimension_clearance"
            else:
                stype = "maintenance"

            content = (
                f"[{sec_title} (Section {sec_id}, Page {page_num}) | {stype.title()}] "
                f"Component: {comp} | Value: {val} {unit} | Subsystem: {subsystem}"
            )
            chunk_id = f"{doc_name}_p{page_num}_kv_{_deterministic_id(comp, val, unit)}"
            chunks.append(
                SpecificationChunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    page_number=page_num,
                    section_id=sec_id,
                    section_title=sec_title,
                    subsystem=subsystem,
                    chunk_type="spec_row",
                    content=content,
                    raw_evidence=line.strip(),
                    table_name="Technical Specifications",
                    component=comp,
                    primary_value=val,
                    primary_unit=unit,
                    alternate_value=alt_val,
                    alternate_unit=alt_unit,
                    metadata={"spec_type": stype},
                )
            )
        return chunks

    def _generate_page_passages(
        self,
        cleaned_text: str,
        page: ExtractedPage,
        doc_name: str,
        doc_id: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
        printed_label: str,
        vehicle_make: str,
        vehicle_model: str,
        vehicle_year: int | None,
    ) -> list[SpecificationChunk]:
        """Create overlapping context passages covering the entire page for procedural & semantic queries."""
        stripped = cleaned_text.strip()
        if len(stripped) < 40:
            return []

        window = 1200
        overlap = 200
        passages: list[SpecificationChunk] = []

        if len(stripped) <= window:
            slices = [(0, stripped)]
        else:
            slices = []
            pos = 0
            while pos < len(stripped):
                end = min(len(stripped), pos + window)
                slices.append((pos, stripped[pos:end]))
                if end >= len(stripped):
                    break
                pos += window - overlap

        for idx, (_offset, snippet) in enumerate(slices):
            suffix = "" if idx == 0 else f"_{idx}"
            chunk_id = f"{doc_name}_p{page.page_number}_passage{suffix}"
            vehicle_prefix = " ".join(str(x) for x in (vehicle_year, vehicle_make, vehicle_model) if x).strip()
            content = (
                f"[{sec_title} (Section {sec_id}, Page {page.page_number}) | Procedural & Overview] "
                f"{f'Vehicle: {vehicle_prefix} | ' if vehicle_prefix else ''}"
                f"Subsystem: {subsystem}\n{snippet}"
            )
            bboxes = PDFParser.find_evidence_bboxes(page, "", "", snippet[:240]) if page.lines else []
            passages.append(
                SpecificationChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_name=doc_name,
                    page_number=page.page_number,
                    printed_page_label=printed_label,
                    section_id=sec_id,
                    section_title=sec_title,
                    subsystem=subsystem,
                    chunk_type="text_passage",
                    content=content,
                    raw_evidence=snippet,
                    vehicle_make=vehicle_make,
                    vehicle_model=vehicle_model,
                    vehicle_year=vehicle_year,
                    evidence_bboxes=bboxes,
                    metadata={"full_length": len(stripped), "slice_index": idx},
                )
            )
        return passages
