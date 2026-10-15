"""Table-aware specification chunking for automotive workshop manuals.

Deconstructs workshop manual pages into semantically enriched chunks:
1. Individual structured specification row chunks (component, value, primary unit, alternate unit).
2. Complete table block chunks preserving headers and column relationships.
3. Text passage chunks for procedural context.

Ensures section and page metadata remain permanently bound to every technical value.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from autospec.cleaner import TextCleaner
from autospec.pdf_parser import ExtractedPage
from autospec.section_detector import SectionDetector, SectionInfo


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
    content: str  # Enriched text for embedding and lexical matching
    raw_evidence: str  # Exact original text excerpt
    table_name: str = ""
    component: str = ""
    primary_value: str = ""
    primary_unit: str = ""
    alternate_value: str = ""
    alternate_unit: str = ""
    part_number: str = ""
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return asdict(self)


class TableAwareChunker:
    """Extracts specification rows, tables, and narrative passages from manual pages."""

    def __init__(self, cleaner: TextCleaner | None = None, section_detector: SectionDetector | None = None) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.section_detector = section_detector or SectionDetector()

    def chunk_document(
        self,
        pages: list[ExtractedPage],
        document_name: str,
    ) -> list[SpecificationChunk]:
        """Process all extracted document pages into structured chunks."""
        all_chunks: list[SpecificationChunk] = []

        for page in pages:
            page_chunks = self.chunk_page(page, document_name)
            all_chunks.extend(page_chunks)

        return all_chunks

    def chunk_page(
        self,
        page: ExtractedPage,
        document_name: str,
    ) -> list[SpecificationChunk]:
        """Chunk a single page using table-aware extraction."""
        cleaned_text = self.cleaner.clean_text(page.raw_text)
        if not cleaned_text or page.is_empty:
            return []

        section_info = self.section_detector.detect_in_page(cleaned_text, page.page_number)
        sec_id = section_info.section_id if section_info else "GENERAL"
        sec_title = section_info.title if section_info else "General Information"
        subsystem = section_info.subsystem if section_info else "general"

        chunks: list[SpecificationChunk] = []

        # If page contains specifications, perform table extraction
        if section_info and section_info.is_spec_section:
            # 1. Parse Torque Specifications
            torque_chunks = self._parse_torque_tables(
                cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem
            )
            chunks.extend(torque_chunks)

            # 2. Parse Material / Fluid / Capacity Specifications
            material_chunks = self._parse_material_tables(
                cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem
            )
            chunks.extend(material_chunks)

            # 3. Parse Dimension / Clearance Specifications
            dimension_chunks = self._parse_dimension_tables(
                cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem
            )
            chunks.extend(dimension_chunks)

        # 4. Generate general section passage chunk for surrounding context
        passage_chunk = self._generate_page_passage(
            cleaned_text, page.page_number, document_name, sec_id, sec_title, subsystem
        )
        if passage_chunk:
            chunks.append(passage_chunk)

        return chunks

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

        # Look for torque table markers
        if "nm" not in text.lower() or ("description" not in text.lower() and "torque" not in text.lower()):
            return chunks

        lines = text.splitlines()
        in_torque_table = False
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Detect start of torque table
            if re.search(r"^\s*Description\s*$", line, re.I) or re.search(r"Nm\s+lb-ft", line, re.I):
                in_torque_table = True
                i += 1
                # Skip header lines like "Nm lb-ft lb-in"
                while i < len(lines) and any(u in lines[i].lower() for u in ("nm", "lb-ft", "lb-in")):
                    i += 1
                continue

            if in_torque_table:
                # End of table check
                if re.match(r"^(SECTION|Material|Item|Procedure revision date|General Specifications)", line, re.I):
                    in_torque_table = False
                    i += 1
                    continue

                if not line or line in ("—", "-"):
                    i += 1
                    continue

                # Potential component description line
                desc = line
                i += 1
                tokens: list[str] = []

                # Handle case where description has numbers on the same line: "Brake caliper anchor plate bolts 250 184 —"
                # or "Brake flexible hose bracket-to-wheel knuckle bolt 30" followed by "22 \n —"
                multi_match = re.search(r"^(.*?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?|—|-)(?:\s+(\d+(?:\.\d+)?|—|-))?$", desc)
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
                    # Look ahead for remaining column values on subsequent lines
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
                    # Numbers are entirely on subsequent lines
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

                    # Build semantic text content
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

                    chunk_id = f"{doc_name}_p{page_num}_tr_{abs(hash(desc)) % 100000}"
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

        # Common fluid capacities in manuals: e.g. "1,420 ml (3 pt)", "2.84L (6.0 pt)"
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

            chunk_id = f"{doc_name}_p{page_num}_mat_{abs(hash(item)) % 100000}"
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

        # Additional specific regex for brake fluid / oil fill capacity blocks
        cap_block_re = re.compile(
            r"([0-9\.,]+\s*(?:ml|l|liters?|pt|qt|gal)(?:\s*\([^\)]+\))?)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            if any(term in line.lower() for term in ("brake fluid", "lubricant", "fluid", "oil", "grease")) and any(
                unit in line.lower() for unit in ("ml", " pt", " qt", "gal", "l ")
            ):
                cap_match = cap_block_re.search(line)
                if cap_match:
                    cap_val = cap_match.group(1)
                    content = (
                        f"[{sec_title} (Section {sec_id}, Page {page_num}) | Capacity] "
                        f"Line: {line.strip()} | Capacity: {cap_val} | Subsystem: {subsystem}"
                    )
                    chunk_id = f"{doc_name}_p{page_num}_cap_{abs(hash(line)) % 100000}"
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
                            component=line.split(cap_val)[0].strip() or "Vehicle Fluid",
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
        """Extract dimensions, minimum thicknesses, runout, and backlash specifications."""
        chunks: list[SpecificationChunk] = []

        # Common dimension patterns: e.g. "32 mm (1.259 in)", "3.0 mm (0.118 in)", "0.203-0.304 mm (0.008-0.012 in)"
        dim_line_re = re.compile(
            r"^(.*?)\s+([0-9\.\-]+)\s*mm\s*(?:\(([0-9\.\-]+)\s*in\))?$",
            re.IGNORECASE,
        )

        lines = text.splitlines()
        for idx, line in enumerate(lines):
            line_str = line.strip()
            dim_match = dim_line_re.match(line_str)
            if dim_match:
                comp = dim_match.group(1).strip()
                val_mm = dim_match.group(2).strip()
                val_in = dim_match.group(3).strip() if dim_match.group(3) else ""

                if not comp and idx > 0:
                    comp = lines[idx - 1].strip()

                content = (
                    f"[{sec_title} (Section {sec_id}, Page {page_num}) | Dimension Specification] "
                    f"Component: {comp} | Minimum/Clearance: {val_mm} mm ({val_in} in) | Subsystem: {subsystem}"
                )
                evidence = f"{comp} {val_mm} mm ({val_in} in)"

                chunk_id = f"{doc_name}_p{page_num}_dim_{abs(hash(comp)) % 100000}"
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
                        metadata={"spec_type": "dimension", "mm": val_mm, "in": val_in},
                    )
                )

        return chunks

    def _generate_page_passage(
        self,
        cleaned_text: str,
        page_num: int,
        doc_name: str,
        sec_id: str,
        sec_title: str,
        subsystem: str,
    ) -> SpecificationChunk | None:
        """Create a holistic page context chunk for broader semantic queries."""
        if len(cleaned_text.strip()) < 40:
            return None

        # Take first 1000 characters for page-level semantic indexing
        snippet = cleaned_text.strip()[:1200]
        content = (
            f"[{sec_title} (Section {sec_id}, Page {page_num}) | Overview] "
            f"Subsystem: {subsystem}\n{snippet}"
        )

        chunk_id = f"{doc_name}_p{page_num}_passage"
        return SpecificationChunk(
            chunk_id=chunk_id,
            document_name=doc_name,
            page_number=page_num,
            section_id=sec_id,
            section_title=sec_title,
            subsystem=subsystem,
            chunk_type="text_passage",
            content=content,
            raw_evidence=snippet,
            metadata={"full_length": len(cleaned_text)},
        )
