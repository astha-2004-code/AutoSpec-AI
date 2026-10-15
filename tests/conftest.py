"""Shared pytest fixtures for AutoSpec AI test suite."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pymupdf
import pytest

from autospec.chunker import SpecificationChunk
from autospec.embeddings import DeterministicMockEmbedder


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_embedder() -> DeterministicMockEmbedder:
    """Deterministic mock embedder with 384 dimensions."""
    return DeterministicMockEmbedder(dimension=384)


@pytest.fixture
def sample_front_brake_text() -> str:
    """Synthetic excerpt replicating Page 636 of Ford F-150 workshop manual."""
    return """General Specifications 
Torque Specifications 
SECTION 206-03: Front Disc Brake 
2014 F-150 Workshop Manual 
SPECIFICATIONS 
Procedure revision date: 10/25/2013 
Material 
Item 
Specification 
Fill 
Capacity 
High Performance DOT 3 
Motor Vehicle Brake Fluid 
PM-1-C (US); CPM-1-C 
(Canada) 
WSS-M6C62-A or 
WSS-M6C65-A1 
1,420 ml 
(3 pt) 
Item 
Specification 
Brake Disc 
Brake disc minimum thickness 
32 mm (1.259 in) 
Description 
Nm lb-ft lb-in 
Brake caliper anchor plate bolts 
250 184 
— 
Brake caliper flow bolt 
35 
26 
— 
Brake caliper guide pin bolts 
37 
27 
— 
Brake disc shield bolts 
18 
— 
159 
Page 1 sur 2
2014-03-01
file:///C:/TSO/tsocache/sample.HTM
"""


@pytest.fixture
def sample_rear_brake_text() -> str:
    """Synthetic excerpt replicating Page 652 of Ford F-150 workshop manual."""
    return """General Specifications 
Torque Specifications 
SECTION 206-04: Rear Disc Brake 
2014 F-150 Workshop Manual 
SPECIFICATIONS 
Procedure revision date: 10/25/2013 
Item 
Specification 
Brake Disc 
Brake disc minimum thickness 
23.0 mm (0.906 in) 
Description 
Nm lb-ft lb-in 
Axle vent valve 
18 
— 
159 
Brake caliper guide pin bolts 
33 
24 
— 
Brake caliper support bracket bolts 
150 111 
— 
Page 1 sur 2
2014-03-01
file:///C:/TSO/tsocache/sample.HTM
"""


@pytest.fixture
def sample_synthetic_pdf(temp_dir: Path, sample_front_brake_text: str, sample_rear_brake_text: str) -> Path:
    """Dynamically generates a valid, minimal multi-page PDF for testing."""
    pdf_path = temp_dir / "test_manual.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 50), sample_front_brake_text)

    p2 = doc.new_page()
    p2.insert_text((50, 50), sample_rear_brake_text)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_chunks() -> list[SpecificationChunk]:
    """Sample specification chunks covering front and rear brake fasteners."""
    return [
        SpecificationChunk(
            chunk_id="chunk_front_anchor",
            document_name="test_manual.pdf",
            page_number=636,
            section_id="206-03",
            section_title="Front Disc Brake",
            subsystem="front_brakes",
            chunk_type="spec_row",
            content="[Front Disc Brake (Section 206-03, Page 636) | Torque Specification] Component: Brake caliper anchor plate bolts | Torque: 250 Nm / 184 lb-ft | Subsystem: front_brakes",
            raw_evidence="Brake caliper anchor plate bolts 250 184 —",
            table_name="Torque Specifications",
            component="Brake caliper anchor plate bolts",
            primary_value="250",
            primary_unit="Nm",
            alternate_value="184",
            alternate_unit="lb-ft",
            metadata={"spec_type": "torque"},
        ),
        SpecificationChunk(
            chunk_id="chunk_front_guide",
            document_name="test_manual.pdf",
            page_number=636,
            section_id="206-03",
            section_title="Front Disc Brake",
            subsystem="front_brakes",
            chunk_type="spec_row",
            content="[Front Disc Brake (Section 206-03, Page 636) | Torque Specification] Component: Brake caliper guide pin bolts | Torque: 37 Nm / 27 lb-ft | Subsystem: front_brakes",
            raw_evidence="Brake caliper guide pin bolts 37 27 —",
            table_name="Torque Specifications",
            component="Brake caliper guide pin bolts",
            primary_value="37",
            primary_unit="Nm",
            alternate_value="27",
            alternate_unit="lb-ft",
            metadata={"spec_type": "torque"},
        ),
        SpecificationChunk(
            chunk_id="chunk_rear_guide",
            document_name="test_manual.pdf",
            page_number=652,
            section_id="206-04",
            section_title="Rear Disc Brake",
            subsystem="rear_brakes",
            chunk_type="spec_row",
            content="[Rear Disc Brake (Section 206-04, Page 652) | Torque Specification] Component: Brake caliper guide pin bolts | Torque: 33 Nm / 24 lb-ft | Subsystem: rear_brakes",
            raw_evidence="Brake caliper guide pin bolts 33 24 —",
            table_name="Torque Specifications",
            component="Brake caliper guide pin bolts",
            primary_value="33",
            primary_unit="Nm",
            alternate_value="24",
            alternate_unit="lb-ft",
            metadata={"spec_type": "torque"},
        ),
    ]
