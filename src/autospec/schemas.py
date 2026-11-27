"""Pydantic data schemas for automotive specification intelligence, multi-doc storage, and evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpecType(str, Enum):
    """Category of vehicle technical specification."""

    TORQUE = "torque"
    FLUID_CAPACITY = "fluid_capacity"
    PART_NUMBER = "part_number"
    LUBRICANT_MATERIAL = "lubricant_material"
    DIMENSION_CLEARANCE = "dimension_clearance"
    ELECTRICAL = "electrical"
    PRESSURE = "pressure"
    MAINTENANCE = "maintenance"
    GENERAL = "general"


class VerificationStatus(str, Enum):
    """Grounding verification status of an extracted record against source evidence."""

    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


class ExtractionMethod(str, Enum):
    """Pipeline method responsible for the extracted specification."""

    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    TABLE_PARSER = "table_parser"
    LLM_EXTRACTION = "llm_extraction"
    MANUAL_BENCHMARK = "manual_benchmark"


class EvidenceBBox(BaseModel):
    """PDF coordinate bounding box in PDF points (top-left origin matching PyMuPDF)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float = 612.0
    page_height: float = 792.0


class VehicleMetadata(BaseModel):
    """Vehicle identity metadata associated with a workshop manual."""

    manufacturer: str = Field(default="Unknown", description="OEM manufacturer (e.g. 'Ford', 'Apex Motors')")
    model: str = Field(default="Unknown", description="Vehicle model name (e.g. 'F-150', 'M400 Pickup')")
    year: int | None = Field(default=None, ge=1950, le=2035, description="Model year")
    engine: str | None = Field(default=None, description="Engine or powertrain variant")
    trim: str | None = Field(default=None, description="Trim or drivetrain configuration (e.g. '4WD', 'Hybrid')")


class VehicleSpecification(BaseModel):
    """Standardized, evidence-grounded vehicle specification record."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    document_id: str | None = Field(default=None, description="Unique document collection identifier")
    component: str = Field(
        ...,
        description="Name of the vehicle part, fastener, or assembly",
    )
    spec_type: str = Field(
        default=SpecType.TORQUE.value,
        description="Specification category: torque, fluid_capacity, dimension_clearance, electrical, etc.",
    )
    value: str = Field(
        ...,
        description="Primary numerical, range, or descriptive specification value",
    )
    unit: str = Field(
        default="",
        description="Primary engineering unit of measurement (e.g. 'Nm', 'ml', 'mm')",
    )
    alternate_value: str | None = Field(
        default=None,
        description="Secondary technical value in imperial or alternate metric units",
    )
    alternate_unit: str | None = Field(
        default=None,
        description="Secondary engineering unit of measurement",
    )
    normalized_value: float | None = Field(
        default=None,
        description="Canonical SI-normalized scalar value for cross-vehicle comparison",
    )
    normalized_unit: str | None = Field(
        default=None,
        description="Canonical unit for normalized_value",
    )
    range_min: str | None = Field(
        default=None,
        description="Minimum bound if specification represents a tolerance or clearance interval",
    )
    range_max: str | None = Field(
        default=None,
        description="Maximum bound if specification represents a tolerance or clearance interval",
    )
    tolerance: str | None = Field(
        default=None,
        description="Explicit tolerance (+/-) if stated in manual",
    )
    conditions: str | None = Field(
        default=None,
        description="Applicable vehicle variants, engine codes, or procedural conditions",
    )
    part_number: str | None = Field(
        default=None,
        description="Manufacturer part number or fluid chemical code",
    )
    material: str | None = Field(
        default=None,
        description="Material, chemical grade, or engineering specification standard",
    )
    section: str = Field(
        ...,
        description="Full workshop manual section header",
    )
    subsystem: str | None = Field(
        default=None,
        description="Classified vehicle subsystem",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed physical PDF page number",
    )
    printed_page_label: str | None = Field(
        default=None,
        description="Printed manual page label when different from physical PDF page index",
    )
    source_document: str = Field(
        ...,
        description="Filename of the source service manual document",
    )
    vehicle_make: str | None = Field(default=None, description="Vehicle manufacturer")
    vehicle_model: str | None = Field(default=None, description="Vehicle model")
    vehicle_year: int | None = Field(default=None, description="Vehicle model year")
    source_quote: str = Field(
        ...,
        description="Verbatim text excerpt from the manual providing evidence for the extracted value",
    )
    evidence_bboxes: list[EvidenceBBox] = Field(
        default_factory=list,
        description="Actual PDF text/row bounding boxes for visual PDF highlighting",
    )
    retrieval_score: float = Field(
        default=0.0,
        description="Ranking relevance score from hybrid retrieval",
    )
    verification_status: str = Field(
        default=VerificationStatus.UNVERIFIED.value,
        description="Strict verification outcome status",
    )
    verification_notes: str | None = Field(
        default=None,
        description="Reasoning or diagnostics reported by the evidence validator",
    )
    extraction_method: str = Field(
        default=ExtractionMethod.DETERMINISTIC_FALLBACK.value,
        description="Mechanism used to extract the value",
    )

    def to_export_dict(self) -> dict[str, Any]:
        """Convert to flattened dictionary optimized for CSV/JSON tabular exports."""
        return {
            "document_id": self.document_id or "",
            "vehicle": " ".join(
                str(x) for x in (self.vehicle_year, self.vehicle_make, self.vehicle_model) if x
            ),
            "component": self.component,
            "spec_type": self.spec_type,
            "value": self.value,
            "unit": self.unit,
            "alternate_value": self.alternate_value or "",
            "alternate_unit": self.alternate_unit or "",
            "range_min": self.range_min or "",
            "range_max": self.range_max or "",
            "conditions": self.conditions or "",
            "part_number": self.part_number or "",
            "material": self.material or "",
            "section": self.section,
            "subsystem": self.subsystem or "",
            "page_number": self.page_number,
            "printed_page_label": self.printed_page_label or "",
            "source_quote": self.source_quote,
            "verification_status": self.verification_status,
            "verification_notes": self.verification_notes or "",
            "extraction_method": self.extraction_method,
            "source_document": self.source_document,
        }


class QueryRequest(BaseModel):
    """Request model for specification queries and conversational follow-ups."""

    query: str = Field(..., min_length=2, max_length=600, description="Natural language engineering question")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of specifications to return")
    document_ids: list[str] | None = Field(
        default=None, description="Optional list of document IDs to scope retrieval"
    )
    subsystem_filter: str | None = Field(
        default=None, description="Optional subsystem filter"
    )
    manufacturer_filter: str | None = Field(default=None, description="Filter by OEM manufacturer")
    model_filter: str | None = Field(default=None, description="Filter by vehicle model")
    year_filter: int | None = Field(default=None, description="Filter by model year")
    retrieval_mode: str = Field(
        default="hybrid_rerank",
        description="Retrieval strategy: 'hybrid_rerank', 'hybrid', 'dense', or 'bm25'",
    )
    conversation_id: str | None = Field(
        default=None, description="Optional conversation thread ID for follow-up context"
    )
    min_score: float | None = Field(
        default=None, ge=0.0, description="Optional retrieval threshold override"
    )


class QueryResponse(BaseModel):
    """Response model returning grounded specification records and conversational state."""

    query: str
    normalized_query: str | None = None
    results: list[VehicleSpecification]
    total_results: int
    extraction_method: str
    retrieval_mode: str = "hybrid_rerank"
    processing_time_ms: float
    status: str = "answered"  # answered, clarification_needed, insufficient_evidence, not_found
    clarification_needed: bool = False
    clarification_prompt: str | None = None
    missing_info_explanation: str | None = None
    conversation_id: str | None = None
    active_context: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class IngestRequest(BaseModel):
    """Request metadata for multipart upload or reindex operations (local paths disabled in REST)."""

    force_rebuild: bool = Field(default=False, description="Force re-indexing even if fingerprint matches")
    manufacturer: str | None = Field(default=None, description="Optional OEM manufacturer override")
    model: str | None = Field(default=None, description="Optional vehicle model override")
    year: int | None = Field(default=None, ge=1950, le=2035, description="Optional model year override")
    enable_ocr: bool = Field(default=False, description="Enable OCR fallback on text-poor pages")


class IngestResponse(BaseModel):
    """Response model for document ingestion outcome."""

    status: str
    document_id: str = ""
    filename: str
    fingerprint: str
    total_pages: int
    chunk_count: int
    is_cached: bool
    processing_time_ms: float
    message: str
    job_id: str | None = None


class DocumentInfo(BaseModel):
    """Metadata regarding an indexed document collection."""

    document_id: str = ""
    filename: str
    display_name: str = ""
    fingerprint: str
    total_pages: int
    indexed_chunks: int
    spec_count: int = 0
    freshness: str
    created_at: str
    is_demo: bool = False
    read_only: bool = False
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    year: int | None = None
    subsystems: list[str] = Field(default_factory=list)
    ocr_pages: int = 0
    file_size_bytes: int = 0


class DocumentRenameRequest(BaseModel):
    """Request payload for renaming a document or updating its vehicle metadata."""

    display_name: str | None = Field(default=None, min_length=1, max_length=160)
    manufacturer: str | None = Field(default=None, max_length=80)
    model: str | None = Field(default=None, max_length=80)
    year: int | None = Field(default=None, ge=1950, le=2035)


class IngestionJobStatus(BaseModel):
    """Durable background ingestion job progress and stage status."""

    job_id: str
    document_id: str | None = None
    filename: str
    status: str  # queued, running, completed, failed, cancelled
    stage: str = "uploading"  # uploading, parsing, chunking, embedding, indexing, validating, completed, failed, cancelled
    current_page: int = 0
    total_pages: int = 0
    progress_pct: float = 0.0
    is_duplicate: bool = False
    message: str = ""
    error: str | None = None
    created_at: str
    updated_at: str


class ComparisonRequest(BaseModel):
    """Request to compare technical specifications across selected vehicles/documents."""

    document_ids: list[str] = Field(..., min_length=2, max_length=6)
    subsystem: str | None = Field(default=None)
    spec_type: str | None = Field(default=None)
    component_query: str | None = Field(default=None)
    preferred_torque_unit: str = Field(default="Nm")
    preferred_volume_unit: str = Field(default="ml")
    preferred_length_unit: str = Field(default="mm")


class ComparisonCell(BaseModel):
    """Single vehicle value cell within a comparison row."""

    document_id: str
    vehicle_label: str
    status: str  # present, missing, incomparable
    original_value: str | None = None
    original_unit: str | None = None
    normalized_value: float | None = None
    normalized_unit: str | None = None
    alternate_value: str | None = None
    alternate_unit: str | None = None
    page_number: int | None = None
    printed_page_label: str | None = None
    section: str | None = None
    source_quote: str | None = None
    verification_status: str | None = None
    evidence_bboxes: list[EvidenceBBox] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    """Aligned specification row across compared vehicles."""

    canonical_component: str
    spec_type: str
    subsystem: str
    cells: list[ComparisonCell]
    has_variance: bool = False
    comparable: bool = True
    notes: str | None = None


class ComparisonResponse(BaseModel):
    """Structured vehicle specification comparison report."""

    vehicles: list[DocumentInfo]
    rows: list[ComparisonRow]
    total_rows: int
    subsystems_covered: list[str]


class SearchHistoryEntry(BaseModel):
    """Persisted query history item scoped to a workspace session."""

    id: int
    query: str
    document_ids: list[str] = Field(default_factory=list)
    subsystem_filter: str | None = None
    result_count: int = 0
    top_status: str | None = None
    latency_ms: float = 0.0
    created_at: str


class HealthResponse(BaseModel):
    """API health status and environment configuration."""

    status: str
    version: str
    environment: str = "development"
    storage_ready: bool
    database_ready: bool = True
    database_backend: str = "sqlite"
    pgvector_enabled: bool = False
    demo_ready: bool = False
    active_llm_provider: str
    embedding_model: str
    reranker_enabled: bool = True
    ocr_available: bool = False
    document_count: int = 0


class OverviewStats(BaseModel):
    """Workspace overview metrics derived strictly from persisted database records."""

    document_count: int
    demo_document_count: int
    user_document_count: int
    total_pages: int
    total_chunks: int
    total_specifications: int
    verified_specifications: int
    query_count: int
    subsystems: list[str] = Field(default_factory=list)
