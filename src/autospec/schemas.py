"""Pydantic data schemas for automotive specification intelligence."""

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
    GENERAL = "general"


class VerificationStatus(str, Enum):
    """Grounding verification status of an extracted record against evidence."""

    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


class ExtractionMethod(str, Enum):
    """Pipeline method responsible for the extracted specification."""

    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    LLM_EXTRACTION = "llm_extraction"
    MANUAL_BENCHMARK = "manual_benchmark"


class VehicleSpecification(BaseModel):
    """Standardized, evidence-grounded vehicle specification record."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    component: str = Field(
        ...,
        description="Name of the vehicle part, fastener, or assembly (e.g. 'Brake caliper anchor plate bolts')",
    )
    spec_type: str = Field(
        default=SpecType.TORQUE.value,
        description="Specification category: torque, fluid_capacity, part_number, dimension_clearance, etc.",
    )
    value: str = Field(
        ...,
        description="Primary numerical or descriptive technical specification value (e.g. '250', '1,420')",
    )
    unit: str = Field(
        default="",
        description="Primary engineering unit of measurement (e.g. 'Nm', 'ml', 'mm')",
    )
    alternate_value: str | None = Field(
        default=None,
        description="Secondary technical value in imperial or alternate metric units (e.g. '184', '3.0')",
    )
    alternate_unit: str | None = Field(
        default=None,
        description="Secondary engineering unit of measurement (e.g. 'lb-ft', 'pt', 'in')",
    )
    range_min: str | None = Field(
        default=None,
        description="Minimum bound if specification represents a tolerance or clearance interval",
    )
    range_max: str | None = Field(
        default=None,
        description="Maximum bound if specification represents a tolerance or clearance interval",
    )
    part_number: str | None = Field(
        default=None,
        description="Manufacturer part number or fluid chemical code (e.g. 'PM-1-C')",
    )
    material: str | None = Field(
        default=None,
        description="Material, chemical grade, or engineering specification standard (e.g. 'WSS-M6C62-A')",
    )
    section: str = Field(
        ...,
        description="Full workshop manual section header (e.g. 'SECTION 206-03: Front Disc Brake')",
    )
    subsystem: str | None = Field(
        default=None,
        description="Classified vehicle subsystem (e.g. 'front_brakes', 'rear_brakes')",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number within the original PDF service manual",
    )
    source_document: str = Field(
        ...,
        description="Filename of the source service manual document",
    )
    source_quote: str = Field(
        ...,
        description="Verbatim text excerpt from the manual providing evidence for the extracted value",
    )
    retrieval_score: float = Field(
        default=0.0,
        description="Ranking relevance score from hybrid retrieval",
    )
    verification_status: str = Field(
        default=VerificationStatus.UNVERIFIED.value,
        description="Verification outcome: verified, ambiguous, or unsupported",
    )
    verification_notes: str | None = Field(
        default=None,
        description="Reasoning or diagnostics reported by the evidence validator",
    )
    extraction_method: str = Field(
        default=ExtractionMethod.DETERMINISTIC_FALLBACK.value,
        description="Mechanism used to extract the value ('llm_extraction' or 'deterministic_fallback')",
    )

    def to_export_dict(self) -> dict[str, Any]:
        """Convert to flattened dictionary optimized for CSV/JSON tabular exports."""
        return {
            "component": self.component,
            "spec_type": self.spec_type,
            "value": self.value,
            "unit": self.unit,
            "alternate_value": self.alternate_value or "",
            "alternate_unit": self.alternate_unit or "",
            "range_min": self.range_min or "",
            "range_max": self.range_max or "",
            "part_number": self.part_number or "",
            "material": self.material or "",
            "section": self.section,
            "subsystem": self.subsystem or "",
            "page_number": self.page_number,
            "source_quote": self.source_quote,
            "verification_status": self.verification_status,
            "extraction_method": self.extraction_method,
            "source_document": self.source_document,
        }


class QueryRequest(BaseModel):
    """Request model for specification queries."""

    query: str = Field(..., min_length=2, max_length=500, description="Natural language engineering question")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of specifications to return")
    subsystem_filter: str | None = Field(
        default=None, description="Optional subsystem filter (e.g. 'front_brakes', 'rear_brakes')"
    )


class QueryResponse(BaseModel):
    """Response model returning grounded specification records."""

    query: str
    results: list[VehicleSpecification]
    total_results: int
    extraction_method: str
    processing_time_ms: float
    message: str | None = None


class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    pdf_path: str | None = Field(default=None, description="Local filesystem path to PDF manual")
    force_rebuild: bool = Field(default=False, description="Force re-indexing even if fingerprint matches")


class IngestResponse(BaseModel):
    """Response model for document ingestion outcome."""

    status: str
    filename: str
    fingerprint: str
    total_pages: int
    chunk_count: int
    is_cached: bool
    processing_time_ms: float
    message: str


class DocumentInfo(BaseModel):
    """Metadata regarding currently indexed document."""

    filename: str
    fingerprint: str
    total_pages: int
    indexed_chunks: int
    freshness: str
    created_at: str


class HealthResponse(BaseModel):
    """API health status and environment configuration."""

    status: str
    version: str
    storage_ready: bool
    active_llm_provider: str
    embedding_model: str
