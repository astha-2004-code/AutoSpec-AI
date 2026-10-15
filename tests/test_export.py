"""Unit tests for CSV and JSON tabular export generation."""

from __future__ import annotations

import json

from autospec.export import ExportManager
from autospec.schemas import VehicleSpecification, VerificationStatus


def test_export_manager_csv():
    """Test ExportManager CSV formatting and header integrity."""
    specs = [
        VehicleSpecification(
            component="Brake caliper anchor plate bolts",
            value="250",
            unit="Nm",
            alternate_value="184",
            alternate_unit="lb-ft",
            section="SECTION 206-03: Front Disc Brake",
            subsystem="front_brakes",
            page_number=636,
            source_document="manual.pdf",
            source_quote="Brake caliper anchor plate bolts 250 184 —",
            verification_status=VerificationStatus.VERIFIED.value,
        )
    ]
    csv_str = ExportManager.to_csv(specs)
    assert "component,spec_type,value,unit" in csv_str
    assert "Brake caliper anchor plate bolts" in csv_str
    assert "250" in csv_str
    assert "184" in csv_str


def test_export_manager_json():
    """Test ExportManager JSON serialization."""
    specs = [
        VehicleSpecification(
            component="Brake caliper guide pin bolts",
            value="37",
            unit="Nm",
            section="SECTION 206-03: Front Disc Brake",
            subsystem="front_brakes",
            page_number=636,
            source_document="manual.pdf",
            source_quote="Brake caliper guide pin bolts 37 27 —",
        )
    ]
    json_str = ExportManager.to_json(specs)
    data = json.loads(json_str)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["component"] == "Brake caliper guide pin bolts"
    assert data[0]["value"] == "37"
