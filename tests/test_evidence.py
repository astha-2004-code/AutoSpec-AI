"""Unit tests for evidence validator, quote verification, and cross-subsystem diagnostics."""

from __future__ import annotations

from autospec.evidence_validator import EvidenceValidator
from autospec.schemas import VehicleSpecification, VerificationStatus


def test_evidence_validator_valid():
    """Test successful evidence verification when value and quote match page text."""
    validator = EvidenceValidator()
    spec = VehicleSpecification(
        component="Brake caliper anchor plate bolts",
        value="250",
        unit="Nm",
        section="SECTION 206-03: Front Disc Brake",
        subsystem="front_brakes",
        page_number=636,
        source_document="manual.pdf",
        source_quote="Brake caliper anchor plate bolts 250 184 —",
    )
    page_map = {636: "Brake caliper anchor plate bolts 250 184 —\nSECTION 206-03: Front Disc Brake"}

    validated = validator.validate(spec, page_text_map=page_map)
    assert validated.verification_status == VerificationStatus.VERIFIED.value
    assert "Confirmed" in validated.verification_notes or "Verified" in validated.verification_notes


def test_evidence_validator_unsupported_value():
    """Test flagging of unsupported value not appearing in quote."""
    validator = EvidenceValidator()
    spec = VehicleSpecification(
        component="Brake caliper anchor plate bolts",
        value="999",  # Hallucinated value
        unit="Nm",
        section="SECTION 206-03: Front Disc Brake",
        subsystem="front_brakes",
        page_number=636,
        source_document="manual.pdf",
        source_quote="Brake caliper anchor plate bolts 250 184 —",
    )

    validated = validator.validate(spec)
    assert validated.verification_status == VerificationStatus.UNSUPPORTED.value
    assert "does not appear in quoted evidence" in validated.verification_notes


def test_evidence_validator_cross_subsystem_conflict():
    """Test flagging of cross-subsystem mismatch between query and section."""
    validator = EvidenceValidator()
    spec = VehicleSpecification(
        component="Brake caliper guide pin bolts",
        value="37",
        unit="Nm",
        section="SECTION 206-03: Front Disc Brake",
        subsystem="front_brakes",
        page_number=636,
        source_document="manual.pdf",
        source_quote="Brake caliper guide pin bolts 37 27 —",
    )

    # Query specifically asks for REAR brakes
    validated = validator.validate(spec, query="Torque for rear brake caliper guide pin bolts")
    assert validated.verification_status == VerificationStatus.AMBIGUOUS.value
    assert "front disc brake section" in validated.verification_notes
