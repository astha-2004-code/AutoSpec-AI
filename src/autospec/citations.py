"""Citation formatting, excerpt highlighting, and provenance presentation."""

from __future__ import annotations

from typing import Any

from autospec.schemas import VehicleSpecification, VerificationStatus


class CitationFormatter:
    """Formats vehicle specification records and evidence quotes into clean Markdown and structured citations."""

    STATUS_LABELS = {
        VerificationStatus.VERIFIED.value: "Verified",
        VerificationStatus.PARTIALLY_SUPPORTED.value: "Partially Supported",
        VerificationStatus.CONFLICTING_EVIDENCE.value: "Conflicting Evidence",
        VerificationStatus.AMBIGUOUS.value: "Ambiguous",
        VerificationStatus.INSUFFICIENT_EVIDENCE.value: "Insufficient Evidence",
        VerificationStatus.NOT_FOUND.value: "Not Found",
        VerificationStatus.UNSUPPORTED.value: "Unsupported",
        VerificationStatus.UNVERIFIED.value: "Unverified",
    }

    @classmethod
    def format_card(cls, spec: VehicleSpecification) -> str:
        """Render an individual specification record as a structured Markdown card."""
        label = cls.STATUS_LABELS.get(spec.verification_status, "Unverified")
        alt_str = ""
        if spec.alternate_value:
            alt_str = f" / **{spec.alternate_value} {spec.alternate_unit or ''}**"

        page_disp = f"Page **{spec.page_number}**"
        if spec.printed_page_label and spec.printed_page_label != str(spec.page_number):
            page_disp += f" (Printed label: `{spec.printed_page_label}`)"

        lines = [
            f"### {spec.component}",
            f"- **Value**: `{spec.value} {spec.unit}`{alt_str}",
            f"- **Category**: `{spec.spec_type}`",
            f"- **Section**: `{spec.section}` (Subsystem: `{spec.subsystem or 'general'}`)",
            f"- **Location**: {page_disp} in *{spec.source_document}*",
            f"- **Verification Status**: `{label}`",
        ]

        if spec.conditions:
            lines.append(f"- **Applicable Conditions**: {spec.conditions}")
        if spec.part_number:
            lines.append(f"- **Part Number**: `{spec.part_number}`")
        if spec.material:
            lines.append(f"- **Material / Spec Code**: `{spec.material}`")

        lines.append(f"- **Method**: `{spec.extraction_method}`")
        lines.append("")
        lines.append("> **Source Evidence Excerpt**:")
        lines.append(f"> \"{spec.source_quote}\"")

        if spec.verification_notes:
            lines.append("")
            lines.append(f"*Verification Diagnostic*: {spec.verification_notes}")

        return "\n".join(lines)

    @classmethod
    def format_table(cls, specs: list[VehicleSpecification]) -> str:
        """Render a collection of specifications into a Markdown summary table."""
        if not specs:
            return "_No vehicle specifications found._"

        headers = ["Component", "Specification", "Alternate", "Section", "Page", "Status"]
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for s in specs:
            alt = f"{s.alternate_value} {s.alternate_unit or ''}" if s.alternate_value else "—"
            label = cls.STATUS_LABELS.get(s.verification_status, s.verification_status)
            row = [
                s.component,
                f"{s.value} {s.unit}".strip(),
                alt,
                s.section.split(":")[-1].strip() if ":" in s.section else s.section,
                f"Page {s.page_number}",
                label,
            ]
            rows.append("| " + " | ".join(row) + " |")

        return "\n".join(rows)

    @classmethod
    def to_citation_payload(cls, spec: VehicleSpecification) -> dict[str, Any]:
        """Build structured citation payload with coordinate bounding boxes for PDF viewer integration."""
        return {
            "document_id": spec.document_id,
            "source_document": spec.source_document,
            "vehicle": " ".join(
                str(x) for x in (spec.vehicle_year, spec.vehicle_make, spec.vehicle_model) if x
            ),
            "section": spec.section,
            "subsystem": spec.subsystem,
            "page_number": spec.page_number,
            "printed_page_label": spec.printed_page_label or str(spec.page_number),
            "source_quote": spec.source_quote,
            "evidence_bboxes": [b.model_dump() for b in spec.evidence_bboxes],
            "verification_status": spec.verification_status,
        }
