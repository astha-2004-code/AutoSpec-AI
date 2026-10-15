"""Citation formatting, excerpt highlighting, and provenance presentation."""

from __future__ import annotations

from autospec.schemas import VehicleSpecification, VerificationStatus


class CitationFormatter:
    """Formats vehicle specification records and evidence quotes into rich Markdown displays."""

    STATUS_BADGES = {
        VerificationStatus.VERIFIED.value: "🟢 Verified Grounding",
        VerificationStatus.AMBIGUOUS.value: "🟡 Ambiguous Evidence",
        VerificationStatus.UNSUPPORTED.value: "🔴 Unsupported / Conflict",
        VerificationStatus.UNVERIFIED.value: "⚪ Unverified",
    }

    @classmethod
    def format_card(cls, spec: VehicleSpecification) -> str:
        """Render an individual specification record as a structured Markdown card."""
        badge = cls.STATUS_BADGES.get(spec.verification_status, "⚪ Unverified")
        alt_str = ""
        if spec.alternate_value:
            alt_str = f" / **{spec.alternate_value} {spec.alternate_unit or ''}**"

        lines = [
            f"### 🔧 {spec.component}",
            f"- **Value**: `{spec.value} {spec.unit}`{alt_str}",
            f"- **Category**: `{spec.spec_type}`",
            f"- **Section**: `{spec.section}` (Subsystem: `{spec.subsystem or 'general'}`)",
            f"- **Location**: Page **{spec.page_number}** in *{spec.source_document}*",
            f"- **Status**: {badge}",
        ]

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

        headers = [
            "Component",
            "Specification",
            "Alternate",
            "Section",
            "Page",
            "Status",
        ]
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for s in specs:
            alt = f"{s.alternate_value} {s.alternate_unit or ''}" if s.alternate_value else "—"
            status_emoji = "✅" if s.verification_status == VerificationStatus.VERIFIED.value else (
                "⚠️" if s.verification_status == VerificationStatus.AMBIGUOUS.value else "❌"
            )
            row = [
                s.component,
                f"{s.value} {s.unit}",
                alt,
                s.section.split(":")[-1].strip() if ":" in s.section else s.section,
                f"Page {s.page_number}",
                f"{status_emoji} {s.verification_status}",
            ]
            rows.append("| " + " | ".join(row) + " |")

        return "\n".join(rows)
