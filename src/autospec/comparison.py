"""Multi-vehicle specification comparison engine with unit normalization and source citations."""

from __future__ import annotations

import math
import re

from autospec.schemas import (
    ComparisonCell,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonRow,
    DocumentInfo,
    VehicleSpecification,
)
from autospec.units import (
    are_units_compatible,
    convert_value,
    normalize_unit_symbol,
    parse_numeric_or_range,
)


class VehicleComparisonEngine:
    """Aligns and compares specifications across multiple vehicle manuals with strict unit safety."""

    @staticmethod
    def _canonical_component_key(spec: VehicleSpecification) -> tuple[str, str, str]:
        """Generate normalized grouping key (subsystem, spec_type, normalized_component)."""
        comp = spec.component.lower().strip()
        comp = re.sub(r"\([^)]*\)", "", comp)
        comp = re.sub(r"\b(front|rear|left|right)\b", "", comp)
        comp = re.sub(r"\s+", " ", comp).strip(" -:")
        subsystem = (spec.subsystem or "general").lower()
        stype = (spec.spec_type or "general").lower()
        return (subsystem, stype, comp)

    @staticmethod
    def _vehicle_label(doc: DocumentInfo) -> str:
        parts = [str(doc.year)] if doc.year else []
        if doc.manufacturer and doc.manufacturer != "Unknown":
            parts.append(doc.manufacturer)
        if doc.model and doc.model != "Unknown":
            parts.append(doc.model)
        return " ".join(parts) if parts else (doc.display_name or doc.filename)

    @staticmethod
    def _preferred_unit_for_spec(spec_type: str, orig_unit: str, req: ComparisonRequest) -> str:
        norm = normalize_unit_symbol(orig_unit)
        if spec_type == "torque" or norm in ("Nm", "lb-ft", "lb-in", "kgf-m"):
            return normalize_unit_symbol(req.preferred_torque_unit)
        if spec_type == "fluid_capacity" or norm in ("ml", "L", "pt", "qt", "gal"):
            return normalize_unit_symbol(req.preferred_volume_unit)
        if spec_type in ("dimension_clearance", "dimension") or norm in ("mm", "cm", "in", "mil"):
            return normalize_unit_symbol(req.preferred_length_unit)
        return norm

    def compare(
        self,
        documents: list[DocumentInfo],
        specs_by_doc: dict[str, list[VehicleSpecification]],
        request: ComparisonRequest,
    ) -> ComparisonResponse:
        """Build side-by-side comparison matrix across selected vehicles."""
        # Group specifications by canonical (subsystem, spec_type, component)
        grouped: dict[tuple[str, str, str], dict[str, VehicleSpecification]] = {}
        display_names: dict[tuple[str, str, str], str] = {}

        for doc in documents:
            doc_specs = specs_by_doc.get(doc.document_id, [])
            for s in doc_specs:
                if request.subsystem and (s.subsystem or "").lower() != request.subsystem.lower():
                    continue
                if request.spec_type and s.spec_type.lower() != request.spec_type.lower():
                    continue
                if request.component_query:
                    q_low = request.component_query.lower()
                    if q_low not in s.component.lower() and q_low not in (s.subsystem or "").lower():
                        continue

                key = self._canonical_component_key(s)
                if key not in grouped:
                    grouped[key] = {}
                    display_names[key] = s.component
                # Keep highest-verified or first record per doc
                if doc.document_id not in grouped[key]:
                    grouped[key][doc.document_id] = s

        rows: list[ComparisonRow] = []
        subsystems_seen: set[str] = set()

        for key, doc_map in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[1].get(documents[0].document_id).component if documents[0].document_id in kv[1] else kv[0][2])):
            subsystem, stype, _ = key
            subsystems_seen.add(subsystem)
            cells: list[ComparisonCell] = []
            norm_vals: list[float] = []
            unit_families: set[str] = set()

            for doc in documents:
                v_label = self._vehicle_label(doc)
                spec = doc_map.get(doc.document_id)
                if not spec:
                    cells.append(
                        ComparisonCell(
                            document_id=doc.document_id,
                            vehicle_label=v_label,
                            status="missing",
                        )
                    )
                    continue

                parsed = parse_numeric_or_range(spec.value)
                orig_unit = normalize_unit_symbol(spec.unit) or parsed.unit
                target_unit = self._preferred_unit_for_spec(stype, orig_unit, request)

                scalar = parsed.point
                if scalar is None and parsed.range_min is not None and parsed.range_max is not None:
                    scalar = (parsed.range_min + parsed.range_max) / 2.0

                norm_val: float | None = None
                norm_unit: str | None = None
                if scalar is not None and orig_unit:
                    if target_unit and are_units_compatible(orig_unit, target_unit):
                        conv = convert_value(scalar, orig_unit, target_unit)
                        if conv is not None:
                            norm_val = round(conv, 3)
                            norm_unit = target_unit
                    if norm_val is None:
                        norm_val = round(scalar, 3)
                        norm_unit = orig_unit

                if norm_unit:
                    unit_families.add(norm_unit)
                if norm_val is not None:
                    norm_vals.append(norm_val)

                cells.append(
                    ComparisonCell(
                        document_id=doc.document_id,
                        vehicle_label=v_label,
                        status="present",
                        original_value=spec.value,
                        original_unit=spec.unit,
                        normalized_value=norm_val,
                        normalized_unit=norm_unit,
                        alternate_value=spec.alternate_value,
                        alternate_unit=spec.alternate_unit,
                        page_number=spec.page_number,
                        printed_page_label=spec.printed_page_label or str(spec.page_number),
                        section=spec.section,
                        source_quote=spec.source_quote,
                        verification_status=spec.verification_status,
                        evidence_bboxes=spec.evidence_bboxes,
                    )
                )

            comparable = len(unit_families) <= 1
            if len(unit_families) > 1:
                for c in cells:
                    if c.status == "present":
                        c.status = "incomparable"

            has_variance = False
            if comparable and len(norm_vals) >= 2:
                has_variance = any(not math.isclose(norm_vals[0], v, rel_tol=0.01, abs_tol=1e-3) for v in norm_vals[1:])

            present_count = sum(1 for c in cells if c.status == "present")
            notes = None
            if not comparable:
                notes = f"Incompatible units across manuals: {sorted(unit_families)}"
            elif present_count < len(documents):
                notes = f"Specification present in {present_count} of {len(documents)} selected manuals."

            rows.append(
                ComparisonRow(
                    canonical_component=display_names[key],
                    spec_type=stype,
                    subsystem=subsystem,
                    cells=cells,
                    has_variance=has_variance,
                    comparable=comparable,
                    notes=notes,
                )
            )

        return ComparisonResponse(
            vehicles=documents,
            rows=rows,
            total_rows=len(rows),
            subsystems_covered=sorted(subsystems_seen),
        )
