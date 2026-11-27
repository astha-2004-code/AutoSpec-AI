"""Tabular and JSON export utilities (zero-pandas runtime for <512MB deployment)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from autospec.schemas import ComparisonResponse, VehicleSpecification


class ExportManager:
    """Manages CSV, JSON, and comparison exports using Python stdlib csv/json."""

    EXPORT_COLUMNS = [
        "vehicle",
        "component",
        "spec_type",
        "value",
        "unit",
        "alternate_value",
        "alternate_unit",
        "range_min",
        "range_max",
        "conditions",
        "section",
        "subsystem",
        "page_number",
        "printed_page_label",
        "source_document",
        "source_quote",
        "part_number",
        "material",
        "verification_status",
        "verification_notes",
        "extraction_method",
        "document_id",
    ]

    @classmethod
    def _build_export_rows(cls, specs: list[VehicleSpecification]) -> tuple[list[str], list[dict[str, Any]]]:
        if not specs:
            return list(cls.EXPORT_COLUMNS), []
        records = [s.to_export_dict() for s in specs]
        extra_cols: list[str] = []
        seen = set(cls.EXPORT_COLUMNS)
        for rec in records:
            for key in rec:
                if key not in seen:
                    seen.add(key)
                    extra_cols.append(key)
        fieldnames = list(cls.EXPORT_COLUMNS) + extra_cols
        normalized = [{col: rec.get(col, "") for col in fieldnames} for rec in records]
        return fieldnames, normalized

    @classmethod
    def to_dataframe(cls, specs: list[VehicleSpecification]):
        """Optional Pandas DataFrame helper (imported lazily to avoid ~47MB startup RAM overhead)."""
        import pandas as pd  # type: ignore[import-not-found,import-untyped]

        fieldnames, rows = cls._build_export_rows(specs)
        if not rows:
            return pd.DataFrame(columns=fieldnames)
        return pd.DataFrame(rows, columns=fieldnames)

    @classmethod
    def to_csv(cls, specs: list[VehicleSpecification]) -> str:
        """Generate formatted CSV string with UTF-8 encoding."""
        fieldnames, rows = cls._build_export_rows(specs)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return output.getvalue()

    @classmethod
    def to_json(cls, specs: list[VehicleSpecification], indent: int = 2) -> str:
        """Generate structured JSON array string."""
        records = [s.to_export_dict() for s in specs]
        return json.dumps(records, indent=indent, ensure_ascii=False)

    @classmethod
    def comparison_to_csv(cls, report: ComparisonResponse) -> str:
        """Export vehicle comparison matrix to CSV with original values, normalized values, and citations."""
        rows_out: list[dict[str, str | int | float]] = []
        fieldnames: list[str] = []
        seen_cols: set[str] = set()

        for row in report.rows:
            rec: dict[str, str | int | float] = {
                "component": row.canonical_component,
                "spec_type": row.spec_type,
                "subsystem": row.subsystem,
                "comparable": "yes" if row.comparable else "no",
                "has_variance": "yes" if row.has_variance else "no",
            }
            for cell in row.cells:
                prefix = cell.vehicle_label
                if cell.status == "present":
                    orig = f"{cell.original_value or ''} {cell.original_unit or ''}".strip()
                    norm = (
                        f"{cell.normalized_value} {cell.normalized_unit}"
                        if cell.normalized_value is not None and cell.normalized_unit
                        else ""
                    )
                    rec[f"{prefix} (Original)"] = orig
                    rec[f"{prefix} (Normalized)"] = norm
                    rec[f"{prefix} (Source)"] = f"p.{cell.page_number} ({cell.section or ''})"
                else:
                    rec[f"{prefix} (Original)"] = cell.status.upper()
                    rec[f"{prefix} (Normalized)"] = ""
                    rec[f"{prefix} (Source)"] = ""
            for k in rec:
                if k not in seen_cols:
                    seen_cols.add(k)
                    fieldnames.append(k)
            rows_out.append(rec)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            for r in rows_out:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        return output.getvalue()

    @classmethod
    def comparison_to_json(cls, report: ComparisonResponse, indent: int = 2) -> str:
        """Export vehicle comparison report as structured JSON."""
        return json.dumps(report.model_dump(), indent=indent, ensure_ascii=False)
