"""Tabular, JSON, and comparison report export utilities."""

from __future__ import annotations

import io
import json

import pandas as pd

from autospec.schemas import ComparisonResponse, VehicleSpecification


class ExportManager:
    """Manages CSV, JSON, and Markdown exports of vehicle specifications and comparison reports."""

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
    def to_dataframe(cls, specs: list[VehicleSpecification]) -> pd.DataFrame:
        """Convert specification list to a well-ordered Pandas DataFrame."""
        if not specs:
            return pd.DataFrame(columns=cls.EXPORT_COLUMNS)

        records = [s.to_export_dict() for s in specs]
        df = pd.DataFrame(records)

        for col in cls.EXPORT_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        ordered_cols = [c for c in cls.EXPORT_COLUMNS if c in df.columns] + [
            c for c in df.columns if c not in cls.EXPORT_COLUMNS
        ]
        return pd.DataFrame(df[ordered_cols])

    @classmethod
    def to_csv(cls, specs: list[VehicleSpecification]) -> str:
        """Generate formatted CSV string with UTF-8 encoding."""
        df = cls.to_dataframe(specs)
        output = io.StringIO()
        df.to_csv(output, index=False, encoding="utf-8")
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
            rows_out.append(rec)

        df = pd.DataFrame(rows_out)
        output = io.StringIO()
        df.to_csv(output, index=False, encoding="utf-8")
        return output.getvalue()

    @classmethod
    def comparison_to_json(cls, report: ComparisonResponse, indent: int = 2) -> str:
        """Export vehicle comparison report as structured JSON."""
        return json.dumps(report.model_dump(), indent=indent, ensure_ascii=False)
