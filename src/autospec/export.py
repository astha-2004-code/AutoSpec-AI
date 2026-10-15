"""Tabular and structured export utilities using Pandas."""

from __future__ import annotations

import io
import json

import pandas as pd

from autospec.schemas import VehicleSpecification


class ExportManager:
    """Manages CSV and JSON exports of vehicle specifications."""

    EXPORT_COLUMNS = [
        "component",
        "spec_type",
        "value",
        "unit",
        "alternate_value",
        "alternate_unit",
        "section",
        "subsystem",
        "page_number",
        "source_document",
        "source_quote",
        "part_number",
        "material",
        "verification_status",
        "extraction_method",
    ]

    @classmethod
    def to_dataframe(cls, specs: list[VehicleSpecification]) -> pd.DataFrame:
        """Convert specification list to a well-ordered Pandas DataFrame."""
        if not specs:
            return pd.DataFrame(columns=cls.EXPORT_COLUMNS)

        records = [s.to_export_dict() for s in specs]
        df = pd.DataFrame(records)

        # Ensure all standard columns exist
        for col in cls.EXPORT_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Reorder columns with core attributes first
        ordered_cols = [c for c in cls.EXPORT_COLUMNS if c in df.columns] + [
            c for c in df.columns if c not in cls.EXPORT_COLUMNS
        ]
        return df[ordered_cols]

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
