"""Text cleaning and normalization for automotive service manuals.

Preserves decimal values, numerical ranges, tolerances, engineering units, part numbers,
and table structures while filtering out repeated page headers, footers, and URL artifacts.
"""

from __future__ import annotations

import re


class TextCleaner:
    """Cleans automotive manual text while safeguarding technical values."""

    # Patterns for running footers and URLs common in workshop manuals
    FOOTER_PATTERNS = [
        re.compile(r"^\s*Page\s+\d+\s+(?:sur|of)\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*file:///[^\s]+\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*http[s]?://[^\s]+\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$", re.MULTILINE),
        re.compile(r"^\s*20\d\d\s+F-150\s+Workshop\s+Manual\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    # Unicode space and dash mappings
    UNICODE_REPLACEMENTS = {
        "\u00a0": " ",      # non-breaking space
        "\u2002": " ",      # en space
        "\u2003": " ",      # em space
        "\u2009": " ",      # thin space
        "\u202f": " ",      # narrow non-breaking space
        "\u200b": "",       # zero-width space
        "\ufb01": "fi",     # ligature fi
        "\ufb02": "fl",     # ligature fl
        "\u2212": "-",      # minus sign to hyphen
        "\u2013": "-",      # en-dash to hyphen for ranges like 1.5-3.0
        "\u00b7": "-",      # middle dot placeholder
    }

    def clean_text(self, text: str, preserve_lines: bool = True) -> str:
        """Clean page text while preserving line structure and table associations."""
        if not text:
            return ""

        cleaned = text
        for char, repl in self.UNICODE_REPLACEMENTS.items():
            cleaned = cleaned.replace(char, repl)

        for pattern in self.FOOTER_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        lines = cleaned.splitlines()
        processed_lines: list[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                if processed_lines and processed_lines[-1] != "":
                    processed_lines.append("")
                continue

            if line_str in ("z", "*", "•", "·", "—", "-") and len(processed_lines) > 0:
                processed_lines.append(line_str)
                continue

            processed_lines.append(line.rstrip())

        while processed_lines and not processed_lines[-1]:
            processed_lines.pop()

        return "\n".join(processed_lines) if preserve_lines else " ".join(processed_lines)

    @staticmethod
    def normalize_for_matching(text: str) -> str:
        """Normalize text for strict quote/substring verification without losing numbers."""
        if not text:
            return ""
        s = text.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2212", "-")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    @staticmethod
    def is_specification_line(line: str) -> bool:
        """Detect whether a line contains specification units or table markers."""
        lower = line.lower()
        units = (
            "nm", "lb-ft", "lb-in", "ft-lb", "in-lb", "ml", "pt", "qt", "gal",
            "liter", "mm", "in", "kpa", "psi", "bar", "volt", "cca", "ah", "rpm",
            "wsk-", "wss-", "pm-1",
        )
        return any(u in lower for u in units)
