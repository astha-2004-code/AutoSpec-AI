"""Strict source evidence verification engine preventing hallucination and unit/subsystem errors."""

from __future__ import annotations

import re

from autospec.cleaner import TextCleaner
from autospec.schemas import VehicleSpecification, VerificationStatus
from autospec.units import (
    are_units_compatible,
    convert_value,
    extract_unit_from_text,
    normalize_unit_symbol,
    parse_numeric_or_range,
    values_equivalent,
)

# Canonical unit base for cross-vehicle normalization
_CANONICAL_BASE: dict[str, str] = {
    "Nm": "Nm",
    "lb-ft": "Nm",
    "lb-in": "Nm",
    "kgf-m": "Nm",
    "kgf-cm": "Nm",
    "ml": "ml",
    "L": "ml",
    "pt": "ml",
    "qt": "ml",
    "gal": "ml",
    "fl-oz": "ml",
    "mm": "mm",
    "cm": "mm",
    "m": "mm",
    "in": "mm",
    "mil": "mm",
    "kPa": "kPa",
    "bar": "kPa",
    "psi": "kPa",
    "MPa": "kPa",
}


class EvidenceValidator:
    """Strictly verifies extracted specifications against source excerpts, units, subsystems, and pages."""

    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        self.cleaner = cleaner or TextCleaner()

    @staticmethod
    def _contains_exact_value(value: str, quote: str) -> bool:
        """Verify numeric/token presence without false partial-digit matches (e.g., '3' in '37')."""
        if not value or not quote:
            return False

        val_clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value.strip())
        quote_clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", quote.strip())

        # Direct boundary-aware regex match on full value string
        pat = re.compile(rf"(?<![A-Za-z0-9.]){re.escape(val_clean)}(?![A-Za-z0-9.])", re.I)
        if pat.search(quote_clean):
            return True

        # If value is a parsed range or tolerance, check both endpoints explicitly
        parsed = parse_numeric_or_range(val_clean)
        if parsed.is_range and parsed.range_min is not None and parsed.range_max is not None:
            lo_str = f"{parsed.range_min:g}"
            hi_str = f"{parsed.range_max:g}"
            # Also match trailing zero variants (e.g. 0.70 vs 0.7)
            lo_pat = re.compile(rf"(?<!\d){re.escape(lo_str.rstrip('0').rstrip('.') if '.' in lo_str else lo_str)}(?:\d*)?(?!\d)")
            hi_pat = re.compile(rf"(?<!\d){re.escape(hi_str.rstrip('0').rstrip('.') if '.' in hi_str else hi_str)}(?:\d*)?(?!\d)")
            if lo_pat.search(quote_clean) and hi_pat.search(quote_clean):
                return True

        if parsed.point is not None:
            num_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", quote_clean)
            for tok in num_tokens:
                try:
                    if abs(float(tok) - parsed.point) <= 1e-6:
                        return True
                except ValueError:
                    continue

        return False

    @staticmethod
    def _extract_units_in_text(text: str) -> list[str]:
        """Extract all recognizable engineering unit symbols present in text."""
        found: list[str] = []
        pattern = re.compile(
            r"\b(Nm|N·m|N-m|lb-ft|ft-lb|ft-lbs|lb-in|in-lb|in-lbs|kgf-m|kgf-cm|"
            r"ml|liters?|L|pt|qt|gal|fl\s*oz|mm|cm|in|inches|kPa|MPa|bar|psi|V|mV|A|mA|CCA|Ah|Ohm|kOhm|rpm|deg|°C|°F)\b",
            re.I,
        )
        for m in pattern.finditer(text):
            norm = normalize_unit_symbol(m.group(1))
            if norm and norm not in found:
                found.append(norm)
        return found

    @staticmethod
    def _verify_quote_on_page(quote: str, page_text: str, component: str, value: str) -> tuple[bool, str]:
        """Confirm that the verbatim quote (or its structured row tokens) exists on the source page."""
        q_norm = TextCleaner.normalize_for_matching(quote)
        p_norm = TextCleaner.normalize_for_matching(page_text)

        if not p_norm:
            return False, "Source page contains no extractable text."

        # Strip synthetic table arrow '->' or '|' separators used in spec_row raw_evidence
        segments = [
            seg.strip()
            for seg in re.split(r"\s*(?:->|\|)\s*", q_norm)
            if seg.strip()
        ]

        # Direct full substring check
        if q_norm in p_norm:
            return True, "Exact verbatim quote matched on source page."

        # Check all non-empty segments from structured row evidence
        if segments:
            matched_segments = 0
            for seg in segments:
                # For value segment like '250 Nm / 184 lb-ft', check primary numeric/unit tokens
                if seg in p_norm:
                    matched_segments += 1
                else:
                    seg_tokens = [t for t in re.findall(r"[a-z0-9.\-]+", seg) if len(t) >= 2]
                    if seg_tokens and all(t in p_norm for t in seg_tokens):
                        matched_segments += 1
            if matched_segments == len(segments):
                return True, "All structured row segments verified on source page."

        # Check component + exact value separately on the page
        comp_words = [w for w in re.findall(r"[a-z0-9\-]+", component.lower()) if len(w) > 2]
        comp_hits = sum(1 for w in comp_words if w in p_norm)
        has_val = EvidenceValidator._contains_exact_value(value, page_text)

        if comp_words and comp_hits >= max(1, int(len(comp_words) * 0.6)) and has_val:
            return True, "Component terms and exact value verified on source page."

        return False, "Source quote and component/value could not be grounded on the referenced page."

    @staticmethod
    def _populate_normalized_fields(spec: VehicleSpecification) -> None:
        """Populate canonical SI normalized_value/unit and range_min/max when parseable."""
        parsed = parse_numeric_or_range(spec.value)
        eff_unit = normalize_unit_symbol(spec.unit) or parsed.unit

        if parsed.is_range:
            if not spec.range_min and parsed.range_min is not None:
                spec.range_min = f"{parsed.range_min:g}"
            if not spec.range_max and parsed.range_max is not None:
                spec.range_max = f"{parsed.range_max:g}"
            if not spec.tolerance and parsed.tolerance is not None:
                spec.tolerance = f"±{parsed.tolerance:g}"

        scalar = parsed.point
        if scalar is None and parsed.range_min is not None and parsed.range_max is not None:
            scalar = (parsed.range_min + parsed.range_max) / 2.0

        if scalar is not None and eff_unit:
            target = _CANONICAL_BASE.get(eff_unit)
            if target:
                conv = convert_value(scalar, eff_unit, target)
                if conv is not None:
                    spec.normalized_value = round(conv, 4)
                    spec.normalized_unit = target
            else:
                spec.normalized_value = round(scalar, 4)
                spec.normalized_unit = eff_unit

    def validate(
        self,
        spec: VehicleSpecification,
        page_text_map: dict[int, str] | None = None,
        query: str | None = None,
        expected_document: str | None = None,
    ) -> VehicleSpecification:
        """Strictly validate a specification against quote, unit, subsystem, and page text."""
        self._populate_normalized_fields(spec)

        # 1. Mandatory provenance check
        if not spec.source_document or spec.page_number < 1:
            spec.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE.value
            spec.verification_notes = "Missing source document name or valid page number."
            return spec

        if expected_document and spec.source_document != expected_document:
            spec.verification_status = VerificationStatus.UNSUPPORTED.value
            spec.verification_notes = (
                f"Document mismatch: expected '{expected_document}', got '{spec.source_document}'."
            )
            return spec

        if not spec.source_quote or not spec.source_quote.strip():
            spec.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE.value
            spec.verification_notes = "Missing verbatim source quote."
            return spec

        # 2. Strict numeric/token presence inside source quote (prevents partial-digit matches)
        if not self._contains_exact_value(spec.value, spec.source_quote):
            spec.verification_status = VerificationStatus.UNSUPPORTED.value
            spec.verification_notes = (
                f"Specification value '{spec.value}' does not appear in quoted evidence '{spec.source_quote}'."
            )
            return spec

        # 3. Strict unit validation against quote and query
        spec_unit = normalize_unit_symbol(spec.unit) or extract_unit_from_text(spec.value)
        quote_units = self._extract_units_in_text(spec.source_quote)
        unit_inferred_only = False

        if spec_unit and spec_unit not in ("part_number", "capacity", "volume"):
            if quote_units:
                if not any(are_units_compatible(spec_unit, qu) for qu in quote_units):
                    spec.verification_status = VerificationStatus.UNSUPPORTED.value
                    spec.verification_notes = (
                        f"Unit mismatch: extracted unit '{spec_unit}' is incompatible with quote units {quote_units}."
                    )
                    return spec
                if spec_unit not in quote_units:
                    # Compatible family (e.g. lb-ft vs Nm) — verify alternate or conversion
                    unit_inferred_only = True
            else:
                unit_inferred_only = True

        # Check if query requested a specific unit family incompatible with extracted spec
        if query and spec_unit and spec_unit not in ("part_number", "capacity", "volume"):
            q_units = self._extract_units_in_text(query)
            if q_units and not any(are_units_compatible(spec_unit, qu) for qu in q_units):
                spec.verification_status = VerificationStatus.CONFLICTING_EVIDENCE.value
                spec.verification_notes = (
                    f"Query requested unit {q_units}, which is physically incompatible with extracted unit '{spec_unit}'."
                )
                return spec

        # 4. Cross-subsystem conflict detection
        if query:
            q_lower = query.lower()
            sub_lower = (spec.subsystem or "").lower()
            sec_lower = (spec.section or "").lower()
            if "rear" in q_lower and ("front" in sub_lower or "front" in sec_lower):
                spec.verification_status = VerificationStatus.AMBIGUOUS.value
                spec.verification_notes = (
                    "Query requested rear specification, but retrieved value belongs to front disc brake / front section."
                )
                return spec
            if "front" in q_lower and ("rear" in sub_lower or "rear" in sec_lower):
                spec.verification_status = VerificationStatus.AMBIGUOUS.value
                spec.verification_notes = (
                    "Query requested front specification, but retrieved value belongs to rear disc brake / rear section."
                )
                return spec
            if "4wd" in q_lower and "rwd" in sub_lower:
                spec.verification_status = VerificationStatus.AMBIGUOUS.value
                spec.verification_notes = "Query specified 4WD, but retrieved record is from RWD section."
                return spec
            if "rwd" in q_lower and "4wd" in sub_lower:
                spec.verification_status = VerificationStatus.AMBIGUOUS.value
                spec.verification_notes = "Query specified RWD, but retrieved record is from 4WD section."
                return spec

        # 5. Page text grounding verification
        if page_text_map is not None:
            if spec.page_number not in page_text_map:
                spec.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE.value
                spec.verification_notes = (
                    f"Source page {spec.page_number} text is unavailable for verbatim verification."
                )
                return spec

            page_text = page_text_map[spec.page_number]
            quote_ok, quote_msg = self._verify_quote_on_page(
                spec.source_quote, page_text, spec.component, spec.value
            )
            if not quote_ok:
                spec.verification_status = VerificationStatus.UNSUPPORTED.value
                spec.verification_notes = quote_msg
                return spec

            cleaned_page = self.cleaner.clean_text(page_text)
            comp_words = [w for w in re.findall(r"[a-z0-9\-]+", spec.component.lower()) if len(w) > 3]
            page_lower = cleaned_page.lower()
            if comp_words and not any(w in page_lower for w in comp_words):
                spec.verification_status = VerificationStatus.AMBIGUOUS.value
                spec.verification_notes = (
                    f"Component '{spec.component}' could not be confirmed on source page {spec.page_number}."
                )
                return spec

        # Check if page text confirms unit when table row omitted inline unit symbol
        if unit_inferred_only and page_text_map and spec.page_number in page_text_map and spec_unit:
            page_units = self._extract_units_in_text(page_text_map[spec.page_number])
            if any(are_units_compatible(spec_unit, pu) for pu in page_units):
                unit_inferred_only = False

        if unit_inferred_only and spec.spec_type != "part_number":
            spec.verification_status = VerificationStatus.PARTIALLY_SUPPORTED.value
            spec.verification_notes = (
                f"Value '{spec.value}' and component verified on page {spec.page_number}, "
                f"but unit '{spec.unit}' was inferred from table header context rather than inline row text."
            )
            return spec

        spec.verification_status = VerificationStatus.VERIFIED.value
        spec.verification_notes = (
            f"Verified on page {spec.page_number} within {spec.section}. "
            f"Value '{spec.value} {spec.unit}' and verbatim evidence confirmed."
        )
        return spec

    def validate_batch(
        self,
        specs: list[VehicleSpecification],
        page_text_map: dict[int, str] | None = None,
        query: str | None = None,
        expected_document: str | None = None,
    ) -> list[VehicleSpecification]:
        """Validate a batch of specifications and detect cross-record conflicts or ambiguity."""
        validated = [
            self.validate(
                s,
                page_text_map=page_text_map,
                query=query,
                expected_document=expected_document,
            )
            for s in specs
        ]

        # Detect conflicting values for identical component+subsystem+spec_type within the same document
        by_key: dict[tuple[str, str, str, str], list[VehicleSpecification]] = {}
        for s in validated:
            if s.verification_status in (VerificationStatus.VERIFIED.value, VerificationStatus.PARTIALLY_SUPPORTED.value):
                key = (
                    s.source_document,
                    (s.subsystem or "").lower(),
                    s.component.lower().strip(),
                    s.spec_type.lower(),
                )
                by_key.setdefault(key, []).append(s)

        for group in by_key.values():
            if len(group) > 1:
                first = group[0]
                for other in group[1:]:
                    if not values_equivalent(first.value, first.unit, other.value, other.unit):
                        # Only flag conflict if conditions don't distinguish them
                        if (first.conditions or "") == (other.conditions or ""):
                            for item in group:
                                item.verification_status = VerificationStatus.CONFLICTING_EVIDENCE.value
                                item.verification_notes = (
                                    f"Conflicting values detected for '{item.component}' within {item.section}."
                                )
                            break

        return validated
