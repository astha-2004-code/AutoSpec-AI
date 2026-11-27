"""Engineering unit normalization, conversion, and tolerance/range parsing for automotive specs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Canonical dimension families -> {canonical_unit: factor_to_base}
# Torque base: Nm
# Volume base: ml
# Length base: mm
# Pressure base: kPa
# Mass/Force base: N
# Electrical base: V / A / Ohm
UNIT_FAMILIES: dict[str, dict[str, float]] = {
    "torque": {
        "Nm": 1.0,
        "lb-ft": 1.3558179483,
        "lb-in": 0.1129848290,
        "kgf-m": 9.80665,
        "kgf-cm": 0.0980665,
    },
    "volume": {
        "ml": 1.0,
        "L": 1000.0,
        "pt": 473.176473,
        "qt": 946.352946,
        "gal": 3785.411784,
        "fl-oz": 29.5735296,
    },
    "length": {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "in": 25.4,
        "mil": 0.0254,
    },
    "pressure": {
        "kPa": 1.0,
        "bar": 100.0,
        "psi": 6.89475729,
        "MPa": 1000.0,
    },
    "electrical_voltage": {
        "V": 1.0,
        "mV": 0.001,
    },
    "electrical_current": {
        "A": 1.0,
        "mA": 0.001,
        "CCA": 1.0,
    },
    "electrical_capacity": {
        "Ah": 1.0,
    },
    "electrical_resistance": {
        "Ohm": 1.0,
        "kOhm": 1000.0,
    },
    "speed": {
        "rpm": 1.0,
    },
    "angle": {
        "deg": 1.0,
    },
}

UNIT_ALIASES: dict[str, str] = {
    # Torque
    "nm": "Nm",
    "n-m": "Nm",
    "n·m": "Nm",
    "n m": "Nm",
    "newton-meter": "Nm",
    "newton-meters": "Nm",
    "newton meter": "Nm",
    "newton meters": "Nm",
    "lb-ft": "lb-ft",
    "lb·ft": "lb-ft",
    "lbft": "lb-ft",
    "ft-lb": "lb-ft",
    "ft-lbs": "lb-ft",
    "lb-fts": "lb-ft",
    "foot-pound": "lb-ft",
    "foot-pounds": "lb-ft",
    "lb-in": "lb-in",
    "lb·in": "lb-in",
    "lbin": "lb-in",
    "in-lb": "lb-in",
    "in-lbs": "lb-in",
    "inch-pound": "lb-in",
    "inch-pounds": "lb-in",
    "kgf-m": "kgf-m",
    "kg-m": "kgf-m",
    "kgf-cm": "kgf-cm",
    "kg-cm": "kgf-cm",
    # Volume
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "cc": "ml",
    "l": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "pt": "pt",
    "pts": "pt",
    "pint": "pt",
    "pints": "pt",
    "qt": "qt",
    "qts": "qt",
    "quart": "qt",
    "quarts": "qt",
    "gal": "gal",
    "gallon": "gal",
    "gallons": "gal",
    "fl oz": "fl-oz",
    "fl-oz": "fl-oz",
    "oz": "fl-oz",
    # Length
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "m": "m",
    "in": "in",
    "inch": "in",
    "inches": "in",
    '"': "in",
    "mil": "mil",
    "mils": "mil",
    # Pressure
    "kpa": "kPa",
    "bar": "bar",
    "bars": "bar",
    "psi": "psi",
    "mpa": "MPa",
    # Temperature
    "c": "C",
    "°c": "C",
    "degc": "C",
    "celsius": "C",
    "f": "F",
    "°f": "F",
    "degf": "F",
    "fahrenheit": "F",
    # Electrical
    "v": "V",
    "volt": "V",
    "volts": "V",
    "vdc": "V",
    "mv": "mV",
    "a": "A",
    "amp": "A",
    "amps": "A",
    "amperes": "A",
    "ma": "mA",
    "cca": "CCA",
    "ah": "Ah",
    "amp-hours": "Ah",
    "ohm": "Ohm",
    "ohms": "Ohm",
    "ω": "Ohm",
    "kohm": "kOhm",
    "kohms": "kOhm",
    # Other
    "rpm": "rpm",
    "deg": "deg",
    "°": "deg",
    "degrees": "deg",
    "part_number": "part_number",
    "capacity": "capacity",
    "volume": "volume",
}


@dataclass(frozen=True)
class ParsedQuantity:
    """Structured representation of a point value, range, or tolerance."""

    raw: str
    point: float | None = None
    range_min: float | None = None
    range_max: float | None = None
    tolerance: float | None = None
    unit: str = ""

    @property
    def is_range(self) -> bool:
        return self.range_min is not None and self.range_max is not None


def normalize_unit_symbol(unit: str | None) -> str:
    """Normalize engineering unit representations into canonical symbols."""
    if not unit:
        return ""
    cleaned = unit.strip()
    if not cleaned:
        return ""
    low = re.sub(r"\s+", " ", cleaned.lower()).strip()
    if low in UNIT_ALIASES:
        return UNIT_ALIASES[low]
    compact = low.replace(" ", "")
    if compact in UNIT_ALIASES:
        return UNIT_ALIASES[compact]
    return cleaned


def get_unit_family(unit: str | None) -> str | None:
    """Return physical dimension family for a canonical or aliased unit."""
    norm = normalize_unit_symbol(unit)
    if not norm:
        return None
    if norm in ("C", "F"):
        return "temperature"
    for family, mapping in UNIT_FAMILIES.items():
        if norm in mapping:
            return family
    return None


def are_units_compatible(u1: str | None, u2: str | None) -> bool:
    """Return True if two units represent the same physical dimension."""
    n1 = normalize_unit_symbol(u1)
    n2 = normalize_unit_symbol(u2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    # Generic placeholder units used in legacy capacity records
    if {n1, n2} <= {"ml", "L", "pt", "qt", "gal", "fl-oz", "capacity", "volume"}:
        return True
    f1 = get_unit_family(n1)
    f2 = get_unit_family(n2)
    return f1 is not None and f1 == f2


def convert_value(val: float, from_unit: str, to_unit: str) -> float | None:
    """Convert a numeric value between compatible engineering units."""
    u_from = normalize_unit_symbol(from_unit)
    u_to = normalize_unit_symbol(to_unit)
    if not u_from or not u_to:
        return None
    if u_from == u_to:
        return val

    if u_from == "C" and u_to == "F":
        return val * 9.0 / 5.0 + 32.0
    if u_from == "F" and u_to == "C":
        return (val - 32.0) * 5.0 / 9.0

    f_from = get_unit_family(u_from)
    f_to = get_unit_family(u_to)
    if not f_from or f_from != f_to:
        return None

    factors = UNIT_FAMILIES[f_from]
    base_val = val * factors[u_from]
    return base_val / factors[u_to]


def parse_numeric_or_range(raw: str | None) -> ParsedQuantity:
    """Parse scalar, comma-separated, range (a - b), or +/- tolerance strings."""
    if not raw:
        return ParsedQuantity(raw="")

    text = raw.strip()
    # Replace unicode dashes and remove commas inside numbers
    cleaned = text.replace("\u2013", "-").replace("\u2212", "-")
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)

    # Tolerance pattern: e.g. "32.0 +/- 0.5" or "32 ± 0.5"
    tol_m = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*([A-Za-z°%/-]+)?",
        cleaned,
    )
    if tol_m:
        center = float(tol_m.group(1))
        tol = float(tol_m.group(2))
        unit = normalize_unit_symbol(tol_m.group(3) or "")
        return ParsedQuantity(
            raw=text,
            point=center,
            range_min=round(center - tol, 6),
            range_max=round(center + tol, 6),
            tolerance=tol,
            unit=unit,
        )

    # Range pattern: e.g. "0.7 - 0.9 mm" or "25 to 30 Nm" or "0.20-0.30"
    range_m = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:-|to)\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z°%/-]+)?",
        cleaned,
        re.I,
    )
    if range_m:
        lo = float(range_m.group(1))
        hi = float(range_m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        unit = normalize_unit_symbol(range_m.group(3) or "")
        return ParsedQuantity(
            raw=text,
            point=None,
            range_min=lo,
            range_max=hi,
            unit=unit,
        )

    # Scalar pattern: e.g. "1420 ml (3 pt)" -> captures 1420 and ml
    scalar_m = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*([A-Za-z°%]+(?:-[A-Za-z]+)?)?",
        cleaned,
    )
    if scalar_m:
        val = float(scalar_m.group(1))
        unit = normalize_unit_symbol(scalar_m.group(2) or "")
        return ParsedQuantity(raw=text, point=val, unit=unit)

    return ParsedQuantity(raw=text)


def extract_unit_from_text(raw: str | None) -> str:
    """Extract canonical unit symbol if embedded inside a value string like '1,420 ml (3 pt)'."""
    if not raw:
        return ""
    parsed = parse_numeric_or_range(raw)
    return parsed.unit


def values_equivalent(
    val1: str | None,
    unit1: str | None,
    val2: str | None,
    unit2: str | None,
    rel_tol: float = 0.03,
) -> bool:
    """Compare two specification values accurately, supporting ranges and unit conversions."""
    if not val1 or not val2:
        return False

    s1 = re.sub(r"\s+", " ", val1.strip().lower())
    s2 = re.sub(r"\s+", " ", val2.strip().lower())
    if s1 == s2 and (not unit1 or not unit2 or are_units_compatible(unit1, unit2)):
        return True

    p1 = parse_numeric_or_range(val1)
    p2 = parse_numeric_or_range(val2)

    u1 = normalize_unit_symbol(unit1) or p1.unit
    u2 = normalize_unit_symbol(unit2) or p2.unit

    # Reject if both have explicit incompatible units
    if u1 and u2 and not are_units_compatible(u1, u2):
        # Handle part numbers (exact token comparison without numeric equivalence)
        if u1 == "part_number" or u2 == "part_number":
            return s1 == s2
        return False

    # Range vs Range comparison
    if p1.is_range and p2.is_range:
        lo1, hi1 = p1.range_min, p1.range_max
        lo2, hi2 = p2.range_min, p2.range_max
        if lo1 is None or hi1 is None or lo2 is None or hi2 is None:
            return False
        if u1 and u2 and u1 != u2:
            c_lo = convert_value(lo1, u1, u2)
            c_hi = convert_value(hi1, u1, u2)
            if c_lo is None or c_hi is None:
                return False
            lo1, hi1 = c_lo, c_hi
        return math.isclose(lo1, lo2, rel_tol=rel_tol, abs_tol=1e-3) and math.isclose(
            hi1, hi2, rel_tol=rel_tol, abs_tol=1e-3
        )

    # Scalar vs Scalar comparison
    if p1.point is not None and p2.point is not None:
        v1 = p1.point
        v2 = p2.point
        if u1 and u2 and u1 != u2:
            converted = convert_value(v1, u1, u2)
            if converted is None:
                return False
            v1 = converted
        return math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=1e-3)

    # Non-numeric strings (e.g., part numbers, material codes)
    if u1 == "part_number" or u2 == "part_number" or (p1.point is None and p2.point is None):
        return s1 == s2

    return False

