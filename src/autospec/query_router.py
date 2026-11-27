"""Query normalization, abbreviation expansion, subsystem disambiguation, and intent routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    """Interpreted structure of a user specification or procedural query."""

    raw_query: str
    normalized_query: str = ""
    target_subsystem: str | None = None
    spec_type: str | None = None
    component_hints: list[str] = field(default_factory=list)
    unit_hint: str | None = None
    part_number_hint: str | None = None
    vehicle_make_hint: str | None = None
    vehicle_model_hint: str | None = None
    vehicle_year_hint: int | None = None
    is_exact: bool = False
    is_procedural: bool = False
    is_comparison: bool = False
    is_follow_up: bool = False
    ambiguity_hint: str | None = None


class QueryRouter:
    """Interprets engineering queries to normalize terminology, route subsystems, and detect ambiguity."""

    ABBREVIATIONS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bft[\-\s]*lbs?\b", re.I), "lb-ft"),
        (re.compile(r"\b(?:in\-lbs?|inch\-pounds?|in\s+lbs)\b", re.I), "lb-in"),
        (re.compile(r"\bn[\-\s·]*m\b", re.I), "Nm"),
        (re.compile(r"\batf\b", re.I), "automatic transmission fluid"),
        (re.compile(r"\bcvt\b", re.I), "continuously variable transmission"),
        (re.compile(r"\babs\b", re.I), "anti-lock brake system"),
        (re.compile(r"\bcca\b", re.I), "cold cranking amps"),
        (re.compile(r"\bdiff\b", re.I), "differential"),
        (re.compile(r"\bcyl\b", re.I), "cylinder"),
        (re.compile(r"\bspecs?\b", re.I), "specification"),
        (re.compile(r"\bseq\b", re.I), "sequence"),
        (re.compile(r"\bcap\b", re.I), "capacity"),
        (re.compile(r"\btemp\b", re.I), "temperature"),
        (re.compile(r"\bdot\s*3\b", re.I), "DOT 3"),
        (re.compile(r"\bdot\s*4\b", re.I), "DOT 4"),
    ]

    FRONT_BRAKE_TERMS = {"front brake", "front disc", "front caliper", "front rotor", "front pad"}
    REAR_BRAKE_TERMS = {"rear brake", "rear disc", "rear caliper", "rear rotor", "rear pad"}
    PARKING_BRAKE_TERMS = {"parking brake", "emergency brake", "handbrake"}

    TORQUE_TERMS = {"torque", "tighten", "tightening", "fastener", "bolt", "nut", "stud", "nm", "lb-ft", "lb-in"}
    CAPACITY_TERMS = {"capacity", "fluid", "fill", "oil", "coolant", "volume", "how much", "quarts", "pints", "liters", "ml", "qt"}
    DIMENSION_TERMS = {"thickness", "minimum thickness", "wear", "clearance", "backlash", "runout", "tolerance", "gap", "diameter", "endplay"}
    PART_NO_TERMS = {"part number", "part #", "part no", "specification code", "spec code"}
    ELECTRICAL_TERMS = {"voltage", "battery", "alternator", "cca", "amperage", "amp", "resistance", "ohm", "fuse", "charging"}
    PRESSURE_TERMS = {"pressure", "psi", "kpa", "bar", "compression", "tire pressure", "oil pressure", "fuel pressure"}
    PROCEDURAL_TERMS = {"sequence", "procedure", "how to", "steps", "bleed", "bleeding", "order", "pattern", "install", "remove"}
    COMPARISON_TERMS = {"compare", "comparison", "versus", "vs", "difference between", "other model", "another model"}
    FOLLOW_UP_PREFIXES = ("what about", "how about", "and the", "and for", "compare it", "what is its", "same for", "now check")

    COMPONENT_PATTERNS = [
        re.compile(r"caliper\s+anchor\s+plate(?:\s+bolts?)?", re.I),
        re.compile(r"caliper\s+guide\s+pin(?:\s+bolts?)?", re.I),
        re.compile(r"caliper\s+flow\s+bolts?", re.I),
        re.compile(r"caliper\s+bleeder\s+screw", re.I),
        re.compile(r"caliper\s+support\s+bracket(?:\s+bolts?)?", re.I),
        re.compile(r"disc\s+shield\s+bolts?", re.I),
        re.compile(r"brake\s+tube\s+fitting", re.I),
        re.compile(r"flexible\s+hose\s+bracket[^\n\r,]+bolts?", re.I),
        re.compile(r"brake\s+pads?", re.I),
        re.compile(r"brake\s+disc|rotor", re.I),
        re.compile(r"brake\s+fluid", re.I),
        re.compile(r"cylinder\s+head(?:\s+bolts?)?", re.I),
        re.compile(r"connecting\s+rod(?:\s+(?:bolts?|bearing|clearance))?", re.I),
        re.compile(r"main\s+bearing(?:\s+(?:cap\s+)?(?:bolts?|clearance))?", re.I),
        re.compile(r"spark\s+plugs?", re.I),
        re.compile(r"wheel\s+lug\s+nuts?|wheel\s+nuts?", re.I),
        re.compile(r"drain\s+plug", re.I),
        re.compile(r"engine\s+oil", re.I),
        re.compile(r"engine\s+coolant|coolant", re.I),
        re.compile(r"transmission\s+fluid", re.I),
        re.compile(r"axle\s+vent\s+valve", re.I),
        re.compile(r"lower\s+ball\s+joint(?:\s+nut)?", re.I),
        re.compile(r"shock\s+absorber(?:\s+[a-z]+)?\s+nuts?", re.I),
        re.compile(r"tie-rod\s+end\s+nut", re.I),
        re.compile(r"flywheel\s+bolts?|flexplate\s+bolts?", re.I),
        re.compile(r"camshaft\s+bearing\s+cap\s+bolts?", re.I),
    ]

    UNIT_PATTERNS = [
        (re.compile(r"\b(?:nm|newton\s*meters?)\b", re.I), "Nm"),
        (re.compile(r"\b(?:lb-ft|foot-pounds?|ft-lbs?)\b", re.I), "lb-ft"),
        (re.compile(r"\b(?:lb-in|inch-pounds?|in-lbs?)\b", re.I), "lb-in"),
        (re.compile(r"\b(?:ml|milliliters?)\b", re.I), "ml"),
        (re.compile(r"\b(?:l|liters?)\b", re.I), "L"),
        (re.compile(r"\b(?:pt|pints?)\b", re.I), "pt"),
        (re.compile(r"\b(?:qt|quarts?)\b", re.I), "qt"),
        (re.compile(r"\b(?:gal|gallons?)\b", re.I), "gal"),
        (re.compile(r"\b(?:mm|millimeters?)\b", re.I), "mm"),
        (re.compile(r"\b(?:in|inches?)\b", re.I), "in"),
        (re.compile(r"\b(?:kpa|kilopascals?)\b", re.I), "kPa"),
        (re.compile(r"\b(?:psi)\b", re.I), "psi"),
        (re.compile(r"\b(?:bar)\b", re.I), "bar"),
    ]

    PART_NO_PATTERN = re.compile(
        r"\b([A-Z0-9]{2,4}-[A-Z0-9\-]+(?:\s*\([A-Za-z]+\))?|WSS-[A-Z0-9\-]+|WSK-[A-Z0-9\-]+|PM-[0-9]-[A-Z]|AX-[0-9]{3})\b",
        re.I,
    )

    KNOWN_MAKES = ("ford", "apex motors", "apex", "toyota", "honda", "chevrolet", "gmc")
    KNOWN_MODELS = ("f-150", "f150", "m400", "s250", "camry", "civic")

    @classmethod
    def normalize_query(cls, query: str) -> str:
        """Normalize whitespace, dashes, and automotive abbreviations."""
        norm = query.strip().replace("\u2013", "-").replace("\u2212", "-")
        for pat, repl in cls.ABBREVIATIONS:
            norm = pat.sub(repl, norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    def parse_query(self, query: str) -> ParsedQuery:
        """Analyze query to route retrieval accurately and detect ambiguity or follow-ups."""
        norm_q = self.normalize_query(query)
        q_lower = norm_q.lower()

        # 1. Target Subsystem
        subsystem: str | None = None
        if any(term in q_lower for term in self.FRONT_BRAKE_TERMS) or ("front" in q_lower and "brake" in q_lower):
            subsystem = "front_brakes"
        elif any(term in q_lower for term in self.REAR_BRAKE_TERMS) or ("rear" in q_lower and "brake" in q_lower):
            subsystem = "rear_brakes"
        elif any(term in q_lower for term in self.PARKING_BRAKE_TERMS):
            subsystem = "parking_brake"
        elif "front suspension" in q_lower:
            subsystem = "front_suspension_rwd"
        elif "rear suspension" in q_lower:
            subsystem = "rear_suspension"
        elif "differential" in q_lower or "axle" in q_lower:
            if "8.8" in q_lower:
                subsystem = "rear_axle_88"
            elif "9.75" in q_lower:
                subsystem = "rear_axle_975"
            else:
                subsystem = "axle_differential"
        elif "coolant" in q_lower or "thermostat" in q_lower or "radiator" in q_lower:
            subsystem = "cooling"
        elif "battery" in q_lower or "alternator" in q_lower or "charging" in q_lower or "spark plug" in q_lower:
            subsystem = "electrical"
        elif "cylinder head" in q_lower or "connecting rod" in q_lower or "main bearing" in q_lower or "camshaft" in q_lower:
            subsystem = "engine"
        elif "transmission" in q_lower:
            subsystem = "transmission"

        # 2. Spec Type
        spec_type: str | None = None
        if any(term in q_lower for term in self.TORQUE_TERMS):
            spec_type = "torque"
        elif any(term in q_lower for term in self.CAPACITY_TERMS):
            spec_type = "fluid_capacity"
        elif any(term in q_lower for term in self.DIMENSION_TERMS):
            spec_type = "dimension_clearance"
        elif any(term in q_lower for term in self.PART_NO_TERMS):
            spec_type = "part_number"
        elif any(term in q_lower for term in self.ELECTRICAL_TERMS):
            spec_type = "electrical"
        elif any(term in q_lower for term in self.PRESSURE_TERMS):
            spec_type = "pressure"

        # 3. Component Hints
        component_hints: list[str] = []
        for pat in self.COMPONENT_PATTERNS:
            match = pat.search(norm_q)
            if match:
                component_hints.append(match.group(0).strip())

        # 4. Unit Hints
        unit_hint: str | None = None
        for pat, unit_name in self.UNIT_PATTERNS:
            if pat.search(norm_q):
                unit_hint = unit_name
                break

        # 5. Part Number Hints
        part_no_hint: str | None = None
        part_match = self.PART_NO_PATTERN.search(norm_q)
        if part_match:
            part_no_hint = part_match.group(0).strip()

        # 6. Vehicle Hints
        make_hint = next((m for m in self.KNOWN_MAKES if m in q_lower), None)
        model_hint = next((m for m in self.KNOWN_MODELS if m in q_lower), None)
        year_m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", norm_q)
        year_hint = int(year_m.group(1)) if year_m else None

        is_procedural = any(term in q_lower for term in self.PROCEDURAL_TERMS)
        is_comparison = any(term in q_lower for term in self.COMPARISON_TERMS)
        is_follow_up = q_lower.startswith(self.FOLLOW_UP_PREFIXES) or bool(
            re.search(r"\b(it|its|that|this|those|them)\b", q_lower) and len(q_lower.split()) <= 10
        )

        ambiguity_hint = None
        if any(h for h in component_hints if "guide pin" in h.lower() or "brake disc" in h.lower()) and not subsystem:
            ambiguity_hint = "Specify front or rear brake assembly if multiple brake specifications exist."

        is_exact = bool(subsystem and component_hints)

        return ParsedQuery(
            raw_query=query,
            normalized_query=norm_q,
            target_subsystem=subsystem,
            spec_type=spec_type,
            component_hints=component_hints,
            unit_hint=unit_hint,
            part_number_hint=part_no_hint,
            vehicle_make_hint=make_hint,
            vehicle_model_hint=model_hint,
            vehicle_year_hint=year_hint,
            is_exact=is_exact,
            is_procedural=is_procedural,
            is_comparison=is_comparison,
            is_follow_up=is_follow_up,
            ambiguity_hint=ambiguity_hint,
        )
