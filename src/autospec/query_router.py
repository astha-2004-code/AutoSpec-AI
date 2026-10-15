"""Query interpretation and routing for automotive technical queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    """Interpreted structure of a user specification query."""

    raw_query: str
    target_subsystem: str | None = None
    spec_type: str | None = None  # 'torque', 'capacity', 'dimension', 'part_number'
    component_hints: list[str] = field(default_factory=list)
    unit_hint: str | None = None
    part_number_hint: str | None = None
    is_exact: bool = False


class QueryRouter:
    """Interprets engineering queries to identify subsystem, component, and specification type."""

    # Subsystem keyword patterns
    FRONT_BRAKE_TERMS = {"front brake", "front disc", "front caliper", "front rotor", "front pad"}
    REAR_BRAKE_TERMS = {"rear brake", "rear disc", "rear caliper", "rear rotor", "rear pad"}
    PARKING_BRAKE_TERMS = {"parking brake", "emergency brake", "handbrake"}

    # Spec type keyword patterns
    TORQUE_TERMS = {"torque", "tighten", "tightening", "fastener", "bolt", "nut", "nm", "lb-ft", "lb-in"}
    CAPACITY_TERMS = {"capacity", "fluid", "fill", "oil", "volume", "how much", "quarts", "pints", "liters", "ml"}
    DIMENSION_TERMS = {"thickness", "minimum thickness", "wear", "clearance", "backlash", "runout", "tolerance", "gap"}
    PART_NO_TERMS = {"part number", "part #", "part no", "specification code", "spec code"}

    # Specific component name patterns in manuals
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
        re.compile(r"axle\s+vent\s+valve", re.I),
        re.compile(r"lower\s+ball\s+joint(?:\s+nut)?", re.I),
        re.compile(r"shock\s+absorber(?:\s+[a-z]+)?\s+nuts?", re.I),
        re.compile(r"tie-rod\s+end\s+nut", re.I),
    ]

    UNIT_PATTERNS = [
        (re.compile(r"\b(?:nm|newton\s*meters?)\b", re.I), "Nm"),
        (re.compile(r"\b(?:lb-ft|foot-pounds?|ft-lbs?)\b", re.I), "lb-ft"),
        (re.compile(r"\b(?:lb-in|inch-pounds?|in-lbs?)\b", re.I), "lb-in"),
        (re.compile(r"\b(?:ml|milliliters?)\b", re.I), "ml"),
        (re.compile(r"\b(?:pt|pints?)\b", re.I), "pt"),
        (re.compile(r"\b(?:qt|quarts?)\b", re.I), "qt"),
        (re.compile(r"\b(?:gal|gallons?)\b", re.I), "gal"),
        (re.compile(r"\b(?:mm|millimeters?)\b", re.I), "mm"),
        (re.compile(r"\b(?:in|inches?)\b", re.I), "in"),
    ]

    PART_NO_PATTERN = re.compile(
        r"\b([A-Z0-9]{2,4}-[A-Z0-9\-]+(?:\s*\([A-Za-z]+\))?|WSS-[A-Z0-9\-]+|WSK-[A-Z0-9\-]+|PM-[0-9]-[A-Z])\b",
        re.I,
    )

    def parse_query(self, query: str) -> ParsedQuery:
        """Analyze query to route retrieval accurately."""
        q_lower = query.lower()

        # 1. Identify Target Subsystem
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

        # 2. Identify Spec Type
        spec_type: str | None = None
        if any(term in q_lower for term in self.TORQUE_TERMS):
            spec_type = "torque"
        elif any(term in q_lower for term in self.CAPACITY_TERMS):
            spec_type = "capacity"
        elif any(term in q_lower for term in self.DIMENSION_TERMS):
            spec_type = "dimension"
        elif any(term in q_lower for term in self.PART_NO_TERMS):
            spec_type = "part_number"

        # 3. Extract Component Hints
        component_hints: list[str] = []
        for pat in self.COMPONENT_PATTERNS:
            match = pat.search(query)
            if match:
                component_hints.append(match.group(0).strip())

        # 4. Extract Unit Hints
        unit_hint: str | None = None
        for pat, unit_name in self.UNIT_PATTERNS:
            if pat.search(query):
                unit_hint = unit_name
                break

        # 5. Extract Part Number Hints
        part_no_hint: str | None = None
        part_match = self.PART_NO_PATTERN.search(query)
        if part_match:
            part_no_hint = part_match.group(0).strip()

        # Specific exact question detection
        is_exact = bool(subsystem and component_hints)

        return ParsedQuery(
            raw_query=query,
            target_subsystem=subsystem,
            spec_type=spec_type,
            component_hints=component_hints,
            unit_hint=unit_hint,
            part_number_hint=part_no_hint,
            is_exact=is_exact,
        )
