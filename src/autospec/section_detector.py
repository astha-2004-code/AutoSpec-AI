"""Section, subsystem, and vehicle metadata detector for automotive workshop manuals."""

from __future__ import annotations

import re
from dataclasses import dataclass

from autospec.schemas import VehicleMetadata


@dataclass
class SectionInfo:
    """Metadata regarding a detected workshop manual section."""

    section_id: str  # e.g., '206-03' or 'CH-03'
    title: str  # e.g., 'Front Disc Brake'
    full_heading: str  # e.g., 'SECTION 206-03: Front Disc Brake'
    subsystem: str  # e.g., 'front_brakes'
    is_spec_section: bool  # True if page contains specification tables
    procedure_date: str = ""
    parent_section: str = ""


class SectionDetector:
    """Parses and tracks manual sections, hierarchy, and vehicle subsystems."""

    # Ford / SAE style: SECTION 206-03: Front Disc Brake
    SECTION_HEADER_RE = re.compile(
        r"SECTION\s+([A-Z0-9]{1,4}-[0-9]{1,3}[A-Za-z]?)\s*[:\-]\s*([^\n\r]+)",
        re.IGNORECASE,
    )

    # Generic OEM style: SECTION 4: Engine Mechanical / CHAPTER 03 - Cooling System
    GENERIC_SECTION_RE = re.compile(
        r"^(?:SECTION|CHAPTER|MODULE|GROUP)\s+([0-9]{1,3}(?:\.[0-9]+)?[A-Za-z]?)\s*[:\-—]\s*([^\n\r]+)",
        re.IGNORECASE | re.MULTILINE,
    )

    PROCEDURE_DATE_RE = re.compile(
        r"Procedure\s+revision\s+date\s*:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})",
        re.IGNORECASE,
    )

    VEHICLE_HEADER_RE = re.compile(
        r"\b(19\d{2}|20[0-3]\d)\s+([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)?)\s+([A-Z0-9][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)?)\s+(?:Workshop|Service|Repair|Technical)\s+Manual",
        re.IGNORECASE,
    )

    SUBSYSTEM_MAP = {
        "204-01A": "front_suspension_rwd",
        "204-01B": "front_suspension_4wd",
        "204-02": "rear_suspension",
        "204-04": "wheels_and_tires",
        "205-00": "driveline_general",
        "205-01": "driveshaft",
        "205-02A": "rear_axle_88",
        "205-02B": "rear_axle_975",
        "205-03": "front_drive_axle",
        "205-04": "front_halfshafts",
        "206-00": "brake_system_general",
        "206-03": "front_brakes",
        "206-04": "rear_brakes",
        "206-05": "parking_brake",
        "206-06": "hydraulic_brake_actuation",
        "206-07": "power_brake_actuation",
        "206-09": "anti_lock_brakes_abs",
    }

    KNOWN_MAKES = (
        "Ford", "Apex Motors", "Toyota", "Honda", "Chevrolet", "GMC", "Ram",
        "Jeep", "Dodge", "BMW", "Mercedes-Benz", "Audi", "Volkswagen",
        "Subaru", "Mazda", "Nissan", "Hyundai", "Kia", "Volvo", "Porsche",
    )

    def __init__(self) -> None:
        self.current_section: SectionInfo | None = None

    def reset(self) -> None:
        """Reset section state for a new document."""
        self.current_section = None

    def detect_vehicle_metadata(self, text: str, filename: str = "") -> VehicleMetadata:
        """Infer vehicle manufacturer, model, and year from cover/header text or filename."""
        year: int | None = None
        make = "Unknown"
        model = "Unknown"

        combined = f"{text[:3000]}\n{filename}"

        m = self.VEHICLE_HEADER_RE.search(text[:3000])
        if m:
            year = int(m.group(1))
            raw_rest = f"{m.group(2)} {m.group(3)}".strip()
            for known in self.KNOWN_MAKES:
                if raw_rest.lower().startswith(known.lower()):
                    make = known
                    model = raw_rest[len(known):].strip() or "Unknown"
                    break
            if make == "Unknown":
                make = m.group(2).strip()
                model = m.group(3).strip()
            return VehicleMetadata(manufacturer=make, model=model, year=year)

        year_m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", combined)
        if year_m:
            year = int(year_m.group(1))

        for known in self.KNOWN_MAKES:
            idx = combined.lower().find(known.lower())
            if idx != -1:
                make = known
                after = combined[idx + len(known):].strip(" _-:")
                model_m = re.match(r"^([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)?)", after)
                if model_m:
                    candidate = model_m.group(1).strip()
                    if candidate.lower() not in ("workshop", "service", "manual", "repair"):
                        model = candidate
                break

        return VehicleMetadata(manufacturer=make, model=model, year=year)

    def detect_in_page(self, text: str, page_number: int) -> SectionInfo | None:
        """Scan a page's text for section headers, updating or inheriting active context."""
        upper = text.upper()
        is_spec = any(
            marker in upper
            for marker in (
                "SPECIFICATIONS",
                "TORQUE SPECIFICATIONS",
                "GENERAL SPECIFICATIONS",
                "CAPACITIES",
                "TECHNICAL DATA",
                "ELECTRICAL SPECIFICATIONS",
                "MAINTENANCE SCHEDULE",
            )
        )

        date_match = self.PROCEDURE_DATE_RE.search(text)
        proc_date = date_match.group(1) if date_match else ""

        match = self.SECTION_HEADER_RE.search(text)
        if not match:
            match = self.GENERIC_SECTION_RE.search(text)

        if match:
            sec_id = match.group(1).strip().upper()
            sec_title = match.group(2).strip()
            sec_title = re.sub(r"\s+20\d\d\s+[A-Za-z0-9\-]+\s+Workshop.*$", "", sec_title, flags=re.I).strip()
            sec_title = sec_title.rstrip("—- ")

            subsystem = self._classify_subsystem(sec_id, sec_title)
            parent = sec_id.split("-")[0] if "-" in sec_id else sec_id.split(".")[0]
            self.current_section = SectionInfo(
                section_id=sec_id,
                title=sec_title,
                full_heading=f"SECTION {sec_id}: {sec_title}",
                subsystem=subsystem,
                is_spec_section=is_spec,
                procedure_date=proc_date,
                parent_section=parent,
            )
            return self.current_section

        if self.current_section:
            return SectionInfo(
                section_id=self.current_section.section_id,
                title=self.current_section.title,
                full_heading=self.current_section.full_heading,
                subsystem=self.current_section.subsystem,
                is_spec_section=is_spec,
                procedure_date=proc_date or self.current_section.procedure_date,
                parent_section=self.current_section.parent_section,
            )

        # Fallback classification if page has spec markers without explicit section header
        if is_spec:
            subsystem = self._classify_subsystem("GEN", text[:500])
            return SectionInfo(
                section_id="GENERAL",
                title="Technical Specifications",
                full_heading="Technical Specifications",
                subsystem=subsystem,
                is_spec_section=True,
                procedure_date=proc_date,
            )

        return None

    def _classify_subsystem(self, sec_id: str, title: str) -> str:
        """Derive normalized subsystem category based on section code and title."""
        if sec_id in self.SUBSYSTEM_MAP:
            return self.SUBSYSTEM_MAP[sec_id]

        title_lower = title.lower()
        if "front" in title_lower and "brake" in title_lower:
            return "front_brakes"
        if "rear" in title_lower and "brake" in title_lower:
            return "rear_brakes"
        if "parking" in title_lower and "brake" in title_lower:
            return "parking_brake"
        if "brake" in title_lower:
            return "brakes"
        if "front" in title_lower and "suspension" in title_lower:
            return "front_suspension"
        if "rear" in title_lower and "suspension" in title_lower:
            return "rear_suspension"
        if "suspension" in title_lower:
            return "suspension"
        if "wheel" in title_lower or "tire" in title_lower:
            return "wheels_and_tires"
        if "axle" in title_lower or "differential" in title_lower:
            return "axle_differential"
        if "steering" in title_lower:
            return "steering"
        if "cool" in title_lower or "radiator" in title_lower or "thermostat" in title_lower:
            return "cooling"
        if "lubrication" in title_lower or "oil" in title_lower:
            return "lubrication"
        if "engine" in title_lower or "cylinder" in title_lower or "valvetrain" in title_lower:
            return "engine"
        if "transmission" in title_lower or "transaxle" in title_lower or "clutch" in title_lower:
            return "transmission"
        if "electric" in title_lower or "battery" in title_lower or "alternator" in title_lower or "charging" in title_lower or "ignition" in title_lower:
            return "electrical"
        if "fuel" in title_lower or "injection" in title_lower:
            return "fuel_system"
        if "maintenance" in title_lower or "capacit" in title_lower or "fluid" in title_lower:
            return "maintenance"

        return "general"
