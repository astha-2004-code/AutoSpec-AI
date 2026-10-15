"""Section and subsection detector for automotive workshop manuals.

Extracts hierarchical section numbers (e.g., 'SECTION 206-03: Front Disc Brake'),
classifies vehicle subsystems, and detects technical specification blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SectionInfo:
    """Metadata regarding a detected workshop manual section."""

    section_id: str  # e.g., '206-03'
    title: str  # e.g., 'Front Disc Brake'
    full_heading: str  # e.g., 'SECTION 206-03: Front Disc Brake'
    subsystem: str  # e.g., 'front_brakes'
    is_spec_section: bool  # True if page contains SPECIFICATIONS
    procedure_date: str = ""  # e.g., '10/25/2013'


class SectionDetector:
    """Parses and tracks active manual sections and subsystem categories."""

    SECTION_HEADER_RE = re.compile(
        r"SECTION\s+([0-9]{3}-[0-9]{2}[A-Za-z]?)\s*:\s*([^\n\r]+)",
        re.IGNORECASE,
    )

    PROCEDURE_DATE_RE = re.compile(
        r"Procedure\s+revision\s+date\s*:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})",
        re.IGNORECASE,
    )

    # Subsystem mapping rules
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

    def __init__(self) -> None:
        self.current_section: SectionInfo | None = None

    def detect_in_page(self, text: str, page_number: int) -> SectionInfo | None:
        """Scan a page's text for new section headers, updating or maintaining state."""
        match = self.SECTION_HEADER_RE.search(text)
        is_spec = "SPECIFICATIONS" in text.upper() or "TORQUE SPECIFICATIONS" in text.upper()

        date_match = self.PROCEDURE_DATE_RE.search(text)
        proc_date = date_match.group(1) if date_match else ""

        if match:
            sec_id = match.group(1).strip().upper()
            sec_title = match.group(2).strip()
            # Clean off trailing artifacts from section title if any
            sec_title = re.sub(r"\s+20\d\d\s+F-150.*$", "", sec_title, flags=re.IGNORECASE).strip()

            subsystem = self._classify_subsystem(sec_id, sec_title)
            self.current_section = SectionInfo(
                section_id=sec_id,
                title=sec_title,
                full_heading=f"SECTION {sec_id}: {sec_title}",
                subsystem=subsystem,
                is_spec_section=is_spec,
                procedure_date=proc_date,
            )
            return self.current_section

        if self.current_section:
            # Inherit active section context from previous pages
            return SectionInfo(
                section_id=self.current_section.section_id,
                title=self.current_section.title,
                full_heading=self.current_section.full_heading,
                subsystem=self.current_section.subsystem,
                is_spec_section=is_spec,
                procedure_date=proc_date or self.current_section.procedure_date,
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
        if "brake" in title_lower:
            return "brakes"
        if "suspension" in title_lower:
            return "suspension"
        if "axle" in title_lower or "differential" in title_lower:
            return "axle_differential"
        if "steering" in title_lower:
            return "steering"
        if "engine" in title_lower:
            return "engine"
        if "transmission" in title_lower:
            return "transmission"

        return "general"
