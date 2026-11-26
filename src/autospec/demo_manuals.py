"""Self-contained, legally redistributable demonstration workshop manuals for AutoSpec AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf


@dataclass(frozen=True)
class DemoManualSpec:
    """Definition of a synthetic OEM-grade demonstration manual."""

    document_id: str
    filename: str
    display_name: str
    manufacturer: str
    model: str
    year: int


DEMO_MANUALS: list[DemoManualSpec] = [
    DemoManualSpec(
        document_id="demo_apex_m400_2022",
        filename="2022_Apex_M400_Utility_Workshop_Manual.pdf",
        display_name="2022 Apex Motors M400 3.5L V6 4WD Workshop Manual",
        manufacturer="Apex Motors",
        model="M400",
        year=2022,
    ),
    DemoManualSpec(
        document_id="demo_apex_s250_2023",
        filename="2023_Apex_S250_Hybrid_Workshop_Manual.pdf",
        display_name="2023 Apex Motors S250 2.0L Hybrid AWD Workshop Manual",
        manufacturer="Apex Motors",
        model="S250",
        year=2023,
    ),
]


def _draw_manual_page(
    doc: pymupdf.Document,
    title: str,
    printed_label: str,
    blocks: list[dict[str, Any]],
    landscape: bool = False,
) -> None:
    """Render a clean engineering workshop manual page with crisp tables and bounding geometry."""
    width, height = (792, 612) if landscape else (612, 792)
    page = doc.new_page(width=width, height=height)

    # Running header
    page.insert_text((44, 34), title, fontsize=8.5, fontname="helv", color=(0.35, 0.4, 0.48))
    page.draw_line((44, 42), (width - 44, 42), color=(0.82, 0.85, 0.9), width=0.6)

    y = 64.0
    for blk in blocks:
        kind = blk["type"]
        if kind == "section":
            page.insert_text((44, y), blk["text"], fontsize=13, fontname="hebo", color=(0.06, 0.12, 0.22))
            y += 18
            if blk.get("subtitle"):
                page.insert_text((44, y), blk["subtitle"], fontsize=9.5, fontname="helv", color=(0.25, 0.32, 0.42))
                y += 16
            y += 4

        elif kind == "heading":
            page.insert_text((44, y), blk["text"], fontsize=10.5, fontname="hebo", color=(0.12, 0.22, 0.38))
            y += 15

        elif kind == "para":
            rect = pymupdf.Rect(44, y - 9, width - 44, y + 90)
            used = page.insert_textbox(rect, blk["text"], fontsize=9.2, fontname="helv", color=(0.12, 0.14, 0.18))
            lines_est = max(1, blk["text"].count("\n") + 1 + len(blk["text"]) // 88)
            y += max(16, lines_est * 12 + 6) if used >= 0 else 44

        elif kind == "table":
            headers: list[str] = blk["headers"]
            rows: list[list[str]] = blk["rows"]
            col_widths: list[float] = blk["widths"]
            row_h = 19.0
            x_start = 44.0

            # Header background + border + column dividers
            total_w = sum(col_widths)
            page.draw_rect(pymupdf.Rect(x_start, y, x_start + total_w, y + row_h), color=(0.65, 0.72, 0.82), fill=(0.93, 0.95, 0.98), width=0.6)
            cx = x_start
            for h_text, cw in zip(headers, col_widths, strict=False):
                page.draw_line((cx, y), (cx, y + row_h), color=(0.72, 0.78, 0.86), width=0.5)
                page.insert_text((cx + 5, y + 13), h_text, fontsize=8.5, fontname="hebo", color=(0.08, 0.14, 0.24))
                cx += cw
            y += row_h

            for r_idx, row in enumerate(rows):
                fill_c = (0.985, 0.99, 0.995) if r_idx % 2 == 1 else (1.0, 1.0, 1.0)
                page.draw_rect(pymupdf.Rect(x_start, y, x_start + total_w, y + row_h), color=(0.78, 0.82, 0.88), fill=fill_c, width=0.5)
                cx = x_start
                for cell_text, cw in zip(row, col_widths, strict=False):
                    page.draw_line((cx, y), (cx, y + row_h), color=(0.82, 0.86, 0.9), width=0.4)
                    page.insert_text((cx + 5, y + 13), cell_text, fontsize=8.4, fontname="helv", color=(0.1, 0.12, 0.16))
                    cx += cw
                y += row_h
            y += 14

    # Footer with distinct printed page label
    page.draw_line((44, height - 40), (width - 44, height - 40), color=(0.82, 0.85, 0.9), width=0.6)
    page.insert_text((44, height - 25), f"Page {printed_label}", fontsize=8.5, fontname="helv", color=(0.35, 0.4, 0.48))


def _build_m400_pdf(dest: Path) -> None:
    """Create the 2022 Apex Motors M400 3.5L V6 4WD Workshop Manual PDF."""
    doc = pymupdf.open()
    header = "2022 Apex Motors M400 Workshop Manual — 3.5L V6 4WD"

    # Page 1 (Cover, printed 'i')
    _draw_manual_page(doc, header, "i", [
        {"type": "section", "text": "2022 Apex Motors M400 Workshop Manual", "subtitle": "3.5L Twin-Turbo V6 Powertrain | 4WD Utility Chassis | Technical Reference Edition"},
        {"type": "para", "text": "SECTION 100-00: General Vehicle Information\nVehicle Identity: 2022 Apex Motors M400 Utility Pickup (4WD / RWD)\nEngine Code: AX-35TT V6\nTransmission: 10-Speed Automatic (AX-10R80)\nAll torque specifications are listed in Newton-meters (Nm) with Imperial equivalents (lb-ft / lb-in)."},
        {"type": "heading", "text": "Document Control & Manual Sections"},
        {"type": "table", "headers": ["Section Code", "Subsystem", "Printed Page"], "widths": [120, 280, 120], "rows": [
            ["SECTION 206-03", "Front Disc Brake Specifications & Service", "206-03-1"],
            ["SECTION 206-04", "Rear Disc Brake Specifications & Service", "206-04-1"],
            ["SECTION 303-01", "Engine Mechanical — 3.5L V6", "303-01-1"],
            ["SECTION 303-03", "Engine Cooling & Lubrication", "303-03-1"],
            ["SECTION 307-01", "Automatic Transmission & Rear Axle", "307-01-1"],
            ["SECTION 204-01B", "Front Suspension (4WD) & Steering", "204-01-1"],
            ["SECTION 414-01", "Electrical, Charging & Battery", "414-01-1"],
        ]},
    ])

    # Page 2: Front Disc Brake (printed 206-03-1)
    _draw_manual_page(doc, header, "206-03-1", [
        {"type": "section", "text": "SECTION 206-03: Front Disc Brake", "subtitle": "SPECIFICATIONS | Procedure revision date: 03/14/2022"},
        {"type": "heading", "text": "Torque Specifications"},
        {"type": "table", "headers": ["Description", "Nm", "lb-ft", "lb-in"], "widths": [260, 85, 85, 90], "rows": [
            ["Brake caliper anchor plate bolts", "250", "184", "—"],
            ["Brake caliper guide pin bolts", "37", "27", "—"],
            ["Brake caliper flow bolt", "35", "26", "—"],
            ["Brake disc shield bolts", "18", "—", "159"],
            ["Brake flexible hose bracket-to-knuckle bolt", "30", "22", "—"],
            ["Brake caliper bleeder screw", "11", "—", "97"],
            ["Wheel lug nuts", "204", "150", "—"],
        ]},
        {"type": "heading", "text": "General & Dimension Specifications"},
        {"type": "table", "headers": ["Component", "Specification", "Part Number / Grade"], "widths": [220, 140, 160], "rows": [
            ["Brake disc minimum thickness", "32.0 mm (1.259 in)", "AX-BRK-FR350"],
            ["Brake disc maximum runout", "0.05 mm (0.002 in)", "—"],
            ["Brake pad minimum thickness", "3.0 mm (0.118 in)", "AX-PAD-M400F"],
            ["High Performance DOT 3 Brake Fluid", "1,420 ml (3.0 pt)", "PM-1-C / WSS-M6C62-A"],
        ]},
    ])

    # Page 3: Front Disc Brake Service Procedure (printed 206-03-2)
    _draw_manual_page(doc, header, "206-03-2", [
        {"type": "section", "text": "SECTION 206-03: Front Disc Brake", "subtitle": "REMOVAL AND INSTALLATION"},
        {"type": "para", "text": "Brake Caliper & Anchor Plate Installation:\n1. Clean the mounting surfaces using Metal Brake Parts Cleaner (PM-4-A).\n2. Install new brake caliper anchor plate bolts and tighten to 250 Nm (184 lb-ft).\n3. Install the lower brake caliper guide pin bolt first, then the upper guide pin bolt. Tighten both guide pin bolts to 37 Nm (27 lb-ft).\n4. Install the brake caliper flow bolt using 2 new copper washers and tighten to 35 Nm (26 lb-ft).\n5. Pressure bleed the front brake circuit starting at the right front caliper, then left front caliper. Tighten bleeder screw to 11 Nm (97 lb-in)."},
    ])

    # Page 4: Rear Disc Brake (printed 206-04-1)
    _draw_manual_page(doc, header, "206-04-1", [
        {"type": "section", "text": "SECTION 206-04: Rear Disc Brake", "subtitle": "SPECIFICATIONS | Procedure revision date: 03/14/2022"},
        {"type": "heading", "text": "Torque Specifications"},
        {"type": "table", "headers": ["Description", "Nm", "lb-ft", "lb-in"], "widths": [260, 85, 85, 90], "rows": [
            ["Brake caliper support bracket bolts", "150", "111", "—"],
            ["Brake caliper guide pin bolts", "33", "24", "—"],
            ["Brake caliper flow bolt", "35", "26", "—"],
            ["Axle vent valve", "18", "—", "159"],
            ["Parking brake actuator bolts", "25", "18", "—"],
        ]},
        {"type": "heading", "text": "Dimension & Clearance Specifications"},
        {"type": "table", "headers": ["Component", "Specification", "Notes"], "widths": [220, 150, 150], "rows": [
            ["Brake disc minimum thickness", "23.0 mm (0.906 in)", "Rear vented rotor"],
            ["Brake pad minimum thickness", "3.0 mm (0.118 in)", "Discard below limit"],
            ["Parking brake shoe-to-drum clearance", "0.25 - 0.40 mm", "Adjust via star wheel"],
        ]},
    ])

    # Page 5: Engine Mechanical — Part 1 (printed 303-01-1)
    _draw_manual_page(doc, header, "303-01-1", [
        {"type": "section", "text": "SECTION 303-01: Engine Mechanical — 3.5L V6", "subtitle": "SPECIFICATIONS & TORQUE TABLES"},
        {"type": "heading", "text": "Fastener Torque Specifications (Part 1)"},
        {"type": "table", "headers": ["Description", "Nm", "lb-ft", "Condition"], "widths": [230, 80, 80, 130], "rows": [
            ["Cylinder head bolts (Stage 1)", "40", "30", "Lubricate threads"],
            ["Cylinder head bolts (Stage 2)", "95", "70", "Plus 90 deg turn"],
            ["Connecting rod cap bolts", "45", "33", "Torque-to-yield"],
            ["Main bearing cap bolts", "110", "81", "Inner bolts first"],
            ["Camshaft bearing cap bolts", "12", "9", "Sequential inner-to-outer"],
            ["Flywheel / flexplate bolts", "85", "63", "Apply threadlocker"],
            ["Spark plugs", "20", "15", "Dry threads only"],
            ["Oil pan drain plug", "27", "20", "Replace crush washer"],
        ]},
    ])

    # Page 6: Engine Mechanical — Part 2 Multi-page Table + Tightening Sequence (printed 303-01-2)
    _draw_manual_page(doc, header, "303-01-2", [
        {"type": "section", "text": "SECTION 303-01: Engine Mechanical — 3.5L V6", "subtitle": "CLEARANCES & TIGHTENING SEQUENCE (Continued)"},
        {"type": "table", "headers": ["Component", "Specification", "Tolerance / Limit", "Condition"], "widths": [200, 110, 100, 110], "rows": [
            ["Spark plug electrode gap", "0.75 - 0.85 mm", "0.80 +/- 0.05 mm", "Iridium tip"],
            ["Connecting rod bearing clearance", "0.025 - 0.055 mm", "0.070 mm max", "Plastigage check"],
            ["Crankshaft endplay", "0.10 - 0.25 mm", "0.30 mm max", "Thrust bearing"],
            ["Piston ring end gap (Top)", "0.20 - 0.35 mm", "0.50 mm max", "Bore checked"],
            ["Valve lash (Intake, cold)", "0.18 - 0.24 mm", "0.21 +/- 0.03 mm", "Cold engine"],
            ["Cylinder compression pressure", "1,240 kPa (180 psi)", "Min 1,035 kPa", "Wide open throttle"],
        ]},
        {"type": "heading", "text": "Cylinder Head Bolt Tightening Sequence"},
        {"type": "para", "text": "Cylinder head tightening sequence: Tighten the 8 cylinder head bolts in a spiral pattern starting from the center pair (#1, #2), moving diagonally outward (#3, #4, #5, #6), and finishing with the end bolts (#7, #8).\nStage 1: Tighten all bolts in sequence to 40 Nm (30 lb-ft).\nStage 2: Tighten all bolts in sequence to 95 Nm (70 lb-ft).\nStage 3: Rotate each bolt an additional 90 deg in the same sequence."},
    ])

    # Page 7: Engine Cooling & Lubrication (printed 303-03-1)
    _draw_manual_page(doc, header, "303-03-1", [
        {"type": "section", "text": "SECTION 303-03: Engine Cooling & Lubrication", "subtitle": "FLUID CAPACITIES & OPERATING PRESSURES"},
        {"type": "table", "headers": ["Item", "Fill Capacity / Spec", "Part Number / Grade", "Condition"], "widths": [190, 115, 115, 100], "rows": [
            ["Engine oil capacity (with filter)", "5.7 L (6.0 qt)", "SAE 5W-30 Full Synthetic", "AX-OIL-5W30"],
            ["Engine coolant capacity", "12.4 L (13.1 qt)", "OAT Orange 50/50", "AX-COOL-OAT"],
            ["Engine oil pressure (at 2,000 rpm)", "275 - 415 kPa", "40 - 60 psi", "Normal temp 90 C"],
            ["Radiator cap relief pressure", "110 kPa (16 psi)", "AX-CAP-110", "Pressure test"],
            ["Thermostat opening temperature", "88 - 92 deg", "Wax pellet OEM", "Full open 102 C"],
            ["Water pump mounting bolts", "24 Nm (18 lb-ft)", "M8 x 1.25", "Cross pattern"],
            ["Thermostat housing bolts", "12 Nm (106 lb-in)", "M6 x 1.0", "Inspect O-ring"],
        ]},
    ])

    # Page 8: Transmission & Rear Axle (printed 307-01-1)
    _draw_manual_page(doc, header, "307-01-1", [
        {"type": "section", "text": "SECTION 307-01: Automatic Transmission & Rear Axle", "subtitle": "SPECIFICATIONS"},
        {"type": "table", "headers": ["Component / Item", "Specification", "Alternate / Grade", "Notes"], "widths": [210, 110, 100, 100], "rows": [
            ["Automatic transmission fluid capacity", "11.8 L (12.5 qt)", "AX-ULV ATF", "Dry fill spec"],
            ["Transfer case fluid capacity", "1.5 L (1.6 qt)", "AX-TCF-4WD", "4WD models only"],
            ["Rear differential fluid capacity", "2.6 L (5.5 pt)", "SAE 75W-140 Synthetic", "9.75-inch axle"],
            ["Differential ring gear backlash", "0.20 - 0.30 mm", "0.008 - 0.012 in", "Measure 4 points"],
            ["Transmission pan bolts", "11 Nm", "97 lb-in", "Torque in star order"],
            ["Driveshaft flange bolts", "103 Nm", "76 lb-ft", "Apply blue threadlocker"],
        ]},
    ])

    # Page 9: Suspension, Steering & Electrical (printed 204-01-1, landscape rotated layout test!)
    _draw_manual_page(doc, header, "204-01-1", [
        {"type": "section", "text": "SECTION 204-01B: Front Suspension (4WD), Steering & Electrical", "subtitle": "LANDSCAPE REFERENCE TABLE"},
        {"type": "table", "headers": ["Component", "Specification", "Alternate", "Subsystem"], "widths": [240, 130, 130, 180], "rows": [
            ["Lower ball joint nut", "175 Nm", "129 lb-ft", "front_suspension_4wd"],
            ["Tie-rod end nut", "85 Nm", "63 lb-ft", "steering"],
            ["Front shock absorber upper nuts", "55 Nm", "41 lb-ft", "front_suspension_4wd"],
            ["12V AGM battery capacity", "800 CCA (80 Ah)", "12.6 V resting", "electrical"],
            ["Alternator charging voltage", "13.8 - 14.6 V", "220 A max output", "electrical"],
            ["Front tire cold inflation pressure", "240 kPa (35 psi)", "2.4 bar", "wheels_and_tires"],
            ["Rear tire cold inflation pressure", "275 kPa (40 psi)", "2.75 bar", "wheels_and_tires"],
        ]},
    ], landscape=True)

    doc.save(str(dest))
    doc.close()


def _build_s250_pdf(dest: Path) -> None:
    """Create the 2023 Apex Motors S250 2.0L Hybrid AWD Workshop Manual PDF."""
    doc = pymupdf.open()
    header = "2023 Apex Motors S250 Hybrid Workshop Manual — 2.0L Hybrid AWD"

    # Page 1 (Cover, printed 'i')
    _draw_manual_page(doc, header, "i", [
        {"type": "section", "text": "2023 Apex Motors S250 Hybrid Workshop Manual", "subtitle": "2.0L Atkinson-Cycle Inline-4 Hybrid | e-CVT AWD | Technical Reference Edition"},
        {"type": "para", "text": "SECTION 100-00: General Vehicle Information\nVehicle Identity: 2023 Apex Motors S250 Sedan (AWD Hybrid)\nEngine Code: AX-20H I4\nTransaxle: e-CVT Hybrid Transaxle\nHigh-Voltage System: 288 V Nominal Lithium-Ion Traction Battery"},
        {"type": "heading", "text": "Manual Section Index"},
        {"type": "table", "headers": ["Section Code", "Subsystem", "Printed Page"], "widths": [120, 280, 120], "rows": [
            ["SECTION 206-03", "Front Disc Brake Specifications", "206-03-1"],
            ["SECTION 206-04", "Rear Disc Brake Specifications", "206-04-1"],
            ["SECTION 303-01", "Engine Mechanical — 2.0L Hybrid I4", "303-01-1"],
            ["SECTION 303-03", "Engine Cooling & Inverter Thermal Loop", "303-03-1"],
            ["SECTION 307-01", "Hybrid Transaxle & AWD Rear Drive Unit", "307-01-1"],
            ["SECTION 414-01", "Electrical, 12V & High-Voltage Hybrid", "414-01-1"],
        ]},
    ])

    # Page 2: Front Disc Brake (printed 206-03-1)
    _draw_manual_page(doc, header, "206-03-1", [
        {"type": "section", "text": "SECTION 206-03: Front Disc Brake", "subtitle": "SPECIFICATIONS | Procedure revision date: 08/10/2023"},
        {"type": "table", "headers": ["Description", "Nm", "lb-ft", "lb-in"], "widths": [260, 85, 85, 90], "rows": [
            ["Brake caliper anchor plate bolts", "195", "144", "—"],
            ["Brake caliper guide pin bolts", "32", "24", "—"],
            ["Brake caliper flow bolt", "30", "22", "—"],
            ["Brake disc shield bolts", "14", "—", "124"],
            ["Brake caliper bleeder screw", "10", "—", "89"],
            ["Wheel lug nuts", "140", "103", "—"],
        ]},
        {"type": "table", "headers": ["Component", "Specification", "Part Number / Grade"], "widths": [220, 140, 160], "rows": [
            ["Brake disc minimum thickness", "26.0 mm (1.024 in)", "AX-BRK-S250F"],
            ["Brake pad minimum thickness", "2.5 mm (0.098 in)", "AX-PAD-S250F"],
            ["High Performance DOT 4 LV Brake Fluid", "950 ml (2.0 pt)", "PM-20-LV / WSS-M6C65-A2"],
        ]},
    ])

    # Page 3: Rear Disc Brake (printed 206-04-1)
    _draw_manual_page(doc, header, "206-04-1", [
        {"type": "section", "text": "SECTION 206-04: Rear Disc Brake", "subtitle": "SPECIFICATIONS | Procedure revision date: 08/10/2023"},
        {"type": "table", "headers": ["Description", "Nm", "lb-ft", "lb-in"], "widths": [260, 85, 85, 90], "rows": [
            ["Brake caliper support bracket bolts", "115", "85", "—"],
            ["Brake caliper guide pin bolts", "28", "21", "—"],
            ["Brake caliper flow bolt", "30", "22", "—"],
            ["Electronic parking brake motor bolts", "12", "—", "106"],
        ]},
        {"type": "table", "headers": ["Component", "Specification", "Notes"], "widths": [220, 150, 150], "rows": [
            ["Brake disc minimum thickness", "18.0 mm (0.709 in)", "Solid rear disc"],
            ["Brake pad minimum thickness", "2.5 mm (0.098 in)", "Discard below limit"],
        ]},
    ])

    # Page 4: Engine Mechanical (printed 303-01-1)
    _draw_manual_page(doc, header, "303-01-1", [
        {"type": "section", "text": "SECTION 303-01: Engine Mechanical — 2.0L Hybrid I4", "subtitle": "SPECIFICATIONS & TIGHTENING SEQUENCE"},
        {"type": "table", "headers": ["Description", "Nm", "lb-ft", "Condition"], "widths": [230, 80, 80, 130], "rows": [
            ["Cylinder head bolts (Stage 1)", "35", "26", "Lubricate threads"],
            ["Cylinder head bolts (Stage 2)", "75", "55", "Plus 90 deg turn"],
            ["Connecting rod cap bolts", "38", "28", "Torque-to-yield"],
            ["Main bearing cap bolts", "88", "65", "Inner bolts first"],
            ["Flywheel / damper bolts", "72", "53", "New bolts required"],
            ["Spark plugs", "18", "13", "Iridium 12mm"],
            ["Oil pan drain plug", "25", "18", "Replace aluminum washer"],
        ]},
        {"type": "table", "headers": ["Component", "Specification", "Tolerance / Limit", "Condition"], "widths": [200, 110, 100, 110], "rows": [
            ["Spark plug electrode gap", "0.80 - 0.90 mm", "0.85 +/- 0.05 mm", "Laser iridium"],
            ["Connecting rod bearing clearance", "0.020 - 0.045 mm", "0.060 mm max", "Plastigage check"],
            ["Cylinder compression pressure", "1,150 kPa (167 psi)", "Min 965 kPa", "Maintenance mode"],
        ]},
        {"type": "para", "text": "Cylinder head tightening sequence: Tighten the 10 cylinder head bolts in two passes from the center outward in a criss-cross pattern (#1 through #10).\nStage 1: Tighten to 35 Nm (26 lb-ft). Stage 2: Tighten to 75 Nm (55 lb-ft). Stage 3: Turn an additional 90 deg."},
    ])

    # Page 5: Cooling, Lubrication, Transaxle & Electrical (printed 303-03-1)
    _draw_manual_page(doc, header, "303-03-1", [
        {"type": "section", "text": "SECTION 303-03: Cooling, Lubrication, Hybrid Transaxle & Electrical", "subtitle": "CAPACITIES, PRESSURES & VOLTAGES"},
        {"type": "table", "headers": ["Item / Component", "Specification", "Grade / Alternate", "Subsystem"], "widths": [200, 115, 115, 90], "rows": [
            ["Engine oil capacity (with filter)", "4.3 L (4.5 qt)", "SAE 0W-20 Synthetic", "lubrication"],
            ["Engine coolant capacity", "7.8 L (8.2 qt)", "OAT Pink 50/50", "cooling"],
            ["Inverter cooling loop capacity", "2.1 L (2.2 qt)", "OAT Pink 50/50", "cooling"],
            ["Engine oil pressure (at 2,000 rpm)", "240 - 360 kPa", "35 - 52 psi", "lubrication"],
            ["Radiator cap relief pressure", "108 kPa (15.7 psi)", "AX-CAP-108", "cooling"],
            ["Hybrid e-CVT transaxle fluid capacity", "4.1 L (4.3 qt)", "AX-WS Hybrid ATF", "transmission"],
            ["Rear e-Axle drive unit fluid capacity", "1.1 L (1.16 qt)", "AX-WS Hybrid ATF", "axle_differential"],
            ["12V auxiliary battery capacity", "450 CCA (45 Ah)", "12.6 V resting", "electrical"],
            ["DC-DC converter charging voltage", "13.6 - 14.4 V", "150 A continuous", "electrical"],
            ["High-voltage traction battery nominal", "288 V", "6.5 Ah Li-Ion", "electrical"],
            ["Front tire cold inflation pressure", "230 kPa (33 psi)", "2.3 bar", "wheels_and_tires"],
            ["Rear tire cold inflation pressure", "230 kPa (33 psi)", "2.3 bar", "wheels_and_tires"],
        ]},
    ])

    doc.save(str(dest))
    doc.close()


def ensure_demo_manuals(demo_dir: Path) -> list[tuple[DemoManualSpec, Path]]:
    """Generate deterministic public demonstration manuals if not already present."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[DemoManualSpec, Path]] = []
    for spec in DEMO_MANUALS:
        dest = demo_dir / spec.filename
        if not dest.exists():
            if spec.document_id == "demo_apex_m400_2022":
                _build_m400_pdf(dest)
            elif spec.document_id == "demo_apex_s250_2023":
                _build_s250_pdf(dest)
        results.append((spec, dest))
    return results
