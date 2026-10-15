# AutoSpec AI — Assignment Scope and Requirements

## 1. Executive Summary

Automotive workshop manuals (such as the 2014 Ford F-150 Workshop Manual) contain critical maintenance specifications, including fastener torque values, fluid capacities, clearances, and lubricant part numbers. When performing repairs, precision is vital: retrieving an incorrect torque value or confusing front and rear brake fasteners can lead to catastrophic mechanical failure.

**AutoSpec AI** is an evidence-grounded automotive specification intelligence system designed to extract verified technical values with strict source traceability, section awareness, and dual-unit preservation.

---

## 2. Source Material Analysis: 2014 Ford F-150 Workshop Manual

Analysis of the 852-page Ford F-150 workshop manual reveals key structural patterns:

1. **Hierarchical Section Organization**:
   - Each system is partitioned into explicit sections:
     - `SECTION 204-01A`: Front Suspension — Rear Wheel Drive (RWD) (Page 24)
     - `SECTION 204-01B`: Front Suspension — Four Wheel Drive (4WD) (Page 59)
     - `SECTION 204-02`: Rear Suspension (Page 128)
     - `SECTION 205-02A`: Rear Drive Axle/Differential — Ford 8.8-Inch Ring Gear (Page 258)
     - `SECTION 205-02B`: Rear Drive Axle/Differential — Ford 9.75-Inch Ring Gear (Page 340)
     - `SECTION 206-03`: Front Disc Brake (Page 636)
     - `SECTION 206-04`: Rear Disc Brake (Page 652)
     - `SECTION 206-05`: Parking Brake and Actuation (Page 672)

2. **The Component Disambiguation Challenge**:
   - A single component name can appear across multiple vehicle subsystems with radically different specifications:
     - **Front Disc Brake Caliper Guide Pin Bolts** (`SECTION 206-03`, Page 636): `37 Nm (27 lb-ft)`
     - **Rear Disc Brake Caliper Guide Pin Bolts** (`SECTION 206-04`, Page 652): `33 Nm (24 lb-ft)`
     - **Front Brake Caliper Anchor Plate Bolts** (`SECTION 206-03`, Page 636): `250 Nm (184 lb-ft)`
     - **Rear Brake Caliper Support Bracket Bolts** (`SECTION 206-04`, Page 652): `150 Nm (111 lb-ft)`
   - Generic RAG pipelines frequently merge these sections or return the wrong subsystem value. AutoSpec AI solves this by keeping section context bound to every chunk and row.

3. **Multi-Table Specification Pages**:
   - Technical pages typically pack multiple distinct specification tables onto a single page:
     - **Material & Fluid Table**: Item, Ford Specification code (e.g., `WSS-M6C62-A`), Part number (e.g., `PM-1-C`), Fill capacity (e.g., `1,420 ml (3 pt)`).
     - **Dimension & Clearance Table**: Rotor minimum thickness (`32 mm / 1.259 in` front vs `23.0 mm / 0.906 in` rear), pad wear limits (`3.0 mm / 0.118 in`).
     - **Torque Specifications Table**: Component description, Primary torque (`Nm`), Alternate torque (`lb-ft` or `lb-in`).

4. **Document Artifacts**:
   - Headers: `General Specifications`, `Torque Specifications`, `2014 F-150 Workshop Manual`.
   - Footers: `Page 1 sur 2`, `2014-03-01`, `file:///C:/TSO/tsocache/...`.
   - Ingestion must strip running footers and URLs without removing decimal numbers, part numbers, or units.

---

## 3. Core Functional Capabilities

| Capability | Requirement | Implementation Strategy |
|---|---|---|
| **1. Document Intelligence** | Clean extraction, section metadata, page preservation | PyMuPDF page-by-page extraction, regex header/footer cleaner, section detector, SHA-256 fingerprinting |
| **2. Table-Aware Chunking** | Preserve row-component-unit association | Structured semantic chunking linking table title, column headers, component names, values, units, and page/section metadata |
| **3. Hybrid Search** | Dense semantic + sparse lexical + query routing | FAISS `IndexFlatIP` (SentenceTransformers `all-MiniLM-L6-v2`) + BM25Okapi + Reciprocal Rank Fusion (RRF) |
| **4. Grounded Extraction** | Structured JSON/CSV records, zero hallucination | Pydantic v2 schemas, strict LLM prompts, and a robust regex deterministic fallback for 100% offline local execution |
| **5. Evidence Verification** | Verifiable citations and quote traceability | Source page validation, exact quote string matching in extracted page text, numerical value/unit integrity check |
| **6. User Interface** | Interactive engineer dashboard | Streamlit multi-page UI (Dashboard, Ask AutoSpec, Specification Explorer, Document Management, Quality & Evaluation) |
| **7. REST API** | Production API service | FastAPI with Pydantic request/response validation, path traversal security, and CSV/JSON export |

---

## 4. Strict Constraints & Compliance

- **No PDF Bundling**: PDFs are development references only. Never commit, bundle, or distribute PDFs in the repository or Docker image. Runtime upload or local path selection only.
- **Offline Fallback**: Must execute with full functionality without requiring an external paid LLM API key.
- **Reproducible Evaluation**: Measured benchmarks against verified reference questions from the Ford F-150 manual.
