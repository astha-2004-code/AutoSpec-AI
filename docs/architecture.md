# AutoSpec AI v2.0 — System Architecture

## 1. High-Level System Architecture

```mermaid
graph TD
    User([Engineer / Recruiter]) -->|SPA UI + PDF.js Viewer| Web[React + TypeScript + Vite]
    Web -->|REST + X-Session-ID| API[FastAPI Service /api/v1]
    API -->|Bounded Worker Pool| Service[WorkspaceService]
    Service --> DB[(SQLAlchemy: SQLite / PostgreSQL)]
    Service --> Collections[Isolated Per-Document Index Collections]

    subgraph Ingestion [Document Intelligence & Security]
        Upload[PDF Upload / Local Path] --> SecVal[Magic Bytes, Encryption & Page Limits]
        SecVal --> Parser[PyMuPDF Tables, Line BBoxes & Optional OCR]
        Parser --> Chunker[TableAwareChunker + Multi-Page Continuation]
    end

    subgraph Storage [Per-Document Persistent Stores]
        Chunker --> FAISS[FAISS IndexFlatIP]
        Chunker --> BM25[BM25Plus JSON Corpus]
        Chunker --> Pages[Persisted Page Text & Label Map]
    end

    subgraph Retrieval [Hybrid RAG & Strict Verification]
        Query[Natural Language / Follow-Up] --> Router[QueryRouter + Abbreviation Expansion]
        Router --> Hybrid[HybridRetriever + RRF + Reranker]
        FAISS --> Hybrid
        BM25 --> Hybrid
        Hybrid --> Extractor[Deterministic Fallback / Sanitized LLM]
        Extractor --> Validator[EvidenceValidator: Value, Unit, Quote-on-Page]
        Pages --> Validator
    end
```

## 2. Core Design Decisions & Trade-Offs

- **Isolated Per-Document Collections (`WorkspaceService`)**: Replaces the legacy shared singleton pipeline with per-document FAISS/BM25/page stores and a SQLAlchemy metadata repository (`documents`, `ingestion_jobs`, `search_history`, `conversation_contexts`), preventing cross-user data leakage and concurrent write corruption.
- **Safe JSON BM25 Persistence**: Eliminates `pickle` deserialization risks by storing tokenized corpora as JSON and reconstructing `BM25Plus` in memory on load.
- **Strict Grounding & Unit Normalization (`units.py`, `evidence_validator.py`)**: Prevents partial-digit matches (`"3"` matching `"37"`), verifies physical unit compatibility, persists page text alongside indexes so cached loads retain full page verification, and exposes six explicit statuses (`verified`, `partially_supported`, `conflicting_evidence`, `ambiguous`, `insufficient_evidence`, `not_found`).
- **Deterministic Offline + CPU Reranking**: Ensures zero-API-key operation and fast local inference, with optional `CrossEncoder` weights and LLM providers configurable via environment variables.
- **PDF.js Evidence Grounding**: Persists real PyMuPDF line/row bounding boxes and both physical page numbers and printed manual page labels so citations open directly to the source page with coordinate overlays.
