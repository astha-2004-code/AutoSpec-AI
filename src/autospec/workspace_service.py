"""Document-isolated workspace service managing multi-doc indexing, pgvector/hybrid search, jobs, and comparison."""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from autospec.chunker import SpecificationChunk
from autospec.comparison import VehicleComparisonEngine
from autospec.config import (
    DocumentValidationError,
    sanitize_filename,
    sanitize_session_id,
    settings,
    validate_pdf_bytes,
)
from autospec.database import DatabaseManager, DocumentRecord, compute_guest_expiry
from autospec.demo_manuals import ensure_demo_manuals
from autospec.embeddings import CandidateReranker, EmbedderProtocol, get_embedder
from autospec.evidence_validator import EvidenceValidator
from autospec.hybrid_retriever import HybridRetriever, RetrievalMode
from autospec.pdf_parser import PDFParser
from autospec.pipeline import AutoSpecPipeline
from autospec.query_router import QueryRouter
from autospec.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    DocumentInfo,
    IngestionJobStatus,
    IngestResponse,
    OverviewStats,
    QueryRequest,
    QueryResponse,
    SearchHistoryEntry,
    VehicleSpecification,
)
from autospec.spec_extractor import SpecificationExtractor

logger = logging.getLogger(__name__)


class WorkspaceService:
    """Thread-safe service layer isolating document indexes, DB persistence, jobs, history, and conversations."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.db = db or DatabaseManager()
        self._embedder: EmbedderProtocol | None = embedder
        self._embedder_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_ingestions,
            thread_name_prefix="autospec-ingest",
        )
        self._doc_cache_lock = threading.Lock()
        self._loaded_pipelines: dict[str, AutoSpecPipeline] = {}
        self.query_router = QueryRouter()
        self.reranker = CandidateReranker()
        self.extractor = SpecificationExtractor()
        self.validator = EvidenceValidator()
        self.comparison_engine = VehicleComparisonEngine()
        self._demo_initialized = False

    @property
    def embedder(self) -> EmbedderProtocol:
        """Lazily resolve embedding provider so /health and /ready avoid eager model loading."""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = get_embedder()
        return self._embedder

    def _pdf_bytes_for_db(self, pdf_path: Path) -> bytes | None:
        try:
            max_bytes = settings.max_pdf_db_mb * 1024 * 1024
            if pdf_path.exists() and pdf_path.stat().st_size <= max_bytes:
                return pdf_path.read_bytes()
        except Exception:
            pass
        return None

    def _persist_pipeline_content(self, document_id: str, pipe: AutoSpecPipeline) -> int:
        pages_payload = []
        if pipe.current_document:
            for p in pipe.current_document.pages:
                pages_payload.append(
                    {
                        "page_number": p.page_number,
                        "text": p.raw_text,
                        "ocr_used": p.ocr_used,
                    }
                )
        else:
            for pnum, text in pipe.page_text_map.items():
                pages_payload.append({"page_number": pnum, "text": text, "ocr_used": False})

        chunks_payload = []
        for ch in pipe.all_spec_chunks:
            chunks_payload.append(
                {
                    "chunk_id": f"{document_id}:{ch.chunk_id}",
                    "page_number": ch.page_number,
                    "section": ch.section_title,
                    "subsection": ch.section_id,
                    "subsystem": ch.subsystem,
                    "chunk_type": ch.chunk_type,
                    "content": ch.content,
                    "bbox": ch.evidence_bboxes,
                    "metadata": ch.to_dict(),
                }
            )

        specs_list = pipe.get_all_specifications()
        specs_payload = [
            {
                "component": s.component,
                "subsystem": s.subsystem or "General",
                "spec_type": s.spec_type,
                "value": s.value,
                "unit": s.unit,
                "tolerance": s.tolerance,
                "condition": s.conditions,
                "verification_status": str(s.verification_status),
                "verification_reason": s.verification_notes,
                "confidence": float(s.retrieval_score or 0.0),
                "page_number": s.page_number,
                "source_quote": s.source_quote,
                "chunk_id": None,
                "vehicle": {
                    "manufacturer": s.vehicle_make,
                    "model": s.vehicle_model,
                    "year": s.vehicle_year,
                },
                "bbox": [b.model_dump() for b in s.evidence_bboxes],
            }
            for s in specs_list
        ]

        embeddings = pipe.last_embeddings
        if len(embeddings) != len(chunks_payload) and chunks_payload:
            embeddings = self.embedder.embed_texts([ch["content"] for ch in chunks_payload])

        self.db.replace_document_content(
            document_id=document_id,
            pages=pages_payload,
            chunks=chunks_payload,
            embeddings=embeddings,
            specs=specs_payload,
        )
        return len(specs_payload)

    def initialize_demo_manuals(self) -> None:
        """Seed and pre-index redistributable demonstration manuals for instant no-login showcase."""
        if self._demo_initialized:
            return
        if not settings.demo_mode:
            self._demo_initialized = True
            return

        try:
            self.db.ensure_session("public")
            demo_items = ensure_demo_manuals(settings.demo_dir.resolve())
            for spec, pdf_path in demo_items:
                existing = self.db.get_authorized_document(spec.document_id, "public")
                has_db_chunks = bool(existing and self.db.load_document_chunks(spec.document_id))
                if existing and has_db_chunks:
                    continue

                pipe = AutoSpecPipeline(embedder=self.embedder, index_dir=self._doc_index_dir(spec.document_id))
                resp = pipe.ingest_file(
                    pdf_path,
                    force_rebuild=not has_db_chunks,
                    document_id=spec.document_id,
                    manufacturer=spec.manufacturer,
                    model=spec.model,
                    year=spec.year,
                )
                subsystems = sorted({c.subsystem for c in pipe.all_spec_chunks if c.subsystem})
                spec_count = sum(1 for c in pipe.all_spec_chunks if c.chunk_type == "spec_row")
                self.db.upsert_document(
                    DocumentRecord(
                        document_id=spec.document_id,
                        owner_session_id="public",
                        filename=spec.filename,
                        display_name=spec.display_name,
                        fingerprint=resp.fingerprint,
                        pdf_path=str(pdf_path.resolve()),
                        pdf_data=self._pdf_bytes_for_db(pdf_path),
                        total_pages=resp.total_pages,
                        indexed_chunks=resp.chunk_count,
                        spec_count=spec_count,
                        is_demo=True,
                        read_only=True,
                        manufacturer=spec.manufacturer,
                        model=spec.model,
                        year=spec.year,
                        subsystems_json=json.dumps(subsystems),
                        ocr_pages=pipe.current_document.ocr_page_count if pipe.current_document else 0,
                        file_size_bytes=pdf_path.stat().st_size,
                        freshness="Demo Pre-Indexed",
                    )
                )
                self._persist_pipeline_content(spec.document_id, pipe)
                with self._doc_cache_lock:
                    self._loaded_pipelines[spec.document_id] = pipe
            self._demo_initialized = True
        except Exception as err:
            logger.exception("Failed to initialize demonstration manuals: %s", err)

    def shutdown(self) -> None:
        """Gracefully shut down background worker pool and release DB connections."""
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.db.engine.dispose()

    def cleanup_expired_uploads(self) -> int:
        """Delete expired guest documents, PDFs, and indexes."""
        expired = self.db.purge_expired_guest_documents()
        for rec in expired:
            self._remove_document_artifacts(rec.document_id, rec.pdf_path)
        return len(expired)

    def _doc_index_dir(self, document_id: str) -> Path:
        p = settings.get_index_dir() / "collections" / document_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _remove_document_artifacts(self, document_id: str, pdf_path: str | None = None) -> None:
        with self._doc_cache_lock:
            self._loaded_pipelines.pop(document_id, None)
        idx_dir = settings.get_index_dir() / "collections" / document_id
        if idx_dir.exists():
            shutil.rmtree(idx_dir, ignore_errors=True)
        if pdf_path:
            up_root = settings.get_uploads_dir().resolve()
            try:
                p = Path(pdf_path).resolve()
                p.relative_to(up_root)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def _load_pipeline_for_doc(self, doc: DocumentInfo) -> AutoSpecPipeline | None:
        with self._doc_cache_lock:
            if doc.document_id in self._loaded_pipelines:
                return self._loaded_pipelines[doc.document_id]

        idx_dir = self._doc_index_dir(doc.document_id)
        pipe = AutoSpecPipeline(embedder=self.embedder, index_dir=idx_dir)
        if not pipe.load_from_cache(doc.document_id):
            # Rehydrate cleanly from Neon PostgreSQL/SQLite records (survives Render ephemeral FS restarts)
            db_chunks = self.db.load_document_chunks(doc.document_id)
            db_pages = self.db.load_document_pages(doc.document_id)
            if db_chunks:
                hydrated_chunks: list[SpecificationChunk] = []
                for row in db_chunks:
                    try:
                        meta = json.loads(row.metadata_json or "{}")
                        if meta:
                            hydrated_chunks.append(SpecificationChunk(**meta))
                            continue
                    except Exception:
                        pass
                    hydrated_chunks.append(
                        SpecificationChunk(
                            chunk_id=row.chunk_id,
                            document_id=doc.document_id,
                            document_name=doc.filename,
                            page_number=row.page_number,
                            section_id=row.subsection,
                            section_title=row.section,
                            subsystem=row.subsystem,
                            chunk_type=row.chunk_type,
                            content=row.content,
                            raw_evidence=row.content,
                            vehicle_make=doc.manufacturer,
                            vehicle_model=doc.model,
                            vehicle_year=doc.year,
                        )
                    )
                if pipe.load_from_records(doc.document_id, doc.filename, db_pages, hydrated_chunks):
                    with self._doc_cache_lock:
                        self._loaded_pipelines[doc.document_id] = pipe
                    return pipe

            session_scope = "public" if doc.is_demo else doc.document_id
            rec = self.db.get_authorized_document(doc.document_id, session_scope)
            if rec and rec.pdf_data:
                pipe.ingest_bytes(
                    rec.pdf_data,
                    filename=doc.filename,
                    document_id=doc.document_id,
                    manufacturer=doc.manufacturer,
                    model=doc.model,
                    year=doc.year,
                )
            elif rec and rec.pdf_path and Path(rec.pdf_path).exists():
                pipe.ingest_file(
                    rec.pdf_path,
                    document_id=doc.document_id,
                    manufacturer=doc.manufacturer,
                    model=doc.model,
                    year=doc.year,
                )
            else:
                return None

        with self._doc_cache_lock:
            self._loaded_pipelines[doc.document_id] = pipe
        return pipe

    def list_documents(
        self,
        session_id: str,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
        search: str | None = None,
    ) -> list[DocumentInfo]:
        self.cleanup_expired_uploads()
        sid = sanitize_session_id(session_id)
        return self.db.list_authorized_documents(
            sid,
            manufacturer=manufacturer,
            model=model,
            year=year,
            search=search,
        )

    def get_document(self, document_id: str, session_id: str) -> DocumentInfo | None:
        sid = sanitize_session_id(session_id)
        rec = self.db.get_authorized_document(document_id, sid)
        return rec.to_schema() if rec else None

    def get_pdf_file_path(self, document_id: str, session_id: str) -> Path | None:
        sid = sanitize_session_id(session_id)
        rec = self.db.get_authorized_document(document_id, sid)
        if not rec or not rec.pdf_path:
            return None
        p = Path(rec.pdf_path)
        return p if p.exists() else None

    def get_pdf_content(self, document_id: str, session_id: str) -> tuple[bytes | None, Path | None, str]:
        """Return raw PDF bytes (from DB BYTEA) or existing local Path plus safe filename."""
        sid = sanitize_session_id(session_id)
        rec = self.db.get_authorized_document(document_id, sid)
        if not rec:
            return None, None, "manual.pdf"
        if rec.pdf_path:
            p = Path(rec.pdf_path)
            if p.exists():
                return None, p, rec.filename
        if rec.pdf_data:
            return rec.pdf_data, None, rec.filename
        return None, None, rec.filename

    def get_overview_stats(self, session_id: str) -> OverviewStats:
        self.cleanup_expired_uploads()
        return self.db.get_overview_stats(sanitize_session_id(session_id))

    def submit_upload_job(
        self,
        data: bytes,
        raw_filename: str,
        session_id: str,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
        enable_ocr: bool = False,
        force_rebuild: bool = False,
    ) -> IngestionJobStatus:
        """Validate uploaded bytes, deduplicate by fingerprint if eligible, and queue ingestion."""
        sid = sanitize_session_id(session_id)
        self.db.ensure_session(sid)
        safe_name = sanitize_filename(raw_filename)
        validate_pdf_bytes(data, filename=safe_name)

        fingerprint = PDFParser.compute_fingerprint_bytes(data)
        if not force_rebuild:
            existing = self.db.find_document_by_fingerprint(sid, fingerprint)
            if existing:
                dup_job_id = f"job_{uuid.uuid4().hex[:12]}"
                self.db.create_job(
                    job_id=dup_job_id,
                    session_id=sid,
                    filename=safe_name,
                    document_id=existing.document_id,
                    is_duplicate=True,
                )
                return self.db.update_job(
                    dup_job_id,
                    status="completed",
                    stage="completed",
                    progress_pct=100.0,
                    current_page=existing.total_pages,
                    total_pages=existing.total_pages,
                    is_duplicate=True,
                    message=f"Duplicate manual detected; linked existing index for '{existing.display_name}'.",
                ) or self.db.get_job(dup_job_id, sid)  # type: ignore[return-value]

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        dest_dir = settings.get_uploads_dir() / sid
        dest_dir.mkdir(parents=True, exist_ok=True)
        pdf_dest = dest_dir / f"{doc_id}_{safe_name}"
        pdf_dest.write_bytes(data)

        job = self.db.create_job(
            job_id=job_id,
            session_id=sid,
            filename=safe_name,
            document_id=doc_id,
            source_pdf_path=str(pdf_dest.resolve()),
        )
        self.executor.submit(
            self._run_ingestion_job,
            job_id=job_id,
            doc_id=doc_id,
            session_id=sid,
            pdf_path=pdf_dest,
            filename=safe_name,
            manufacturer=manufacturer,
            model=model,
            year=year,
            enable_ocr=enable_ocr,
            force_rebuild=force_rebuild,
        )
        return job

    def retry_job(self, job_id: str, session_id: str) -> tuple[IngestionJobStatus | None, str | None]:
        """Re-submit a failed or cancelled upload job when the source upload still exists."""
        sid = sanitize_session_id(session_id)
        rec = self.db.get_job_record(job_id, sid)
        if not rec:
            return None, "not_found"
        if rec.status not in ("failed", "cancelled"):
            return None, "not_retryable"
        if not rec.source_pdf_path or not Path(rec.source_pdf_path).exists():
            return None, "source_missing"

        doc_id = rec.document_id or f"doc_{uuid.uuid4().hex[:12]}"
        updated = self.db.update_job(
            job_id,
            status="queued",
            stage="uploading",
            progress_pct=0.0,
            current_page=0,
            error=None,
            cancel_requested=False,
            retry_count=rec.retry_count + 1,
            message="Retrying ingestion job...",
        )
        self.executor.submit(
            self._run_ingestion_job,
            job_id=job_id,
            doc_id=doc_id,
            session_id=sid,
            pdf_path=Path(rec.source_pdf_path),
            filename=rec.filename,
            manufacturer=None,
            model=None,
            year=None,
            enable_ocr=False,
            force_rebuild=True,
        )
        return updated, None

    def ingest_upload_sync(
        self,
        data: bytes,
        raw_filename: str,
        session_id: str,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
        enable_ocr: bool = False,
        force_rebuild: bool = False,
    ) -> IngestResponse:
        """Synchronous upload ingestion for direct API callers expecting immediate IngestResponse."""
        sid = sanitize_session_id(session_id)
        self.db.ensure_session(sid)
        safe_name = sanitize_filename(raw_filename)
        validate_pdf_bytes(data, filename=safe_name)

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        dest_dir = settings.get_uploads_dir() / sid
        dest_dir.mkdir(parents=True, exist_ok=True)
        pdf_dest = dest_dir / f"{doc_id}_{safe_name}"
        pdf_dest.write_bytes(data)

        return self._execute_ingestion(
            doc_id=doc_id,
            session_id=sid,
            pdf_path=pdf_dest,
            filename=safe_name,
            manufacturer=manufacturer,
            model=model,
            year=year,
            enable_ocr=enable_ocr,
            force_rebuild=force_rebuild,
        )

    def _run_ingestion_job(
        self,
        job_id: str,
        doc_id: str,
        session_id: str,
        pdf_path: Path,
        filename: str,
        manufacturer: str | None,
        model: str | None,
        year: int | None,
        enable_ocr: bool,
        force_rebuild: bool,
    ) -> None:
        self.db.update_job(job_id, status="running", stage="parsing", message="Extracting PDF layout and tables...")

        def on_progress(cur: int, total: int) -> None:
            pct = (cur / max(1, total)) * 65.0
            self.db.update_job(
                job_id,
                current_page=cur,
                total_pages=total,
                progress_pct=pct,
                stage="parsing",
                message=f"Parsing page {cur} of {total}...",
            )

        def check_cancel() -> bool:
            return self.db.is_job_cancelled(job_id)

        try:
            resp = self._execute_ingestion(
                doc_id=doc_id,
                session_id=session_id,
                pdf_path=pdf_path,
                filename=filename,
                manufacturer=manufacturer,
                model=model,
                year=year,
                enable_ocr=enable_ocr,
                force_rebuild=force_rebuild,
                progress_callback=on_progress,
                cancel_check=check_cancel,
                job_id=job_id,
            )
            self.db.update_job(
                job_id,
                status="completed",
                stage="completed",
                progress_pct=100.0,
                current_page=resp.total_pages,
                total_pages=resp.total_pages,
                message=resp.message,
            )
        except InterruptedError:
            self._remove_document_artifacts(doc_id, None)
            self.db.update_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                message="Ingestion cancelled by user.",
            )
        except Exception as err:
            logger.exception("Background ingestion job %s failed", job_id)
            self._remove_document_artifacts(doc_id, None)
            self.db.update_job(
                job_id,
                status="failed",
                stage="failed",
                error=str(err),
                message=f"Ingestion failed: {type(err).__name__}",
            )

    def _execute_ingestion(
        self,
        doc_id: str,
        session_id: str,
        pdf_path: Path,
        filename: str,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
        enable_ocr: bool = False,
        force_rebuild: bool = False,
        progress_callback=None,
        cancel_check=None,
        job_id: str | None = None,
    ) -> IngestResponse:
        idx_dir = self._doc_index_dir(doc_id)
        pipe = AutoSpecPipeline(embedder=self.embedder, index_dir=idx_dir, enable_ocr=enable_ocr)
        if job_id:
            self.db.update_job(job_id, stage="chunking", progress_pct=70.0, message="Chunking tables & generating embeddings...")
        resp = pipe.ingest_file(
            pdf_path,
            force_rebuild=force_rebuild,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            document_id=doc_id,
            manufacturer=manufacturer,
            model=model,
            year=year,
        )
        doc_info = pipe.get_document_info()
        subsystems = doc_info.subsystems if doc_info else []
        spec_count = doc_info.spec_count if doc_info else 0

        if job_id:
            self.db.update_job(job_id, stage="indexing", progress_pct=86.0, message="Writing vector & BM25 indexes to database...")

        self.db.upsert_document(
            DocumentRecord(
                document_id=doc_id,
                owner_session_id=session_id,
                filename=filename,
                display_name=filename.rsplit(".", 1)[0].replace("_", " "),
                fingerprint=resp.fingerprint,
                pdf_path=str(pdf_path.resolve()),
                pdf_data=self._pdf_bytes_for_db(pdf_path),
                total_pages=resp.total_pages,
                indexed_chunks=resp.chunk_count,
                spec_count=spec_count,
                is_demo=False,
                read_only=False,
                manufacturer=doc_info.manufacturer if doc_info else (manufacturer or "Unknown"),
                model=doc_info.model if doc_info else (model or "Unknown"),
                year=doc_info.year if doc_info else year,
                subsystems_json=json.dumps(subsystems),
                ocr_pages=doc_info.ocr_pages if doc_info else 0,
                file_size_bytes=pdf_path.stat().st_size,
                freshness="Indexed & Verified",
                expires_at=compute_guest_expiry(),
            )
        )
        if job_id:
            self.db.update_job(job_id, stage="validating", progress_pct=95.0, message="Validating extracted specifications and citations...")
        self._persist_pipeline_content(doc_id, pipe)
        with self._doc_cache_lock:
            self._loaded_pipelines[doc_id] = pipe
        return resp

    def reindex_document(
        self, document_id: str, session_id: str, enable_ocr: bool = False
    ) -> tuple[IngestResponse | None, str | None]:
        sid = sanitize_session_id(session_id)
        rec = self.db.get_authorized_document(document_id, sid)
        if not rec:
            return None, "not_found"
        if rec.read_only or rec.is_demo:
            return None, "read_only"

        pdf_bytes, pdf_path, filename = self.get_pdf_content(document_id, sid)
        if pdf_path and pdf_path.exists():
            resp = self._execute_ingestion(
                doc_id=rec.document_id,
                session_id=sid,
                pdf_path=pdf_path,
                filename=filename,
                manufacturer=rec.manufacturer,
                model=rec.model,
                year=rec.year,
                enable_ocr=enable_ocr,
                force_rebuild=True,
            )
            return resp, None
        if pdf_bytes:
            dest_dir = settings.get_uploads_dir() / sid
            dest_dir.mkdir(parents=True, exist_ok=True)
            restored_path = dest_dir / f"{rec.document_id}_{filename}"
            restored_path.write_bytes(pdf_bytes)
            resp = self._execute_ingestion(
                doc_id=rec.document_id,
                session_id=sid,
                pdf_path=restored_path,
                filename=filename,
                manufacturer=rec.manufacturer,
                model=rec.model,
                year=rec.year,
                enable_ocr=enable_ocr,
                force_rebuild=True,
            )
            return resp, None
        return None, "source_missing"

    def delete_document(self, document_id: str, session_id: str) -> str | None:
        sid = sanitize_session_id(session_id)
        rec, err = self.db.delete_document(document_id, sid)
        if err:
            return err
        if rec:
            self._remove_document_artifacts(rec.document_id, rec.pdf_path)
        return None

    def query(self, request: QueryRequest, session_id: str) -> QueryResponse:
        """Execute multi-document hybrid/pgvector search, conversational resolution, and strict evidence grounding."""
        start_t = time.perf_counter()
        sid = sanitize_session_id(session_id)
        self.db.ensure_session(sid)

        authorized_docs = self.list_documents(
            sid,
            manufacturer=request.manufacturer_filter,
            model=request.model_filter,
            year=request.year_filter,
        )
        if request.document_ids:
            req_set = set(request.document_ids)
            authorized_docs = [d for d in authorized_docs if d.document_id in req_set]

        if not authorized_docs:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            return QueryResponse(
                query=request.query,
                results=[],
                total_results=0,
                extraction_method=self.extractor.provider,
                retrieval_mode=request.retrieval_mode,
                processing_time_ms=round(elapsed_ms, 2),
                status="not_found",
                missing_info_explanation="No authorized manuals match the selected document or vehicle filters.",
                message="No authorized documents available for search.",
            )

        conv_ctx: dict[str, Any] = {}
        if request.conversation_id:
            conv_ctx = self.db.get_conversation_context(request.conversation_id, sid)

        parsed = self.query_router.parse_query(request.query)
        effective_query = parsed.normalized_query or request.query
        effective_subsystem = request.subsystem_filter or parsed.target_subsystem

        if parsed.is_follow_up and conv_ctx:
            prev_comp = conv_ctx.get("component")
            prev_sub = conv_ctx.get("subsystem")
            if prev_comp:
                base_comp = re.sub(r"\([^)]*\)", "", prev_comp).strip()
                if base_comp.lower() not in effective_query.lower():
                    effective_query = f"{base_comp} {effective_query}"
            if not effective_subsystem and prev_sub:
                effective_subsystem = prev_sub
            if parsed.is_comparison and len(authorized_docs) == 1:
                authorized_docs = self.list_documents(sid)

        candidate_hits: list[tuple[SpecificationChunk, float]] = []
        page_maps_by_doc: dict[str, dict[int, str]] = {}

        mode: RetrievalMode = (
            request.retrieval_mode
            if request.retrieval_mode in ("dense", "bm25", "hybrid", "hybrid_rerank")
            else "hybrid_rerank"
        )

        # When running against PostgreSQL + pgvector, also query pgvector index directly for dense signal
        pgvector_scores: dict[str, float] = {}
        if self.db.is_postgres and mode in ("dense", "hybrid", "hybrid_rerank"):
            try:
                q_emb = self.embedder.embed_query(effective_query)
                pg_rows = self.db.search_pgvector_chunks(
                    document_ids=[d.document_id for d in authorized_docs],
                    query_embedding=q_emb,
                    top_k=max(request.top_k * 2, 12),
                    subsystem=effective_subsystem,
                )
                for chunk_row, sim_score in pg_rows:
                    pgvector_scores[chunk_row.chunk_id] = sim_score
            except Exception as err:
                logger.warning("pgvector search fallback to in-memory store: %s", type(err).__name__)

        for doc in authorized_docs:
            pipe = self._load_pipeline_for_doc(doc)
            if not pipe:
                continue
            page_maps_by_doc[doc.document_id] = pipe.page_text_map
            doc_hits = pipe.retriever.retrieve(
                effective_query,
                top_k=max(request.top_k, 8),
                subsystem_filter=effective_subsystem,
                retrieval_mode=mode,
                min_score=request.min_score if request.min_score is not None else 0.0,
            )
            if pgvector_scores:
                boosted: list[tuple[SpecificationChunk, float]] = []
                for ch, sc in doc_hits:
                    pg_s = pgvector_scores.get(f"{doc.document_id}:{ch.chunk_id}", 0.0)
                    boosted.append((ch, sc + 0.15 * max(0.0, pg_s)))
                candidate_hits.extend(boosted)
            else:
                candidate_hits.extend(doc_hits)

        if not candidate_hits:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            self.db.add_history(
                sid,
                request.query,
                [d.document_id for d in authorized_docs],
                effective_subsystem,
                0,
                "not_found",
                elapsed_ms,
            )
            return QueryResponse(
                query=request.query,
                normalized_query=effective_query,
                results=[],
                total_results=0,
                extraction_method=self.extractor.provider,
                retrieval_mode=mode,
                processing_time_ms=round(elapsed_ms, 2),
                status="not_found",
                conversation_id=request.conversation_id,
                active_context=conv_ctx,
                missing_info_explanation=(
                    "No indexed specification rows or passages matched your query in the selected manual(s)."
                ),
                message="Specification not found in the authorized manuals.",
            )

        if mode == "hybrid_rerank" and settings.enable_reranker:
            candidate_hits = self.reranker.rerank(effective_query, candidate_hits)
        else:
            candidate_hits.sort(key=lambda x: x[1], reverse=True)

        ranked_hits = HybridRetriever._deduplicate(candidate_hits, top_k=max(request.top_k * 2, 10))
        extracted = self.extractor.extract(effective_query, ranked_hits)
        if not extracted:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            self.db.add_history(
                sid,
                request.query,
                [d.document_id for d in authorized_docs],
                effective_subsystem,
                0,
                "insufficient_evidence",
                elapsed_ms,
            )
            return QueryResponse(
                query=request.query,
                normalized_query=effective_query,
                results=[],
                total_results=0,
                extraction_method=self.extractor.provider,
                retrieval_mode=mode,
                processing_time_ms=round(elapsed_ms, 2),
                status="insufficient_evidence",
                conversation_id=request.conversation_id,
                active_context=conv_ctx,
                missing_info_explanation=(
                    "Passages were retrieved, but none contained an explicit verifiable value matching all query terms."
                ),
                message="Insufficient evidence to answer reliably without hallucination.",
            )

        validated: list[VehicleSpecification] = []
        for s in extracted[: request.top_k]:
            p_map = page_maps_by_doc.get(s.document_id or "")
            validated.append(
                self.validator.validate(
                    s,
                    page_text_map=p_map,
                    query=effective_query,
                    expected_document=s.source_document,
                )
            )

        clarification_needed = False
        clarification_prompt: str | None = None
        if not effective_subsystem and len(validated) > 1:
            subs = {s.subsystem for s in validated if s.subsystem}
            if len(subs) > 1 and any(
                validated[0].component.lower() == s.component.lower() for s in validated[1:]
            ):
                clarification_needed = True
                clarification_prompt = (
                    f"Multiple subsystems match '{validated[0].component}' ({', '.join(sorted(subs))}). "
                    "Select a subsystem filter or specify front/rear in your follow-up."
                )

        new_ctx = {
            "component": validated[0].component if validated else conv_ctx.get("component"),
            "subsystem": validated[0].subsystem if validated else effective_subsystem,
            "document_ids": [d.document_id for d in authorized_docs],
            "last_query": request.query,
        }
        if request.conversation_id:
            self.db.save_conversation_context(request.conversation_id, sid, new_ctx)

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        top_status = validated[0].verification_status if validated else "not_found"
        self.db.add_history(
            sid,
            request.query,
            [d.document_id for d in authorized_docs],
            effective_subsystem,
            len(validated),
            top_status,
            elapsed_ms,
        )

        active_method = validated[0].extraction_method if validated else self.extractor.provider
        return QueryResponse(
            query=request.query,
            normalized_query=effective_query,
            results=validated,
            total_results=len(validated),
            extraction_method=active_method,
            retrieval_mode=mode,
            processing_time_ms=round(elapsed_ms, 2),
            status="clarification_needed" if clarification_needed else "answered",
            clarification_needed=clarification_needed,
            clarification_prompt=clarification_prompt,
            conversation_id=request.conversation_id,
            active_context=new_ctx,
            message=f"Returned {len(validated)} verified specification record(s).",
        )

    def list_specifications(
        self,
        session_id: str,
        document_ids: list[str] | None = None,
        subsystem: str | None = None,
        spec_type: str | None = None,
        verification_status: str | None = None,
        search: str | None = None,
    ) -> list[VehicleSpecification]:
        docs = self.list_documents(session_id)
        if document_ids:
            req_set = set(document_ids)
            docs = [d for d in docs if d.document_id in req_set]

        all_specs: list[VehicleSpecification] = []
        for doc in docs:
            pipe = self._load_pipeline_for_doc(doc)
            if pipe:
                all_specs.extend(pipe.get_all_specifications())

        filtered = all_specs
        if subsystem:
            filtered = [s for s in filtered if (s.subsystem or "").lower() == subsystem.lower()]
        if spec_type:
            filtered = [s for s in filtered if s.spec_type.lower() == spec_type.lower()]
        if verification_status:
            filtered = [s for s in filtered if s.verification_status.lower() == verification_status.lower()]
        if search:
            s_low = search.lower()
            filtered = [
                s
                for s in filtered
                if s_low in s.component.lower()
                or s_low in s.value.lower()
                or s_low in (s.part_number or "").lower()
                or s_low in s.section.lower()
            ]
        return filtered

    def compare_vehicles(self, request: ComparisonRequest, session_id: str) -> ComparisonResponse:
        docs = self.list_documents(session_id)
        doc_map = {d.document_id: d for d in docs}
        selected_docs = [doc_map[did] for did in request.document_ids if did in doc_map]
        if len(selected_docs) < 2:
            raise DocumentValidationError("Please select at least 2 authorized vehicles to compare.")

        specs_by_doc: dict[str, list[VehicleSpecification]] = {}
        for d in selected_docs:
            pipe = self._load_pipeline_for_doc(d)
            specs_by_doc[d.document_id] = pipe.get_all_specifications() if pipe else []

        return self.comparison_engine.compare(selected_docs, specs_by_doc, request)

    def get_history(self, session_id: str, limit: int = 30) -> list[SearchHistoryEntry]:
        return self.db.list_history(sanitize_session_id(session_id), limit=limit)

    def clear_history(self, session_id: str) -> int:
        return self.db.clear_history(sanitize_session_id(session_id))

    def delete_history_entry(self, entry_id: int, session_id: str) -> bool:
        return self.db.delete_history_entry(entry_id, sanitize_session_id(session_id))

    def list_evaluation_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.list_evaluation_runs(limit=limit)
