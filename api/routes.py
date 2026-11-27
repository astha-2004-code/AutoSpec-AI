"""FastAPI REST API for AutoSpec AI with Neon/pgvector persistence, isolation, security, and concurrency."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from autospec.config import (
    DocumentValidationError,
    configure_logging,
    get_process_memory_mb,
    redact_sensitive,
    sanitize_session_id,
    settings,
)
from autospec.embeddings import resolve_embedding_provider
from autospec.export import ExportManager
from autospec.pdf_parser import is_ocr_available
from autospec.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    DocumentInfo,
    DocumentRenameRequest,
    HealthResponse,
    IngestionJobStatus,
    IngestResponse,
    OverviewStats,
    QueryRequest,
    QueryResponse,
    SearchHistoryEntry,
    VehicleSpecification,
)
from autospec.workspace_service import WorkspaceService

configure_logging()
logger = logging.getLogger("autospec.api")

_service_instance: WorkspaceService | None = None
_service_lock = threading.Lock()


def get_service(init_demo: bool = True) -> WorkspaceService:
    """Thread-safe service singleton factory with non-blocking demo bootstrap."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                svc = WorkspaceService()
                _service_instance = svc
    if init_demo and _service_instance._demo_state == "pending":
        if settings.defer_demo_init:
            _service_instance.start_background_demo_init()
        else:
            _service_instance.initialize_demo_manuals()
    return _service_instance


class _SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter protecting public endpoints."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_per_minute:
                return False
            q.append(now)
            return True


_rate_limiter = _SlidingWindowRateLimiter(settings.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open HTTP port immediately while initializing demo manuals cleanly in background."""
    svc = get_service(init_demo=False)
    if settings.defer_demo_init:
        svc.start_background_demo_init()
    else:
        await anyio.to_thread.run_sync(svc.initialize_demo_manuals)
    logger.info(
        "AutoSpec AI API started (env=%s, llm=%s, emb_provider=%s, postgres=%s, rss_mb=%.1f)",
        settings.env,
        settings.llm_provider,
        resolve_embedding_provider(),
        svc.db.is_postgres,
        get_process_memory_mb(),
    )
    yield
    svc.shutdown()
    logger.info("AutoSpec AI API shut down cleanly.")


app = FastAPI(
    title="AutoSpec AI REST API",
    version="2.0.0",
    description="Evidence-Grounded Automotive Specification Intelligence Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-ID", "X-Request-ID"],
)


@app.middleware("http")
async def security_and_request_id_middleware(request: Request, call_next):
    """Attach X-Request-ID, enforce rate limiting, and set defensive HTTP security headers."""
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    client_ip = request.client.host if request.client else "unknown"
    session_key = request.headers.get("X-Session-ID") or client_ip

    if not request.url.path.endswith(("/health", "/ready")) and not _rate_limiter.check(session_key):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please wait before retrying.", "request_id": req_id},
            headers={"X-Request-ID": req_id},
        )

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.exception_handler(DocumentValidationError)
async def document_validation_exception_handler(request: Request, exc: DocumentValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "DocumentValidationError"},
    )


def _session_dep(x_session_id: str | None = Header(default=None)) -> str:
    return sanitize_session_id(x_session_id)


v1 = APIRouter(prefix="/api/v1")


# ------------------------------------------------------------------------------
# System Health, Readiness & Overview
# ------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
@v1.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health(svc: WorkspaceService = Depends(get_service)) -> HealthResponse:
    """Fast non-blocking liveness check reporting DB, pgvector, demo state, and RSS memory."""
    storage_ready = settings.get_storage_path().exists()
    db_ok, db_backend, pgvec = await anyio.to_thread.run_sync(svc.db.ping)
    docs = await anyio.to_thread.run_sync(svc.list_documents, "public")
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version="2.0.0",
        environment=settings.env,
        storage_ready=storage_ready,
        database_ready=db_ok,
        database_backend=db_backend,
        pgvector_enabled=pgvec,
        demo_ready=svc._demo_initialized or any(d.is_demo for d in docs),
        demo_state=svc._demo_state,
        active_llm_provider=settings.llm_provider,
        embedding_provider=resolve_embedding_provider(),
        embedding_model=settings.embedding_model,
        reranker_enabled=settings.enable_reranker,
        ocr_available=is_ocr_available(),
        document_count=len(docs),
        memory_rss_mb=get_process_memory_mb(),
    )


@app.get("/ready", tags=["System"])
@v1.get("/ready", tags=["System"])
async def get_ready(svc: WorkspaceService = Depends(get_service)) -> dict[str, Any]:
    """Readiness probe verifying storage, database connectivity, and demo status."""
    db_ok, db_backend, pgvec = await anyio.to_thread.run_sync(svc.db.ping)
    docs = await anyio.to_thread.run_sync(svc.list_documents, "public")
    return {
        "ready": settings.get_storage_path().exists()
        and db_ok
        and (not settings.demo_mode or len(docs) > 0 or svc._demo_state == "initializing"),
        "demo_state": svc._demo_state,
        "database_backend": db_backend,
        "pgvector_enabled": pgvec,
        "demo_documents": sum(1 for d in docs if d.is_demo),
        "memory_rss_mb": get_process_memory_mb(),
    }


@v1.get("/overview", response_model=OverviewStats, tags=["System"])
async def get_workspace_overview(
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> OverviewStats:
    """Return real database-computed statistics for the current workspace session."""
    await anyio.to_thread.run_sync(svc.ensure_demo_ready)
    return await anyio.to_thread.run_sync(svc.get_overview_stats, session_id)


@v1.get("/demo", tags=["System"])
async def get_demo_workspace(svc: WorkspaceService = Depends(get_service)) -> dict[str, Any]:
    """Return redistributable demonstration manuals and curated example verification queries."""
    await anyio.to_thread.run_sync(svc.ensure_demo_ready)
    docs = await anyio.to_thread.run_sync(svc.list_documents, "public")
    demo_docs = [d.model_dump() for d in docs if d.is_demo]
    return {
        "demo_ready": len(demo_docs) > 0,
        "demo_state": svc._demo_state,
        "documents": demo_docs,
        "sample_queries": [
            "What is the cylinder head bolt tightening torque?",
            "What is the spark plug tightening torque?",
            "Compare wheel lug nut torque between Apex M400 and S250",
            "What engine oil viscosity and fill capacity does Apex M400 require?",
            "What is the brake caliper mounting bolt torque?",
        ],
    }


# ------------------------------------------------------------------------------
# Document Management & Background Ingestion Jobs
# ------------------------------------------------------------------------------


@v1.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents(
    manufacturer: str | None = None,
    model: str | None = None,
    year: int | None = None,
    search: str | None = None,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> list[DocumentInfo]:
    """List authorized service manuals (public read-only demos + session-scoped uploads)."""
    await anyio.to_thread.run_sync(svc.ensure_demo_ready)
    return await anyio.to_thread.run_sync(
        lambda: svc.list_documents(
            session_id,
            manufacturer=manufacturer,
            model=model,
            year=year,
            search=search,
        )
    )


@v1.get("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def get_document_by_id(
    document_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> DocumentInfo:
    doc = await anyio.to_thread.run_sync(svc.get_document, document_id, session_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@v1.get("/documents/{document_id}/pdf", tags=["Documents"])
async def stream_document_pdf(
    document_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
):
    """Serve original PDF bytes from disk cache or Neon PostgreSQL BYTEA storage."""
    pdf_bytes, pdf_path, filename = await anyio.to_thread.run_sync(
        svc.get_pdf_content, document_id, session_id
    )
    if pdf_path and pdf_path.exists():
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=filename,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    if pdf_bytes:
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF binary not found.")


@v1.post("/documents/upload", response_model=IngestionJobStatus, tags=["Documents"])
async def upload_document_async(
    file: UploadFile = File(...),
    manufacturer: str | None = Form(default=None),
    model: str | None = Form(default=None),
    year: int | None = Form(default=None),
    enable_ocr: bool = Form(default=False),
    force_rebuild: bool = Form(default=False),
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> IngestionJobStatus:
    """Upload a PDF manual and start bounded background ingestion with stage progress tracking."""
    content = await file.read()
    filename = file.filename or "uploaded_manual.pdf"
    return await anyio.to_thread.run_sync(
        lambda: svc.submit_upload_job(
            data=content,
            raw_filename=filename,
            session_id=session_id,
            manufacturer=manufacturer,
            model=model,
            year=year,
            enable_ocr=enable_ocr,
            force_rebuild=force_rebuild,
        )
    )


@v1.get("/jobs/{job_id}", response_model=IngestionJobStatus, tags=["Documents"])
async def get_ingestion_job_status(
    job_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> IngestionJobStatus:
    job = await anyio.to_thread.run_sync(svc.db.get_job, job_id, session_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
    return job


@v1.post("/jobs/{job_id}/cancel", response_model=IngestionJobStatus, tags=["Documents"])
async def cancel_ingestion_job(
    job_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> IngestionJobStatus:
    job = await anyio.to_thread.run_sync(svc.db.request_job_cancel, job_id, session_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
    return job


@v1.post("/jobs/{job_id}/retry", response_model=IngestionJobStatus, tags=["Documents"])
async def retry_ingestion_job(
    job_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> IngestionJobStatus:
    job, err = await anyio.to_thread.run_sync(svc.retry_job, job_id, session_id)
    if err == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
    if err == "not_retryable":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only failed or cancelled jobs can be retried.",
        )
    if err == "source_missing":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Original uploaded PDF file is no longer available to retry.",
        )
    assert job is not None
    return job


@v1.patch("/documents/{document_id}", response_model=DocumentInfo, tags=["Documents"])
async def update_document(
    document_id: str,
    payload: DocumentRenameRequest,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> DocumentInfo:
    updated, err = await anyio.to_thread.run_sync(
        lambda: svc.db.update_document_metadata(
            document_id,
            session_id,
            display_name=payload.display_name,
            manufacturer=payload.manufacturer,
            model=payload.model,
            year=payload.year,
        )
    )
    if err == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if err == "read_only":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public demonstration manuals are read-only.",
        )
    assert updated is not None
    return updated


@v1.post("/documents/{document_id}/reindex", response_model=IngestResponse, tags=["Documents"])
async def reindex_document(
    document_id: str,
    enable_ocr: bool = Query(default=False),
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> IngestResponse:
    resp, err = await anyio.to_thread.run_sync(
        lambda: svc.reindex_document(document_id, session_id, enable_ocr=enable_ocr)
    )
    if err == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if err == "read_only":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Public demonstration manuals are read-only.")
    if err == "source_missing":
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Source PDF binary is no longer available.")
    assert resp is not None
    return resp


@v1.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> dict[str, str]:
    err = await anyio.to_thread.run_sync(svc.delete_document, document_id, session_id)
    if err == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if err == "read_only":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public demonstration manuals are read-only.",
        )
    return {"status": "deleted", "document_id": document_id}


@app.get("/document", response_model=DocumentInfo, tags=["Document Management"])
async def get_primary_document(
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> DocumentInfo:
    docs = await anyio.to_thread.run_sync(svc.list_documents, session_id)
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No document has been indexed yet.",
        )
    return docs[0]


@app.post("/ingest", response_model=IngestResponse, tags=["Document Management"])
@v1.post("/documents/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document_sync(
    file: UploadFile | None = File(default=None),
    manufacturer: str | None = Form(default=None),
    model: str | None = Form(default=None),
    year: int | None = Form(default=None),
    enable_ocr: bool = Form(default=False),
    force_rebuild: bool = Form(default=False),
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> IngestResponse:
    """Ingest an uploaded PDF manual synchronously."""
    if file is None:
        raise DocumentValidationError(
            "A multipart PDF file upload is required. Use scripts/ingest_manual.py for local filesystem ingestion."
        )
    content = await file.read()
    filename = file.filename or "uploaded_manual.pdf"
    return await anyio.to_thread.run_sync(
        lambda: svc.ingest_upload_sync(
            data=content,
            raw_filename=filename,
            session_id=session_id,
            manufacturer=manufacturer,
            model=model,
            year=year,
            enable_ocr=enable_ocr,
            force_rebuild=force_rebuild,
        )
    )


# ------------------------------------------------------------------------------
# Query, Specification Explorer & Conversational Search
# ------------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse, tags=["Specification Retrieval"])
@v1.post("/query", response_model=QueryResponse, tags=["Specification Retrieval"])
async def query_specifications(
    request: QueryRequest,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> QueryResponse:
    """Execute evidence-grounded hybrid/pgvector retrieval, extraction, and strict verification."""
    try:
        with anyio.fail_after(settings.request_timeout_seconds):
            return await anyio.to_thread.run_sync(svc.query, request, session_id)
    except TimeoutError as err:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Query processing timed out.",
        ) from err
    except DocumentValidationError:
        raise
    except Exception as err:
        logger.error("Query processing failed: %s", redact_sensitive(str(err)))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {type(err).__name__}",
        ) from err


@app.get("/specifications", response_model=list[VehicleSpecification], tags=["Specification Retrieval"])
@v1.get("/specifications", response_model=list[VehicleSpecification], tags=["Specification Retrieval"])
async def list_specifications(
    document_ids: str | None = Query(default=None, description="Comma-separated document IDs"),
    subsystem: str | None = Query(default=None),
    spec_type: str | None = Query(default=None),
    verification_status: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> list[VehicleSpecification]:
    """List verified specifications across authorized manuals with filtering and pagination."""
    doc_list = [d.strip() for d in document_ids.split(",") if d.strip()] if document_ids else None
    specs = await anyio.to_thread.run_sync(
        lambda: svc.list_specifications(
            session_id=session_id,
            document_ids=doc_list,
            subsystem=subsystem,
            spec_type=spec_type,
            verification_status=verification_status,
            search=search,
        )
    )
    return specs[offset : offset + limit]


@v1.post("/conversations/{conversation_id}/reset", tags=["Specification Retrieval"])
async def reset_conversation(
    conversation_id: str,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> dict[str, str]:
    await anyio.to_thread.run_sync(svc.db.clear_conversation_context, conversation_id, session_id)
    return {"status": "reset", "conversation_id": conversation_id}


# ------------------------------------------------------------------------------
# Vehicle Comparison Engine
# ------------------------------------------------------------------------------


@v1.post("/compare", response_model=ComparisonResponse, tags=["Vehicle Comparison"])
async def compare_vehicles(
    request: ComparisonRequest,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> ComparisonResponse:
    """Compare technical specifications across 2-6 selected vehicle manuals."""
    return await anyio.to_thread.run_sync(svc.compare_vehicles, request, session_id)


@v1.post("/compare/export", tags=["Vehicle Comparison"])
async def export_comparison(
    request: ComparisonRequest,
    format: str = Query(default="csv", pattern="^(csv|json)$"),
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
):
    report = await anyio.to_thread.run_sync(svc.compare_vehicles, request, session_id)
    if format == "json":
        return Response(
            content=ExportManager.comparison_to_json(report),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=autospec_comparison.json"},
        )
    return Response(
        content=ExportManager.comparison_to_csv(report),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=autospec_comparison.csv"},
    )


# ------------------------------------------------------------------------------
# Search History & Exports
# ------------------------------------------------------------------------------


@v1.get("/history", response_model=list[SearchHistoryEntry], tags=["History"])
async def get_search_history(
    limit: int = Query(default=30, ge=1, le=100),
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> list[SearchHistoryEntry]:
    return await anyio.to_thread.run_sync(svc.get_history, session_id, limit)


@v1.delete("/history", tags=["History"])
async def clear_search_history(
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> dict[str, int]:
    deleted = await anyio.to_thread.run_sync(svc.clear_history, session_id)
    return {"deleted": deleted}


@v1.delete("/history/{entry_id}", tags=["History"])
async def delete_history_item(
    entry_id: int,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
) -> dict[str, Any]:
    removed = await anyio.to_thread.run_sync(svc.delete_history_entry, entry_id, session_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History entry not found.")
    return {"deleted": entry_id}


@app.get("/export/json", tags=["Exports"])
@v1.get("/export/json", tags=["Exports"])
async def export_specifications_json(
    document_ids: str | None = None,
    subsystem: str | None = None,
    spec_type: str | None = None,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
):
    doc_list = [d.strip() for d in document_ids.split(",") if d.strip()] if document_ids else None
    specs = await anyio.to_thread.run_sync(
        lambda: svc.list_specifications(
            session_id=session_id,
            document_ids=doc_list,
            subsystem=subsystem,
            spec_type=spec_type,
        )
    )
    return Response(
        content=ExportManager.to_json(specs),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=autospec_specifications.json"},
    )


@app.get("/export/csv", tags=["Exports"])
@v1.get("/export/csv", tags=["Exports"])
async def export_specifications_csv(
    document_ids: str | None = None,
    subsystem: str | None = None,
    spec_type: str | None = None,
    session_id: str = Depends(_session_dep),
    svc: WorkspaceService = Depends(get_service),
):
    doc_list = [d.strip() for d in document_ids.split(",") if d.strip()] if document_ids else None
    specs = await anyio.to_thread.run_sync(
        lambda: svc.list_specifications(
            session_id=session_id,
            document_ids=doc_list,
            subsystem=subsystem,
            spec_type=spec_type,
        )
    )
    return Response(
        content=ExportManager.to_csv(specs),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=autospec_specifications.csv"},
    )


# ------------------------------------------------------------------------------
# Evaluation & Analytics (Real Measured Benchmark Outputs Only)
# ------------------------------------------------------------------------------


@v1.get("/analytics/benchmark", tags=["Analytics"])
async def get_benchmark_analytics() -> dict[str, Any]:
    """Return latest real measured benchmark report and ablation study results."""
    report_path = settings.get_benchmark_dir() / "latest_benchmark.json"
    if not report_path.exists():
        fallback = Path("data/benchmarks/latest_benchmark.json")
        if fallback.exists():
            report_path = fallback
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No measured benchmark report has been generated yet. Run scripts/evaluate.py.",
            )
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@v1.get("/evaluation/runs", tags=["Analytics"])
async def list_persisted_evaluation_runs(
    svc: WorkspaceService = Depends(get_service),
) -> list[dict[str, Any]]:
    """Return benchmark evaluation runs persisted in Neon PostgreSQL / SQLite."""
    return await anyio.to_thread.run_sync(svc.list_evaluation_runs, 20)


app.include_router(v1)
