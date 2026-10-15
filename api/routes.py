"""FastAPI interface for AutoSpec AI specification intelligence service."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from autospec.config import DocumentValidationError, settings
from autospec.export import ExportManager
from autospec.pipeline import AutoSpecPipeline
from autospec.schemas import (
    DocumentInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    VehicleSpecification,
)

logger = logging.getLogger("autospec.api")

app = FastAPI(
    title="AutoSpec AI REST API",
    version="0.1.0",
    description="Evidence-Grounded Automotive Specification Intelligence Service",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend and development tooling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline dependency singleton
_pipeline_instance: AutoSpecPipeline | None = None


def get_pipeline() -> AutoSpecPipeline:
    """Return shared application pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AutoSpecPipeline()
        # If a default document was indexed previously, try auto-loading it
        index_dir = settings.get_index_dir()
        manifests = list(index_dir.glob("*_manifest.json"))
        if manifests:
            try:
                # Load the most recently created manifest
                manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                fp = manifests[0].name.replace("_manifest.json", "")
                if _pipeline_instance.vector_store.is_fresh(fp):
                    _pipeline_instance.vector_store.load_index(fp)
                    _pipeline_instance.all_spec_chunks = _pipeline_instance.vector_store.chunks
                    bm25_p = index_dir / f"{fp}.bm25"
                    if bm25_p.exists():
                        _pipeline_instance.keyword_retriever.load(bm25_p, _pipeline_instance.all_spec_chunks)
            except Exception as e:
                logger.warning("Could not auto-load cached index: %s", e)
    return _pipeline_instance


@app.exception_handler(DocumentValidationError)
async def document_validation_exception_handler(request, exc: DocumentValidationError):
    """Handle document validation errors with 400 Bad Request."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "DocumentValidationError"},
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def get_health(pipeline: AutoSpecPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Check service health and active configuration."""
    storage_ready = settings.get_storage_path().exists()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        storage_ready=storage_ready,
        active_llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )


@app.get("/document", response_model=DocumentInfo, tags=["Document Management"])
async def get_document(pipeline: AutoSpecPipeline = Depends(get_pipeline)) -> DocumentInfo:
    """Retrieve metadata about the currently indexed service manual."""
    doc_info = pipeline.get_document_info()
    if not doc_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No document has been indexed yet. Please ingest a manual via /ingest.",
        )
    return doc_info


@app.post("/ingest", response_model=IngestResponse, tags=["Document Management"])
async def ingest_document(
    request: IngestRequest | None = None,
    file: UploadFile | None = File(default=None),
    pipeline: AutoSpecPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Ingest a PDF service manual from a validated local path or file upload."""
    try:
        if file is not None:
            # Handle uploaded multipart file
            content = await file.read()
            filename = file.filename or "uploaded_manual.pdf"
            return pipeline.ingest_bytes(
                data=content,
                filename=filename,
                force_rebuild=request.force_rebuild if request else False,
            )

        if request and request.pdf_path:
            return pipeline.ingest_file(
                pdf_path_str=request.pdf_path,
                force_rebuild=request.force_rebuild,
            )

        raise DocumentValidationError("Either a file upload or a 'pdf_path' must be provided.")

    except DocumentValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {type(e).__name__}",
        ) from e


@app.post("/query", response_model=QueryResponse, tags=["Specification Retrieval"])
async def query_specifications(
    request: QueryRequest,
    pipeline: AutoSpecPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Query vehicle specifications using hybrid retrieval, extraction, and verification."""
    if not pipeline.all_spec_chunks:
        # Check if index can be loaded
        doc_info = pipeline.get_document_info()
        if not doc_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No manual is indexed yet. Please ingest a document first.",
            )

    try:
        return pipeline.query(
            question=request.query,
            top_k=request.top_k,
            subsystem_filter=request.subsystem_filter,
        )
    except Exception as e:
        logger.error("Query processing failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {type(e).__name__}",
        ) from e


@app.get("/specifications", response_model=list[VehicleSpecification], tags=["Specification Retrieval"])
async def list_specifications(
    subsystem: str | None = Query(default=None, description="Filter by subsystem code (e.g. 'front_brakes')"),
    spec_type: str | None = Query(default=None, description="Filter by specification category"),
    search: str | None = Query(default=None, description="Filter component name"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    pipeline: AutoSpecPipeline = Depends(get_pipeline),
) -> list[VehicleSpecification]:
    """List indexed specifications with filtering, pagination, and sorting."""
    all_specs = pipeline.get_all_specifications()

    filtered = all_specs
    if subsystem:
        filtered = [s for s in filtered if (s.subsystem or "").lower() == subsystem.lower()]
    if spec_type:
        filtered = [s for s in filtered if s.spec_type.lower() == spec_type.lower()]
    if search:
        search_low = search.lower()
        filtered = [s for s in filtered if search_low in s.component.lower() or search_low in s.value]

    return filtered[offset : offset + limit]


@app.get("/export/json", tags=["Exports"])
async def export_specifications_json(
    subsystem: str | None = None,
    spec_type: str | None = None,
    pipeline: AutoSpecPipeline = Depends(get_pipeline),
):
    """Export specifications as a downloadable JSON file."""
    specs = pipeline.get_all_specifications()
    if subsystem:
        specs = [s for s in specs if (s.subsystem or "").lower() == subsystem.lower()]
    if spec_type:
        specs = [s for s in specs if s.spec_type.lower() == spec_type.lower()]

    json_data = ExportManager.to_json(specs)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=autospec_specifications.json"},
    )


@app.get("/export/csv", tags=["Exports"])
async def export_specifications_csv(
    subsystem: str | None = None,
    spec_type: str | None = None,
    pipeline: AutoSpecPipeline = Depends(get_pipeline),
):
    """Export specifications as a downloadable CSV spreadsheet."""
    specs = pipeline.get_all_specifications()
    if subsystem:
        specs = [s for s in specs if (s.subsystem or "").lower() == subsystem.lower()]
    if spec_type:
        specs = [s for s in specs if s.spec_type.lower() == spec_type.lower()]

    csv_data = ExportManager.to_csv(specs)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=autospec_specifications.csv"},
    )
