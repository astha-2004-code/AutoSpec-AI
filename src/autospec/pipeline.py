"""End-to-end AutoSpec AI pipeline orchestrating ingestion, hybrid search, extraction, and validation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from autospec.chunker import SpecificationChunk, TableAwareChunker
from autospec.cleaner import TextCleaner
from autospec.config import settings, validate_pdf_bytes, validate_pdf_path
from autospec.embeddings import CandidateReranker, EmbedderProtocol, get_embedder
from autospec.evidence_validator import EvidenceValidator
from autospec.hybrid_retriever import HybridRetriever, RetrievalMode
from autospec.keyword_retriever import BM25KeywordRetriever
from autospec.pdf_parser import DocumentMetadata, PDFParser
from autospec.query_router import QueryRouter
from autospec.schemas import (
    DocumentInfo,
    EvidenceBBox,
    IngestResponse,
    QueryResponse,
    VehicleSpecification,
    VerificationStatus,
)
from autospec.section_detector import SectionDetector
from autospec.spec_extractor import SpecificationExtractor
from autospec.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class AutoSpecPipeline:
    """Document-scoped or multi-document pipeline connecting parsing, hybrid indexing, extraction, and verification."""

    def __init__(
        self,
        embedder: EmbedderProtocol | None = None,
        llm_provider: str | None = None,
        index_dir: Path | None = None,
        enable_ocr: bool | None = None,
    ) -> None:
        self.parser = PDFParser(enable_ocr=enable_ocr)
        self.cleaner = TextCleaner()
        self.section_detector = SectionDetector()
        self.chunker = TableAwareChunker(cleaner=self.cleaner, section_detector=self.section_detector)
        self.embedder = embedder or get_embedder()

        self.vector_store = FAISSVectorStore(
            index_dir=index_dir,
            embedding_dim=self.embedder.dimension,
        )
        self.keyword_retriever = BM25KeywordRetriever()
        self.query_router = QueryRouter()
        self.reranker = CandidateReranker()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            keyword_retriever=self.keyword_retriever,
            embedder=self.embedder,
            query_router=self.query_router,
            reranker=self.reranker,
        )

        self.extractor = SpecificationExtractor(provider=llm_provider or settings.llm_provider)
        self.validator = EvidenceValidator(cleaner=self.cleaner)

        self.current_document: DocumentMetadata | None = None
        self.page_text_map: dict[int, str] = {}
        self.all_spec_chunks: list[SpecificationChunk] = []
        self.last_embeddings: list[list[float]] = []

    def ingest_file(
        self,
        pdf_path_str: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
        force_rebuild: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        document_id: str = "",
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
    ) -> IngestResponse:
        """Ingest a local PDF manual, using fingerprint caching if already indexed."""
        start_t = time.perf_counter()
        validated_path = validate_pdf_path(pdf_path_str)
        fingerprint = PDFParser.compute_fingerprint_file(validated_path)
        cache_key = document_id or fingerprint
        bm25_path = self.vector_store.index_dir / f"{cache_key}.bm25"

        if not force_rebuild and self.vector_store.is_fresh(cache_key) and bm25_path.exists():
            if self.load_from_cache(cache_key):
                manifest = self.vector_store.manifest
                elapsed_ms = (time.perf_counter() - start_t) * 1000
                return IngestResponse(
                    status="success",
                    document_id=cache_key,
                    filename=manifest.get("document_filename", validated_path.name),
                    fingerprint=fingerprint,
                    total_pages=manifest.get("total_pages", len(self.page_text_map)),
                    chunk_count=len(self.all_spec_chunks),
                    is_cached=True,
                    processing_time_ms=round(elapsed_ms, 2),
                    message="Loaded pre-indexed specification index from cache.",
                )

        doc_metadata = self.parser.parse_file(
            validated_path,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        return self._index_parsed_document(
            doc_metadata=doc_metadata,
            filename=doc_metadata.filename,
            cache_key=cache_key,
            start_t=start_t,
            manufacturer=manufacturer,
            model=model,
            year=year,
        )

    def ingest_bytes(
        self,
        data: bytes,
        filename: str = "uploaded_manual.pdf",
        start_page: int | None = None,
        end_page: int | None = None,
        force_rebuild: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        document_id: str = "",
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
    ) -> IngestResponse:
        """Ingest raw PDF bytes uploaded via UI or REST API."""
        start_t = time.perf_counter()
        validate_pdf_bytes(data, filename=filename)
        fingerprint = PDFParser.compute_fingerprint_bytes(data)
        cache_key = document_id or fingerprint
        bm25_path = self.vector_store.index_dir / f"{cache_key}.bm25"

        if not force_rebuild and self.vector_store.is_fresh(cache_key) and bm25_path.exists():
            if self.load_from_cache(cache_key):
                manifest = self.vector_store.manifest
                elapsed_ms = (time.perf_counter() - start_t) * 1000
                return IngestResponse(
                    status="success",
                    document_id=cache_key,
                    filename=filename,
                    fingerprint=fingerprint,
                    total_pages=manifest.get("total_pages", len(self.page_text_map)),
                    chunk_count=len(self.all_spec_chunks),
                    is_cached=True,
                    processing_time_ms=round(elapsed_ms, 2),
                    message="Loaded pre-indexed specification index from cache.",
                )

        doc_metadata = self.parser.parse_bytes(
            data,
            filename=filename,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        return self._index_parsed_document(
            doc_metadata=doc_metadata,
            filename=filename,
            cache_key=cache_key,
            start_t=start_t,
            manufacturer=manufacturer,
            model=model,
            year=year,
        )

    def _index_parsed_document(
        self,
        doc_metadata: DocumentMetadata,
        filename: str,
        cache_key: str,
        start_t: float,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
    ) -> IngestResponse:
        """Chunk, embed, and persist FAISS + BM25 + page evidence for a parsed document."""
        self.current_document = doc_metadata
        self.page_text_map = {p.page_number: p.raw_text for p in doc_metadata.pages}
        page_meta_map = {
            p.page_number: {
                "printed_page_label": p.printed_page_label or str(p.page_number),
                "page_width": p.page_width,
                "page_height": p.page_height,
                "rotation": p.rotation,
                "ocr_used": p.ocr_used,
            }
            for p in doc_metadata.pages
        }

        first_pages_text = "\n".join(p.raw_text for p in doc_metadata.pages[:3])
        inferred = self.section_detector.detect_vehicle_metadata(first_pages_text, filename)
        v_make = manufacturer or inferred.manufacturer
        v_model = model or inferred.model
        v_year = year if year is not None else inferred.year

        chunks = self.chunker.chunk_document(
            doc_metadata.pages,
            filename,
            document_id=cache_key,
            vehicle_make=v_make,
            vehicle_model=v_model,
            vehicle_year=v_year,
        )
        if not chunks:
            for p in doc_metadata.pages:
                chunks.append(
                    SpecificationChunk(
                        chunk_id=f"{filename}_p{p.page_number}_raw",
                        document_id=cache_key,
                        document_name=filename,
                        page_number=p.page_number,
                        printed_page_label=p.printed_page_label or str(p.page_number),
                        section_id="RAW",
                        section_title="General Text",
                        subsystem="general",
                        chunk_type="text_passage",
                        content=p.raw_text[:800],
                        raw_evidence=p.raw_text[:800],
                        vehicle_make=v_make,
                        vehicle_model=v_model,
                        vehicle_year=v_year,
                    )
                )

        self.all_spec_chunks = chunks
        texts_to_embed = [ch.content for ch in chunks]
        embeddings = self.embedder.embed_texts(texts_to_embed)
        self.last_embeddings = embeddings

        self.vector_store.build_index(
            fingerprint=cache_key,
            document_filename=filename,
            chunks=chunks,
            embeddings=embeddings,
            model_name=getattr(self.embedder, "model_name", "custom"),
            total_pages=doc_metadata.total_pages,
            page_text_map=self.page_text_map,
            page_meta_map=page_meta_map,
            extra_manifest={
                "pdf_sha256": doc_metadata.fingerprint,
                "manufacturer": v_make,
                "model": v_model,
                "year": v_year,
                "ocr_page_count": doc_metadata.ocr_page_count,
                "file_size_bytes": doc_metadata.file_size_bytes,
            },
        )

        bm25_path = self.vector_store.index_dir / f"{cache_key}.bm25"
        self.keyword_retriever.build_index(chunks)
        self.keyword_retriever.save(bm25_path)

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        return IngestResponse(
            status="success",
            document_id=cache_key,
            filename=filename,
            fingerprint=doc_metadata.fingerprint,
            total_pages=doc_metadata.total_pages,
            chunk_count=len(chunks),
            is_cached=False,
            processing_time_ms=round(elapsed_ms, 2),
            message=f"Successfully extracted {len(chunks)} specification chunks across {doc_metadata.extracted_page_count} pages.",
        )

    def load_from_cache(self, cache_key: str) -> bool:
        """Load FAISS, chunks, BM25, and page_text_map safely from disk cache."""
        if not self.vector_store.load_index(cache_key):
            return False
        self.all_spec_chunks = self.vector_store.chunks
        self.page_text_map = dict(self.vector_store.page_text_map)

        # Fallback: if legacy cache didn't save _pages.json, reconstruct page text from passage chunks
        if not self.page_text_map:
            reconstructed: dict[int, list[str]] = {}
            for ch in self.all_spec_chunks:
                reconstructed.setdefault(ch.page_number, []).append(ch.raw_evidence)
            self.page_text_map = {p: "\n".join(parts) for p, parts in reconstructed.items()}

        bm25_path = self.vector_store.index_dir / f"{cache_key}.bm25"
        if not self.keyword_retriever.load(bm25_path, self.all_spec_chunks):
            if self.all_spec_chunks:
                self.keyword_retriever.build_index(self.all_spec_chunks)
                self.keyword_retriever.save(bm25_path)
        return True

    def load_from_records(
        self,
        cache_key: str,
        filename: str,
        page_text_map: dict[int, str],
        chunks: list[SpecificationChunk],
    ) -> bool:
        """Hydrate an in-memory pipeline directly from PostgreSQL/SQLite persisted records."""
        if not chunks:
            return False
        self.all_spec_chunks = chunks
        self.page_text_map = dict(page_text_map)
        self.keyword_retriever.build_index(chunks)
        embeddings = self.embedder.embed_texts([ch.content for ch in chunks])
        self.last_embeddings = embeddings
        self.vector_store.build_index(
            fingerprint=cache_key,
            document_filename=filename,
            chunks=chunks,
            embeddings=embeddings,
            model_name=getattr(self.embedder, "model_name", "custom"),
            total_pages=max(page_text_map.keys()) if page_text_map else 0,
            page_text_map=self.page_text_map,
        )
        return True

    def query(
        self,
        question: str,
        top_k: int = 5,
        subsystem_filter: str | None = None,
        document_ids: set[str] | None = None,
        manufacturer_filter: str | None = None,
        model_filter: str | None = None,
        year_filter: int | None = None,
        retrieval_mode: RetrievalMode = "hybrid_rerank",
        min_score: float | None = None,
    ) -> QueryResponse:
        """Execute hybrid search, extract structured specifications, and validate evidence."""
        start_t = time.perf_counter()
        parsed = self.query_router.parse_query(question)

        hits = self.retriever.retrieve(
            question,
            top_k=top_k,
            subsystem_filter=subsystem_filter,
            document_ids=document_ids,
            manufacturer_filter=manufacturer_filter,
            model_filter=model_filter,
            year_filter=year_filter,
            retrieval_mode=retrieval_mode,
            min_score=min_score,
        )

        if not hits:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            return QueryResponse(
                query=question,
                normalized_query=parsed.normalized_query,
                results=[],
                total_results=0,
                extraction_method=self.extractor.provider,
                retrieval_mode=retrieval_mode,
                processing_time_ms=round(elapsed_ms, 2),
                status="not_found",
                missing_info_explanation=(
                    "No indexed passages or specification rows matched the requested component, subsystem, or vehicle filters."
                ),
                message="No relevant automotive specifications found for this query in the indexed manual(s).",
            )

        extracted_specs = self.extractor.extract(question, hits)
        if not extracted_specs:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            return QueryResponse(
                query=question,
                normalized_query=parsed.normalized_query,
                results=[],
                total_results=0,
                extraction_method=self.extractor.provider,
                retrieval_mode=retrieval_mode,
                processing_time_ms=round(elapsed_ms, 2),
                status="insufficient_evidence",
                missing_info_explanation=(
                    "Candidate passages were retrieved, but none contained an explicit, verifiable specification matching all query constraints."
                ),
                message="Insufficient evidence to provide a verified specification.",
            )

        page_map = self.page_text_map if self.page_text_map else None
        validated_specs = self.validator.validate_batch(
            extracted_specs,
            page_text_map=page_map,
            query=question,
        )

        # Detect cross-subsystem ambiguity when query omits subsystem (e.g., front vs rear)
        clarification_needed = False
        clarification_prompt: str | None = None
        if not subsystem_filter and not parsed.target_subsystem and len(validated_specs) > 1:
            subsystems = {s.subsystem for s in validated_specs if s.subsystem}
            if len(subsystems) > 1 and any(
                validated_specs[0].component.lower() == s.component.lower()
                for s in validated_specs[1:]
            ):
                clarification_needed = True
                clarification_prompt = (
                    f"Multiple subsystems match '{validated_specs[0].component}' ({', '.join(sorted(subsystems))}). "
                    "Please specify which subsystem or vehicle variant you need."
                )

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        active_method = validated_specs[0].extraction_method if validated_specs else self.extractor.provider

        return QueryResponse(
            query=question,
            normalized_query=parsed.normalized_query,
            results=validated_specs,
            total_results=len(validated_specs),
            extraction_method=active_method,
            retrieval_mode=retrieval_mode,
            processing_time_ms=round(elapsed_ms, 2),
            status="clarification_needed" if clarification_needed else "answered",
            clarification_needed=clarification_needed,
            clarification_prompt=clarification_prompt,
            message=f"Extracted and validated {len(validated_specs)} specification records.",
        )

    def get_document_info(self) -> DocumentInfo | None:
        """Return information about the currently indexed manual."""
        if not self.vector_store.manifest:
            return None

        m = self.vector_store.manifest
        subsystems = sorted({ch.subsystem for ch in self.all_spec_chunks if ch.subsystem})
        spec_count = sum(1 for ch in self.all_spec_chunks if ch.chunk_type == "spec_row")
        return DocumentInfo(
            document_id=m.get("fingerprint", ""),
            filename=m.get("document_filename", "Unknown"),
            display_name=m.get("document_filename", "Unknown"),
            fingerprint=m.get("pdf_sha256", m.get("fingerprint", "Unknown")),
            total_pages=m.get("total_pages", 0),
            indexed_chunks=m.get("chunk_count", 0),
            spec_count=spec_count,
            freshness="Indexed & Verified",
            created_at=m.get("created_at", "N/A"),
            manufacturer=m.get("manufacturer", "Unknown"),
            model=m.get("model", "Unknown"),
            year=m.get("year"),
            subsystems=subsystems,
            ocr_pages=m.get("ocr_page_count", 0),
            file_size_bytes=m.get("file_size_bytes", 0),
        )

    def get_all_specifications(self) -> list[VehicleSpecification]:
        """Convert all indexed spec_row chunks into strictly validated VehicleSpecification models."""
        candidates: list[VehicleSpecification] = []
        for ch in self.all_spec_chunks:
            if ch.chunk_type == "spec_row" and ch.component and (ch.primary_value or ch.part_number):
                bboxes = [EvidenceBBox(**b) for b in ch.evidence_bboxes if isinstance(b, dict)]
                stype = ch.metadata.get("spec_type", "torque")
                if stype == "dimension":
                    stype = "dimension_clearance"
                spec = VehicleSpecification(
                    document_id=ch.document_id or None,
                    component=ch.component,
                    spec_type=stype,
                    value=ch.primary_value,
                    unit=ch.primary_unit,
                    alternate_value=ch.alternate_value or None,
                    alternate_unit=ch.alternate_unit or None,
                    range_min=ch.range_min or None,
                    range_max=ch.range_max or None,
                    tolerance=ch.tolerance or None,
                    conditions=ch.conditions or None,
                    part_number=ch.part_number or None,
                    material=ch.metadata.get("ford_spec") or None,
                    section=f"SECTION {ch.section_id}: {ch.section_title}",
                    subsystem=ch.subsystem,
                    page_number=ch.page_number,
                    printed_page_label=ch.printed_page_label or str(ch.page_number),
                    source_document=ch.document_name,
                    vehicle_make=ch.vehicle_make or None,
                    vehicle_model=ch.vehicle_model or None,
                    vehicle_year=ch.vehicle_year,
                    source_quote=ch.raw_evidence,
                    evidence_bboxes=bboxes,
                    verification_status=VerificationStatus.UNVERIFIED.value,
                    extraction_method="table_parser",
                )
                candidates.append(spec)

        page_map = self.page_text_map if self.page_text_map else None
        return self.validator.validate_batch(candidates, page_text_map=page_map)
