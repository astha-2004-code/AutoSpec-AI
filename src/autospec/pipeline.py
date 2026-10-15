"""End-to-end AutoSpec AI pipeline orchestrating ingestion, hybrid search, extraction, and validation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from autospec.chunker import SpecificationChunk, TableAwareChunker
from autospec.cleaner import TextCleaner
from autospec.config import settings, validate_pdf_bytes, validate_pdf_path
from autospec.embeddings import EmbedderProtocol, get_embedder
from autospec.evidence_validator import EvidenceValidator
from autospec.hybrid_retriever import HybridRetriever
from autospec.keyword_retriever import BM25KeywordRetriever
from autospec.pdf_parser import DocumentMetadata, PDFParser
from autospec.query_router import QueryRouter
from autospec.schemas import DocumentInfo, IngestResponse, QueryResponse, VehicleSpecification
from autospec.section_detector import SectionDetector
from autospec.spec_extractor import SpecificationExtractor
from autospec.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class AutoSpecPipeline:
    """Production pipeline connecting document parsing, hybrid indexing, LLM/fallback extraction, and verification."""

    def __init__(
        self,
        embedder: EmbedderProtocol | None = None,
        llm_provider: str | None = None,
    ) -> None:
        self.parser = PDFParser()
        self.cleaner = TextCleaner()
        self.section_detector = SectionDetector()
        self.chunker = TableAwareChunker(cleaner=self.cleaner, section_detector=self.section_detector)
        self.embedder = embedder or get_embedder()

        self.vector_store = FAISSVectorStore(embedding_dim=self.embedder.dimension)
        self.keyword_retriever = BM25KeywordRetriever()
        self.query_router = QueryRouter()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            keyword_retriever=self.keyword_retriever,
            embedder=self.embedder,
            query_router=self.query_router,
        )

        self.extractor = SpecificationExtractor(provider=llm_provider or settings.llm_provider)
        self.validator = EvidenceValidator(cleaner=self.cleaner)

        self.current_document: DocumentMetadata | None = None
        self.page_text_map: dict[int, str] = {}
        self.all_spec_chunks: list[SpecificationChunk] = []

    def ingest_file(
        self,
        pdf_path_str: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
        force_rebuild: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IngestResponse:
        """Ingest a local PDF manual, using fingerprint caching if already indexed."""
        start_t = time.perf_counter()
        validated_path = validate_pdf_path(pdf_path_str)
        fingerprint = PDFParser.compute_fingerprint_file(validated_path)

        bm25_path = settings.get_index_dir() / f"{fingerprint}.bm25"

        # Check if index is cached and valid
        if not force_rebuild and self.vector_store.is_fresh(fingerprint) and bm25_path.exists():
            self.vector_store.load_index(fingerprint)
            self.all_spec_chunks = self.vector_store.chunks
            self.keyword_retriever.load(bm25_path, self.all_spec_chunks)

            # Fast metadata reconstruction
            manifest = self.vector_store.manifest
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            return IngestResponse(
                status="success",
                filename=manifest.get("document_filename", validated_path.name),
                fingerprint=fingerprint,
                total_pages=manifest.get("total_pages", len(self.vector_store.chunks)),
                chunk_count=len(self.vector_store.chunks),
                is_cached=True,
                processing_time_ms=round(elapsed_ms, 2),
                message="Loaded pre-indexed specification index from cache.",
            )

        # Parse document
        doc_metadata = self.parser.parse_file(
            validated_path,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
        )
        self.current_document = doc_metadata
        self.page_text_map = {p.page_number: p.raw_text for p in doc_metadata.pages}

        # Chunk document
        chunks = self.chunker.chunk_document(doc_metadata.pages, doc_metadata.filename)
        if not chunks:
            # Fallback if no specs: add page passages
            for p in doc_metadata.pages:
                chunks.append(
                    SpecificationChunk(
                        chunk_id=f"{doc_metadata.filename}_p{p.page_number}_raw",
                        document_name=doc_metadata.filename,
                        page_number=p.page_number,
                        section_id="RAW",
                        section_title="General Text",
                        subsystem="general",
                        chunk_type="text_passage",
                        content=p.raw_text[:500],
                        raw_evidence=p.raw_text[:500],
                    )
                )

        self.all_spec_chunks = chunks

        # Compute embeddings
        texts_to_embed = [ch.content for ch in chunks]
        embeddings = self.embedder.embed_texts(texts_to_embed)

        # Build & persist FAISS index
        self.vector_store.build_index(
            fingerprint=fingerprint,
            document_filename=doc_metadata.filename,
            chunks=chunks,
            embeddings=embeddings,
            model_name=getattr(self.embedder, "model_name", "custom"),
        )
        # Update manifest with total_pages
        self.vector_store.manifest["total_pages"] = doc_metadata.total_pages

        # Build & persist BM25 index
        self.keyword_retriever.build_index(chunks)
        self.keyword_retriever.save(bm25_path)

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        return IngestResponse(
            status="success",
            filename=doc_metadata.filename,
            fingerprint=fingerprint,
            total_pages=doc_metadata.total_pages,
            chunk_count=len(chunks),
            is_cached=False,
            processing_time_ms=round(elapsed_ms, 2),
            message=f"Successfully extracted {len(chunks)} specification chunks across {doc_metadata.extracted_page_count} pages.",
        )

    def ingest_bytes(
        self,
        data: bytes,
        filename: str = "uploaded_manual.pdf",
        start_page: int | None = None,
        end_page: int | None = None,
        force_rebuild: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IngestResponse:
        """Ingest raw PDF bytes uploaded via UI or REST API."""
        start_t = time.perf_counter()
        validate_pdf_bytes(data, filename=filename)
        fingerprint = PDFParser.compute_fingerprint_bytes(data)

        bm25_path = settings.get_index_dir() / f"{fingerprint}.bm25"

        if not force_rebuild and self.vector_store.is_fresh(fingerprint) and bm25_path.exists():
            self.vector_store.load_index(fingerprint)
            self.all_spec_chunks = self.vector_store.chunks
            self.keyword_retriever.load(bm25_path, self.all_spec_chunks)
            manifest = self.vector_store.manifest
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            return IngestResponse(
                status="success",
                filename=filename,
                fingerprint=fingerprint,
                total_pages=manifest.get("total_pages", 0),
                chunk_count=len(self.vector_store.chunks),
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
        )
        self.current_document = doc_metadata
        self.page_text_map = {p.page_number: p.raw_text for p in doc_metadata.pages}

        chunks = self.chunker.chunk_document(doc_metadata.pages, filename)
        if not chunks:
            for p in doc_metadata.pages:
                chunks.append(
                    SpecificationChunk(
                        chunk_id=f"{filename}_p{p.page_number}_raw",
                        document_name=filename,
                        page_number=p.page_number,
                        section_id="RAW",
                        section_title="General Text",
                        subsystem="general",
                        chunk_type="text_passage",
                        content=p.raw_text[:500],
                        raw_evidence=p.raw_text[:500],
                    )
                )

        self.all_spec_chunks = chunks
        texts_to_embed = [ch.content for ch in chunks]
        embeddings = self.embedder.embed_texts(texts_to_embed)

        self.vector_store.build_index(
            fingerprint=fingerprint,
            document_filename=filename,
            chunks=chunks,
            embeddings=embeddings,
            model_name=getattr(self.embedder, "model_name", "custom"),
        )
        self.vector_store.manifest["total_pages"] = doc_metadata.total_pages

        self.keyword_retriever.build_index(chunks)
        self.keyword_retriever.save(bm25_path)

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        return IngestResponse(
            status="success",
            filename=filename,
            fingerprint=fingerprint,
            total_pages=doc_metadata.total_pages,
            chunk_count=len(chunks),
            is_cached=False,
            processing_time_ms=round(elapsed_ms, 2),
            message=f"Successfully extracted {len(chunks)} specification chunks across {doc_metadata.extracted_page_count} pages.",
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        subsystem_filter: str | None = None,
    ) -> QueryResponse:
        """Execute hybrid search, extract structured specifications, and validate evidence."""
        start_t = time.perf_counter()

        # 1. Hybrid Retrieval
        hits = self.retriever.retrieve(
            question,
            top_k=top_k,
            subsystem_filter=subsystem_filter,
        )

        if not hits:
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            return QueryResponse(
                query=question,
                results=[],
                total_results=0,
                extraction_method=self.extractor.provider,
                processing_time_ms=round(elapsed_ms, 2),
                message="No relevant automotive specifications found for this query in the indexed manual.",
            )

        # 2. Specification Extraction
        extracted_specs = self.extractor.extract(question, hits)

        # 3. Evidence Validation
        validated_specs = self.validator.validate_batch(
            extracted_specs,
            page_text_map=self.page_text_map,
            query=question,
        )

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        active_method = validated_specs[0].extraction_method if validated_specs else self.extractor.provider

        return QueryResponse(
            query=question,
            results=validated_specs,
            total_results=len(validated_specs),
            extraction_method=active_method,
            processing_time_ms=round(elapsed_ms, 2),
            message=f"Extracted and validated {len(validated_specs)} specification records.",
        )

    def get_document_info(self) -> DocumentInfo | None:
        """Return information about the currently indexed manual."""
        if not self.vector_store.manifest:
            return None

        m = self.vector_store.manifest
        return DocumentInfo(
            filename=m.get("document_filename", "Unknown"),
            fingerprint=m.get("fingerprint", "Unknown"),
            total_pages=m.get("total_pages", 0),
            indexed_chunks=m.get("chunk_count", 0),
            freshness="Indexed & Verified",
            created_at=m.get("created_at", "N/A"),
        )

    def get_all_specifications(self) -> list[VehicleSpecification]:
        """Convert all indexed spec_row chunks into VehicleSpecification models for exploration."""
        specs: list[VehicleSpecification] = []
        for ch in self.all_spec_chunks:
            if ch.chunk_type == "spec_row" and ch.component and ch.primary_value:
                spec = VehicleSpecification(
                    component=ch.component,
                    spec_type=ch.metadata.get("spec_type", "torque"),
                    value=ch.primary_value,
                    unit=ch.primary_unit,
                    alternate_value=ch.alternate_value or None,
                    alternate_unit=ch.alternate_unit or None,
                    part_number=ch.part_number or None,
                    material=ch.metadata.get("ford_spec"),
                    section=f"SECTION {ch.section_id}: {ch.section_title}",
                    subsystem=ch.subsystem,
                    page_number=ch.page_number,
                    source_document=ch.document_name,
                    source_quote=ch.raw_evidence,
                    verification_status="verified",
                    extraction_method="table_parser",
                )
                specs.append(spec)
        return specs
