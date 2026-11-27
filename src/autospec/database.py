"""SQLAlchemy persistence layer for Neon PostgreSQL + pgvector with SQLite test fallback."""

from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    TypeDecorator,
    create_engine,
    delete,
    event,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from autospec.config import settings
from autospec.schemas import DocumentInfo, IngestionJobStatus, OverviewStats, SearchHistoryEntry


class PgVectorOrJson(TypeDecorator):
    """Uses pgvector Vector(dim) on PostgreSQL and JSON array text on SQLite."""

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        vec = [float(x) for x in value]
        if dialect.name == "postgresql":
            return vec
        return json.dumps(vec)

    def process_result_value(self, value: Any, dialect: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return [float(x) for x in json.loads(value)]
            except Exception:
                return None
        return [float(x) for x in value]


class Base(DeclarativeBase):
    """Declarative base for AutoSpec AI ORM models."""


class SessionRecord(Base):
    """Workspace browser session metadata for isolation."""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class VehicleRecord(Base):
    """Normalized vehicle metadata referenced by indexed service manuals."""

    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    manufacturer: Mapped[str] = mapped_column(String(120), default="Unknown")
    model: Mapped[str] = mapped_column(String(120), default="Unknown")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    variant: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class DocumentRecord(Base):
    """Persisted metadata and optional binary PDF blob for an indexed PDF service manual."""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_session_id: Mapped[str] = mapped_column(String(64), index=True, default="public")
    vehicle_id: Mapped[str | None] = mapped_column(
        String(120), ForeignKey("vehicles.vehicle_id", ondelete="SET NULL"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(255))
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    pdf_path: Mapped[str] = mapped_column(Text, default="")
    pdf_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    indexed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    spec_count: Mapped[int] = mapped_column(Integer, default=0)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    read_only: Mapped[bool] = mapped_column(Boolean, default=False)
    manufacturer: Mapped[str] = mapped_column(String(120), default="Unknown")
    model: Mapped[str] = mapped_column(String(120), default="Unknown")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    subsystems_json: Mapped[str] = mapped_column(Text, default="[]")
    ocr_pages: Mapped[int] = mapped_column(Integer, default=0)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(120), default="")
    embedding_dim: Mapped[int] = mapped_column(Integer, default=384)
    freshness: Mapped[str] = mapped_column(String(64), default="Indexed & Verified")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_documents_session_fingerprint", "owner_session_id", "fingerprint"),
    )

    def to_schema(self) -> DocumentInfo:
        try:
            subsystems = json.loads(self.subsystems_json or "[]")
        except Exception:
            subsystems = []
        return DocumentInfo(
            document_id=self.document_id,
            filename=self.filename,
            display_name=self.display_name,
            fingerprint=self.fingerprint,
            total_pages=self.total_pages,
            indexed_chunks=self.indexed_chunks,
            spec_count=self.spec_count,
            freshness=self.freshness,
            created_at=self.created_at.isoformat() if self.created_at else "",
            is_demo=self.is_demo,
            read_only=self.read_only,
            manufacturer=self.manufacturer,
            model=self.model,
            year=self.year,
            subsystems=subsystems,
            ocr_pages=self.ocr_pages,
            file_size_bytes=self.file_size_bytes,
        )


class DocumentPageRecord(Base):
    """Extracted text per PDF page for citation verification and restart-safe reload."""

    __tablename__ = "document_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_document_pages_doc_page", "document_id", "page_number", unique=True),)


class DocumentChunkRecord(Base):
    """Table-aware and section-aware document chunks."""

    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(255), default="")
    subsection: Mapped[str] = mapped_column(String(255), default="")
    subsystem: Mapped[str] = mapped_column(String(120), default="General")
    chunk_type: Mapped[str] = mapped_column(String(32), default="text")
    content: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    bbox_json: Mapped[str] = mapped_column(Text, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class ChunkEmbeddingRecord(Base):
    """Vector embeddings stored in pgvector (or JSON fallback for SQLite unit tests)."""

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    embedding: Mapped[list[float]] = mapped_column(PgVectorOrJson(settings.embedding_dim))


class SpecificationRecord(Base):
    """Extracted and verified engineering specifications persisted for the Explorer and API."""

    __tablename__ = "specifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    component: Mapped[str] = mapped_column(String(255), default="Unknown")
    subsystem: Mapped[str] = mapped_column(String(120), default="General", index=True)
    spec_type: Mapped[str] = mapped_column(String(120), default="Specification")
    value: Mapped[str] = mapped_column(String(255), default="")
    unit: Mapped[str] = mapped_column(String(64), default="")
    tolerance: Mapped[str | None] = mapped_column(String(120), nullable=True)
    condition: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification_status: Mapped[str] = mapped_column(String(64), default="UNVERIFIED")
    verification_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    vehicle_json: Mapped[str] = mapped_column(Text, default="{}")
    bbox_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class CitationRecord(Base):
    """Exact page and quote grounding citations linked to extracted specifications."""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    specification_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("specifications.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str] = mapped_column(String(128), default="")
    page_number: Mapped[int] = mapped_column(Integer, default=1)
    quote: Mapped[str] = mapped_column(Text, default="")
    bbox_json: Mapped[str] = mapped_column(Text, default="[]")


class IngestionJobRecord(Base):
    """Persisted background PDF ingestion job state."""

    __tablename__ = "ingestion_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_session_id: Mapped[str] = mapped_column(String(64), index=True)
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    stage: Mapped[str] = mapped_column(String(64), default="uploading")
    current_page: Mapped[int] = mapped_column(Integer, default=0)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    source_pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_schema(self) -> IngestionJobStatus:
        return IngestionJobStatus(
            job_id=self.job_id,
            document_id=self.document_id,
            filename=self.filename,
            status=self.status,
            stage=self.stage,
            current_page=self.current_page,
            total_pages=self.total_pages,
            progress_pct=round(self.progress_pct, 1),
            is_duplicate=self.is_duplicate,
            message=self.message,
            error=self.error,
            created_at=self.created_at.isoformat() if self.created_at else "",
            updated_at=self.updated_at.isoformat() if self.updated_at else "",
        )


class SearchHistoryRecord(Base):
    """Session-isolated specification query history."""

    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_session_id: Mapped[str] = mapped_column(String(64), index=True)
    query: Mapped[str] = mapped_column(Text)
    document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    subsystem_filter: Mapped[str | None] = mapped_column(String(120), nullable=True)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    top_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_schema(self) -> SearchHistoryEntry:
        try:
            doc_ids = json.loads(self.document_ids_json or "[]")
        except Exception:
            doc_ids = []
        return SearchHistoryEntry(
            id=self.id,
            query=self.query,
            document_ids=doc_ids,
            subsystem_filter=self.subsystem_filter,
            result_count=self.result_count,
            top_status=self.top_status,
            latency_ms=round(self.latency_ms, 2),
            created_at=self.created_at.isoformat() if self.created_at else "",
        )


class ConversationContextRecord(Base):
    """Session-isolated conversational search context."""

    __tablename__ = "conversation_contexts"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_session_id: Mapped[str] = mapped_column(String(64), index=True)
    context_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class EvaluationRunRecord(Base):
    """Persisted benchmark evaluation run summary."""

    __tablename__ = "evaluation_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(64), default="hybrid_rerank")
    dataset_version: Mapped[str] = mapped_column(String(64), default="v2")
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    top_1_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    top_3_recall: Mapped[float] = mapped_column(Float, default=0.0)
    mrr: Mapped[float] = mapped_column(Float, default=0.0)
    exact_numeric_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    unit_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    citation_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ablation_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class EvaluationResultRecord(Base):
    """Per-case evaluation result linked to an EvaluationRunRecord."""

    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evaluation_runs.run_id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(64))
    query: Mapped[str] = mapped_column(Text)
    expected_value: Mapped[str] = mapped_column(String(255), default="")
    predicted_value: Mapped[str] = mapped_column(String(255), default="")
    verification_status: Mapped[str] = mapped_column(String(64), default="UNVERIFIED")
    is_hit_top1: Mapped[bool] = mapped_column(Boolean, default=False)
    is_hit_top3: Mapped[bool] = mapped_column(Boolean, default=False)
    exact_match: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class DatabaseManager:
    """Thread-safe SQLAlchemy repository managing Postgres+pgvector / SQLite persistence."""

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or settings.get_database_url()
        self.database_url = url
        self.is_postgres = url.startswith("postgresql")
        self._write_lock = threading.RLock()
        if url.startswith("sqlite"):
            self.engine = create_engine(
                url,
                connect_args={"check_same_thread": False, "timeout": 30.0},
                pool_pre_ping=True,
            )

            @event.listens_for(self.engine, "connect")
            def _configure_sqlite_connection(dbapi_connection: Any, _record: Any) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            self.engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=3,
                max_overflow=5,
            )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        self.ensure_pgvector_extension()
        Base.metadata.create_all(self.engine)
        if not self.is_postgres:
            self._migrate_legacy_sqlite_columns()

    def _migrate_legacy_sqlite_columns(self) -> None:
        """Ensure pre-2.0 local SQLite DB files get new columns cleanly without failing startup."""
        alters = [
            ("documents", "vehicle_id", "ALTER TABLE documents ADD COLUMN vehicle_id VARCHAR(120)"),
            ("documents", "pdf_data", "ALTER TABLE documents ADD COLUMN pdf_data BLOB"),
            ("documents", "embedding_model", "ALTER TABLE documents ADD COLUMN embedding_model VARCHAR(120) DEFAULT ''"),
            ("documents", "embedding_dim", "ALTER TABLE documents ADD COLUMN embedding_dim INTEGER DEFAULT 384"),
            ("ingestion_jobs", "is_duplicate", "ALTER TABLE ingestion_jobs ADD COLUMN is_duplicate BOOLEAN DEFAULT 0"),
            ("ingestion_jobs", "retry_count", "ALTER TABLE ingestion_jobs ADD COLUMN retry_count INTEGER DEFAULT 0"),
            ("ingestion_jobs", "source_pdf_path", "ALTER TABLE ingestion_jobs ADD COLUMN source_pdf_path TEXT"),
        ]
        try:
            with self.engine.begin() as conn:
                for table_name, col_name, ddl in alters:
                    cols = {
                        row[1]
                        for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                    }
                    if cols and col_name not in cols:
                        conn.execute(text(ddl))
        except Exception:
            pass

    def ensure_pgvector_extension(self) -> bool:
        """Enable pgvector extension when connected to PostgreSQL."""
        if not self.is_postgres:
            return False
        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            return True
        except Exception:
            return False

    def ping(self) -> tuple[bool, str, bool]:
        """Check DB connectivity and pgvector availability lightly without loading ML models."""
        backend = "postgresql" if self.is_postgres else "sqlite"
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                pgvec = False
                if self.is_postgres:
                    res = conn.execute(
                        text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                    ).scalar()
                    pgvec = bool(res)
            return True, backend, pgvec
        except Exception:
            return False, backend, False

    def get_session(self) -> Session:
        return self.SessionLocal()

    def ensure_session(self, session_id: str) -> None:
        with self._write_lock, self.get_session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec:
                rec.last_seen_at = datetime.now(UTC)
            else:
                db.add(SessionRecord(session_id=session_id))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()

    def ensure_vehicle(
        self, manufacturer: str, model: str, year: int | None, variant: str = ""
    ) -> str:
        safe_mfg = (manufacturer or "Unknown").strip()
        safe_model = (model or "Unknown").strip()
        vehicle_id = f"{safe_mfg.lower()}_{safe_model.lower()}_{year or 'na'}"[:120]
        with self._write_lock, self.get_session() as db:
            rec = db.get(VehicleRecord, vehicle_id)
            if not rec:
                db.add(
                    VehicleRecord(
                        vehicle_id=vehicle_id,
                        manufacturer=safe_mfg,
                        model=safe_model,
                        year=year,
                        variant=variant,
                    )
                )
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
        return vehicle_id

    def upsert_document(self, record: DocumentRecord) -> DocumentInfo:
        vehicle_id = self.ensure_vehicle(record.manufacturer, record.model, record.year)
        record.vehicle_id = vehicle_id
        with self._write_lock, self.get_session() as db:
            existing = db.get(DocumentRecord, record.document_id)
            if existing:
                for attr in (
                    "vehicle_id",
                    "filename",
                    "display_name",
                    "fingerprint",
                    "pdf_path",
                    "pdf_data",
                    "total_pages",
                    "indexed_chunks",
                    "spec_count",
                    "manufacturer",
                    "model",
                    "year",
                    "subsystems_json",
                    "ocr_pages",
                    "file_size_bytes",
                    "embedding_model",
                    "embedding_dim",
                    "freshness",
                    "expires_at",
                ):
                    val = getattr(record, attr)
                    if attr == "pdf_data" and val is None:
                        continue
                    setattr(existing, attr, val)
                db.commit()
                db.refresh(existing)
                return existing.to_schema()
            db.add(record)
            db.commit()
            db.refresh(record)
            return record.to_schema()

    def find_document_by_fingerprint(self, session_id: str, fingerprint: str) -> DocumentRecord | None:
        with self.get_session() as db:
            stmt = select(DocumentRecord).where(
                DocumentRecord.fingerprint == fingerprint,
                or_(DocumentRecord.is_demo.is_(True), DocumentRecord.owner_session_id == session_id),
            )
            return db.scalars(stmt).first()

    def get_document_pdf_bytes(self, document_id: str, session_id: str) -> bytes | None:
        with self.get_session() as db:
            rec = db.get(DocumentRecord, document_id)
            if not rec or (not rec.is_demo and rec.owner_session_id != session_id):
                return None
            return rec.pdf_data

    def list_authorized_documents(
        self,
        session_id: str,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
        search: str | None = None,
    ) -> list[DocumentInfo]:
        with self.get_session() as db:
            stmt = select(DocumentRecord).where(
                or_(DocumentRecord.is_demo.is_(True), DocumentRecord.owner_session_id == session_id)
            )
            rows = db.scalars(stmt.order_by(DocumentRecord.is_demo.desc(), DocumentRecord.created_at.desc())).all()
            docs = [r.to_schema() for r in rows]

        if manufacturer:
            docs = [d for d in docs if manufacturer.lower() in d.manufacturer.lower()]
        if model:
            docs = [d for d in docs if model.lower() in d.model.lower()]
        if year is not None:
            docs = [d for d in docs if d.year == year]
        if search:
            s_low = search.lower()
            docs = [
                d
                for d in docs
                if s_low in d.display_name.lower()
                or s_low in d.filename.lower()
                or s_low in d.manufacturer.lower()
                or s_low in d.model.lower()
            ]
        return docs

    def get_authorized_document(self, document_id: str, session_id: str) -> DocumentRecord | None:
        with self.get_session() as db:
            rec = db.get(DocumentRecord, document_id)
            if not rec:
                return None
            if rec.is_demo or rec.owner_session_id == session_id:
                return rec
            return None

    def update_document_metadata(
        self,
        document_id: str,
        session_id: str,
        display_name: str | None = None,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
    ) -> tuple[DocumentInfo | None, str | None]:
        with self.get_session() as db:
            rec = db.get(DocumentRecord, document_id)
            if not rec or (not rec.is_demo and rec.owner_session_id != session_id):
                return None, "not_found"
            if rec.read_only or rec.is_demo:
                return None, "read_only"
            if display_name is not None:
                rec.display_name = display_name.strip()
            if manufacturer is not None:
                rec.manufacturer = manufacturer.strip()
            if model is not None:
                rec.model = model.strip()
            if year is not None:
                rec.year = year
            db.commit()
            db.refresh(rec)
            return rec.to_schema(), None

    def delete_document(self, document_id: str, session_id: str) -> tuple[DocumentRecord | None, str | None]:
        with self.get_session() as db:
            rec = db.get(DocumentRecord, document_id)
            if not rec or (not rec.is_demo and rec.owner_session_id != session_id):
                return None, "not_found"
            if rec.read_only or rec.is_demo:
                return None, "read_only"
            db.delete(rec)
            db.commit()
            return rec, None

    def purge_expired_guest_documents(self) -> list[DocumentRecord]:
        now = datetime.now(UTC)
        with self.get_session() as db:
            stmt = select(DocumentRecord).where(
                DocumentRecord.is_demo.is_(False),
                DocumentRecord.expires_at.is_not(None),
                DocumentRecord.expires_at < now,
            )
            expired = list(db.scalars(stmt).all())
            for rec in expired:
                db.delete(rec)
            if expired:
                db.commit()
            return expired

    # Normalized pages, chunks, embeddings, specs & citations
    def replace_document_content(
        self,
        document_id: str,
        pages: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        specs: list[dict[str, Any]],
    ) -> None:
        """Persist extracted pages, chunks, vector embeddings, and specs transactionally."""
        with self._write_lock, self.get_session() as db:
            db.execute(delete(CitationRecord).where(CitationRecord.document_id == document_id))
            db.execute(delete(SpecificationRecord).where(SpecificationRecord.document_id == document_id))
            db.execute(delete(ChunkEmbeddingRecord).where(ChunkEmbeddingRecord.document_id == document_id))
            db.execute(delete(DocumentChunkRecord).where(DocumentChunkRecord.document_id == document_id))
            db.execute(delete(DocumentPageRecord).where(DocumentPageRecord.document_id == document_id))

            for p in pages:
                text_val = p.get("text", "")
                db.add(
                    DocumentPageRecord(
                        document_id=document_id,
                        page_number=int(p.get("page_number", 1)),
                        text=text_val,
                        char_count=len(text_val),
                        ocr_used=bool(p.get("ocr_used", False)),
                    )
                )

            seen_cids: set[str] = set()
            embedding_rows: list[ChunkEmbeddingRecord] = []
            for idx, c in enumerate(chunks):
                cid = str(c.get("chunk_id") or f"{document_id}_c{idx}")
                if cid in seen_cids:
                    cid = f"{cid}_{idx}"
                seen_cids.add(cid)
                db.add(
                    DocumentChunkRecord(
                        chunk_id=cid,
                        document_id=document_id,
                        page_number=int(c.get("page_number", 1)),
                        section=str(c.get("section", ""))[:255],
                        subsection=str(c.get("subsection", ""))[:255],
                        subsystem=str(c.get("subsystem", "General"))[:120],
                        chunk_type=str(c.get("chunk_type", "text"))[:32],
                        content=str(c.get("content", "")),
                        token_count=len(str(c.get("content", "")).split()),
                        bbox_json=json.dumps(c.get("bbox", [])),
                        metadata_json=json.dumps(c.get("metadata", {})),
                    )
                )
                if idx < len(embeddings):
                    embedding_rows.append(
                        ChunkEmbeddingRecord(
                            chunk_id=cid,
                            document_id=document_id,
                            embedding=[float(x) for x in embeddings[idx]],
                        )
                    )

            db.flush()
            for emb_row in embedding_rows:
                db.add(emb_row)
            db.flush()

            for s in specs:
                spec_row = SpecificationRecord(
                    document_id=document_id,
                    component=str(s.get("component", "Unknown"))[:255],
                    subsystem=str(s.get("subsystem", "General"))[:120],
                    spec_type=str(s.get("spec_type", "Specification"))[:120],
                    value=str(s.get("value", ""))[:255],
                    unit=str(s.get("unit", ""))[:64],
                    tolerance=s.get("tolerance"),
                    condition=s.get("condition"),
                    verification_status=str(s.get("verification_status", "UNVERIFIED"))[:64],
                    verification_reason=s.get("verification_reason"),
                    confidence=float(s.get("confidence", 0.0)),
                    page_number=s.get("page_number"),
                    source_quote=s.get("source_quote"),
                    chunk_id=s.get("chunk_id"),
                    vehicle_json=json.dumps(s.get("vehicle", {})),
                    bbox_json=json.dumps(s.get("bbox", [])),
                )
                db.add(spec_row)
                db.flush()
                if s.get("page_number") and s.get("source_quote"):
                    db.add(
                        CitationRecord(
                            specification_id=spec_row.id,
                            document_id=document_id,
                            chunk_id=str(s.get("chunk_id") or ""),
                            page_number=int(s["page_number"]),
                            quote=str(s["source_quote"]),
                            bbox_json=json.dumps(s.get("bbox", [])),
                        )
                    )

            db.commit()

    def load_document_pages(self, document_id: str) -> dict[int, str]:
        with self.get_session() as db:
            rows = db.scalars(
                select(DocumentPageRecord).where(DocumentPageRecord.document_id == document_id)
            ).all()
            return {r.page_number: r.text for r in rows}

    def load_document_chunks(self, document_id: str) -> list[DocumentChunkRecord]:
        with self.get_session() as db:
            return list(
                db.scalars(
                    select(DocumentChunkRecord).where(DocumentChunkRecord.document_id == document_id)
                ).all()
            )

    def update_chunk_embeddings(
        self,
        document_id: str,
        chunk_embeddings: list[tuple[str, list[float]]],
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        """Replace stored chunk vectors and update document embedding metadata safely."""
        with self._write_lock, self.get_session() as db:
            db.execute(delete(ChunkEmbeddingRecord).where(ChunkEmbeddingRecord.document_id == document_id))
            db.flush()
            for chunk_id, vec in chunk_embeddings:
                db.add(
                    ChunkEmbeddingRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        embedding=[float(x) for x in vec],
                    )
                )
            doc = db.get(DocumentRecord, document_id)
            if doc:
                doc.embedding_model = embedding_model
                doc.embedding_dim = embedding_dim
            db.commit()

    def search_pgvector_chunks(
        self,
        document_ids: list[str],
        query_embedding: list[float],
        top_k: int = 8,
        subsystem: str | None = None,
    ) -> list[tuple[DocumentChunkRecord, float]]:
        """Run pgvector cosine similarity search on PostgreSQL or pure-Python cosine on SQLite."""
        if not document_ids:
            return []
        with self.get_session() as db:
            if self.is_postgres:
                distance = ChunkEmbeddingRecord.embedding.cosine_distance(query_embedding)
                stmt = (
                    select(DocumentChunkRecord, (1.0 - distance).label("score"))
                    .join(ChunkEmbeddingRecord, ChunkEmbeddingRecord.chunk_id == DocumentChunkRecord.chunk_id)
                    .where(DocumentChunkRecord.document_id.in_(document_ids))
                )
                if subsystem:
                    stmt = stmt.where(DocumentChunkRecord.subsystem == subsystem)
                stmt = stmt.order_by(distance.asc()).limit(top_k)
                rows = db.execute(stmt).all()
                return [(chunk, float(score)) for chunk, score in rows]

            stmt = (
                select(DocumentChunkRecord, ChunkEmbeddingRecord.embedding)
                .join(ChunkEmbeddingRecord, ChunkEmbeddingRecord.chunk_id == DocumentChunkRecord.chunk_id)
                .where(DocumentChunkRecord.document_id.in_(document_ids))
            )
            if subsystem:
                stmt = stmt.where(DocumentChunkRecord.subsystem == subsystem)
            rows = db.execute(stmt).all()
            scored: list[tuple[DocumentChunkRecord, float]] = []
            for chunk, emb in rows:
                if emb:
                    scored.append((chunk, _cosine_similarity(query_embedding, list(emb))))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_k]

    def get_overview_stats(self, session_id: str) -> OverviewStats:
        docs = self.list_authorized_documents(session_id)
        doc_ids = [d.document_id for d in docs]
        with self.get_session() as db:
            query_count = int(
                db.scalar(
                    select(func.count(SearchHistoryRecord.id)).where(
                        SearchHistoryRecord.owner_session_id == session_id
                    )
                )
                or 0
            )
            verified_count = 0
            total_specs_rows = 0
            if doc_ids:
                total_specs_rows = int(
                    db.scalar(
                        select(func.count(SpecificationRecord.id)).where(
                            SpecificationRecord.document_id.in_(doc_ids)
                        )
                    )
                    or 0
                )
                verified_count = int(
                    db.scalar(
                        select(func.count(SpecificationRecord.id)).where(
                            SpecificationRecord.document_id.in_(doc_ids),
                            SpecificationRecord.verification_status.in_(
                                ["VERIFIED", "VERIFIED_WITH_OCR_WARNING"]
                            ),
                        )
                    )
                    or 0
                )

        subsystems = sorted({s for d in docs for s in d.subsystems})
        return OverviewStats(
            document_count=len(docs),
            demo_document_count=sum(1 for d in docs if d.is_demo),
            user_document_count=sum(1 for d in docs if not d.is_demo),
            total_pages=sum(d.total_pages for d in docs),
            total_chunks=sum(d.indexed_chunks for d in docs),
            total_specifications=max(total_specs_rows, sum(d.spec_count for d in docs)),
            verified_specifications=verified_count,
            query_count=query_count,
            subsystems=subsystems,
        )

    # Job management
    def create_job(
        self,
        job_id: str,
        session_id: str,
        filename: str,
        document_id: str | None = None,
        source_pdf_path: str | None = None,
        is_duplicate: bool = False,
    ) -> IngestionJobStatus:
        with self._write_lock, self.get_session() as db:
            rec = IngestionJobRecord(
                job_id=job_id,
                owner_session_id=session_id,
                document_id=document_id,
                filename=filename,
                status="queued",
                stage="uploading",
                is_duplicate=is_duplicate,
                source_pdf_path=source_pdf_path,
                message="Upload queued for processing.",
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
            return rec.to_schema()

    def update_job(self, job_id: str, **fields: Any) -> IngestionJobStatus | None:
        with self._write_lock, self.get_session() as db:
            rec = db.get(IngestionJobRecord, job_id)
            if not rec:
                return None
            for k, v in fields.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.updated_at = datetime.now(UTC)
            db.commit()
            db.refresh(rec)
            return rec.to_schema()

    def get_job(self, job_id: str, session_id: str) -> IngestionJobStatus | None:
        with self.get_session() as db:
            rec = db.get(IngestionJobRecord, job_id)
            if not rec or rec.owner_session_id != session_id:
                return None
            return rec.to_schema()

    def get_job_record(self, job_id: str, session_id: str) -> IngestionJobRecord | None:
        with self.get_session() as db:
            rec = db.get(IngestionJobRecord, job_id)
            if not rec or rec.owner_session_id != session_id:
                return None
            return rec

    def is_job_cancelled(self, job_id: str) -> bool:
        with self.get_session() as db:
            rec = db.get(IngestionJobRecord, job_id)
            return bool(rec and rec.cancel_requested)

    def request_job_cancel(self, job_id: str, session_id: str) -> IngestionJobStatus | None:
        with self._write_lock, self.get_session() as db:
            rec = db.get(IngestionJobRecord, job_id)
            if not rec or rec.owner_session_id != session_id:
                return None
            rec.cancel_requested = True
            if rec.status in ("queued", "running"):
                rec.message = "Cancellation requested..."
            rec.updated_at = datetime.now(UTC)
            db.commit()
            db.refresh(rec)
            return rec.to_schema()

    # Search history
    def add_history(
        self,
        session_id: str,
        query: str,
        document_ids: list[str],
        subsystem_filter: str | None,
        result_count: int,
        top_status: str | None,
        latency_ms: float,
    ) -> SearchHistoryEntry:
        with self.get_session() as db:
            rec = SearchHistoryRecord(
                owner_session_id=session_id,
                query=query,
                document_ids_json=json.dumps(document_ids),
                subsystem_filter=subsystem_filter,
                result_count=result_count,
                top_status=top_status,
                latency_ms=latency_ms,
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
            return rec.to_schema()

    def list_history(self, session_id: str, limit: int = 30) -> list[SearchHistoryEntry]:
        with self.get_session() as db:
            stmt = (
                select(SearchHistoryRecord)
                .where(SearchHistoryRecord.owner_session_id == session_id)
                .order_by(SearchHistoryRecord.created_at.desc())
                .limit(limit)
            )
            return [r.to_schema() for r in db.scalars(stmt).all()]

    def clear_history(self, session_id: str) -> int:
        with self.get_session() as db:
            res = db.execute(
                delete(SearchHistoryRecord).where(SearchHistoryRecord.owner_session_id == session_id)
            )
            db.commit()
            return int(res.rowcount or 0)

    def delete_history_entry(self, entry_id: int, session_id: str) -> bool:
        with self.get_session() as db:
            rec = db.get(SearchHistoryRecord, entry_id)
            if not rec or rec.owner_session_id != session_id:
                return False
            db.delete(rec)
            db.commit()
            return True

    # Conversation context
    def get_conversation_context(self, conversation_id: str, session_id: str) -> dict[str, Any]:
        with self.get_session() as db:
            rec = db.get(ConversationContextRecord, conversation_id)
            if not rec or rec.owner_session_id != session_id:
                return {}
            try:
                return json.loads(rec.context_json or "{}")
            except Exception:
                return {}

    def save_conversation_context(
        self,
        conversation_id: str,
        session_id: str,
        context: dict[str, Any],
    ) -> None:
        with self.get_session() as db:
            rec = db.get(ConversationContextRecord, conversation_id)
            if rec:
                if rec.owner_session_id != session_id:
                    return
                rec.context_json = json.dumps(context)
                rec.updated_at = datetime.now(UTC)
            else:
                db.add(
                    ConversationContextRecord(
                        conversation_id=conversation_id,
                        owner_session_id=session_id,
                        context_json=json.dumps(context),
                    )
                )
            db.commit()

    def clear_conversation_context(self, conversation_id: str, session_id: str) -> None:
        with self.get_session() as db:
            rec = db.get(ConversationContextRecord, conversation_id)
            if rec and rec.owner_session_id == session_id:
                db.delete(rec)
                db.commit()

    # Evaluation runs
    def save_evaluation_run(
        self,
        run_id: str,
        mode: str,
        dataset_version: str,
        summary: dict[str, Any],
        cases: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.get_session() as db:
            existing = db.get(EvaluationRunRecord, run_id)
            if existing:
                return
            db.add(
                EvaluationRunRecord(
                    run_id=run_id,
                    mode=mode,
                    dataset_version=dataset_version,
                    total_cases=int(summary.get("total_cases", 0)),
                    top_1_accuracy=float(summary.get("top_1_accuracy", 0.0)),
                    top_3_recall=float(summary.get("top_3_recall", 0.0)),
                    mrr=float(summary.get("mrr", 0.0)),
                    exact_numeric_accuracy=float(summary.get("exact_numeric_accuracy", 0.0)),
                    unit_accuracy=float(summary.get("unit_accuracy", 0.0)),
                    citation_accuracy=float(summary.get("citation_accuracy", 0.0)),
                    avg_latency_ms=float(summary.get("avg_latency_ms", 0.0)),
                    ablation_json=json.dumps(summary.get("ablation", {})),
                )
            )
            for case in cases or []:
                db.add(
                    EvaluationResultRecord(
                        run_id=run_id,
                        case_id=str(case.get("case_id", "")),
                        query=str(case.get("query", "")),
                        expected_value=str(case.get("expected_value", "")),
                        predicted_value=str(case.get("predicted_value", "")),
                        verification_status=str(case.get("verification_status", "UNVERIFIED")),
                        is_hit_top1=bool(case.get("is_hit_top1", False)),
                        is_hit_top3=bool(case.get("is_hit_top3", False)),
                        exact_match=bool(case.get("exact_match", False)),
                        latency_ms=float(case.get("latency_ms", 0.0)),
                    )
                )
            db.commit()

    def list_evaluation_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.get_session() as db:
            rows = db.scalars(
                select(EvaluationRunRecord).order_by(EvaluationRunRecord.created_at.desc()).limit(limit)
            ).all()
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "run_id": r.run_id,
                        "mode": r.mode,
                        "dataset_version": r.dataset_version,
                        "total_cases": r.total_cases,
                        "top_1_accuracy": r.top_1_accuracy,
                        "top_3_recall": r.top_3_recall,
                        "mrr": r.mrr,
                        "exact_numeric_accuracy": r.exact_numeric_accuracy,
                        "unit_accuracy": r.unit_accuracy,
                        "citation_accuracy": r.citation_accuracy,
                        "avg_latency_ms": r.avg_latency_ms,
                        "ablation": json.loads(r.ablation_json or "{}"),
                        "created_at": r.created_at.isoformat() if r.created_at else "",
                    }
                )
            return out


def compute_guest_expiry(hours: int | None = None) -> datetime:
    ttl = hours if hours is not None else settings.guest_upload_ttl_hours
    return datetime.now(UTC) + timedelta(hours=ttl)
