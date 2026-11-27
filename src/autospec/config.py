"""Configuration, Neon environment handling, and runtime document security validation."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOSPEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment mode",
    )
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    secret_key: str = Field(
        default="",
        description="Application secret key for signing/session HMAC (set via env in production)",
    )
    demo_mode: bool = Field(
        default=True,
        description="Enable read-only public demonstration manuals",
    )

    # Storage & Database Paths
    storage_dir: Path = Field(
        default=Path("./storage"),
        description="Base directory for local dev storage and benchmark artifacts",
    )
    index_dir: Path | None = Field(default=None)
    cache_dir: Path | None = Field(default=None)
    export_dir: Path | None = Field(default=None)
    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("AUTOSPEC_DATABASE_URL", "DATABASE_URL"),
        description="Pooled SQLAlchemy database URL (Neon Postgres in prod, SQLite fallback in local dev)",
    )
    database_url_unpooled: str = Field(
        default="",
        validation_alias=AliasChoices("AUTOSPEC_DATABASE_URL_UNPOOLED", "DATABASE_URL_UNPOOLED"),
        description="Direct unpooled Postgres URL used for Alembic schema migrations",
    )
    demo_dir: Path = Field(
        default=Path("./data/demo"),
        description="Directory containing redistributable public demonstration manuals",
    )
    benchmark_dir: Path = Field(
        default=Path("./storage/benchmarks"),
        description="Directory storing reproducible benchmark reports",
    )

    # Upload & Document Limits
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    max_pdf_db_mb: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Max PDF size stored directly in PostgreSQL BYTEA for restart persistence",
    )
    max_pdf_pages: int = Field(default=2500, ge=1, le=10000)
    default_pdf_path: str = Field(default="")
    allowed_local_roots: list[str] = Field(default_factory=list)
    guest_upload_ttl_hours: int = Field(default=24, ge=1, le=720)

    # Embedding & Retrieval Settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)
    device: str = Field(default="cpu")
    offline_models: bool = Field(default=False)
    top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    min_retrieval_score: float = Field(default=0.005, ge=0.0)
    enable_reranker: bool = Field(default=True)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_cross_encoder_model: bool = Field(default=False)
    enable_ocr: bool = Field(default=False)

    # LLM Settings
    llm_provider: Literal["none", "openai", "anthropic", "gemini", "groq"] = Field(default="none")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_api_key: str = Field(default="")
    llm_base_url: str = Field(default="")

    # API, Security & Concurrency Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:4173",
        ],
        validation_alias=AliasChoices("AUTOSPEC_ALLOWED_ORIGINS", "AUTOSPEC_CORS_ORIGINS"),
        description="Explicit approved CORS origins",
    )
    rate_limit_per_minute: int = Field(default=90, ge=5)
    max_concurrent_ingestions: int = Field(default=2, ge=1, le=8)
    request_timeout_seconds: int = Field(default=30, ge=5, le=300)

    def get_storage_path(self) -> Path:
        path = self.storage_dir.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_index_dir(self) -> Path:
        idx_dir = self.index_dir.resolve() if self.index_dir else (self.get_storage_path() / "indexes")
        idx_dir.mkdir(parents=True, exist_ok=True)
        return idx_dir

    def get_metadata_dir(self) -> Path:
        meta_dir = self.cache_dir.resolve() if self.cache_dir else (self.get_storage_path() / "metadata")
        meta_dir.mkdir(parents=True, exist_ok=True)
        return meta_dir

    def get_uploads_dir(self) -> Path:
        up_dir = self.get_storage_path() / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        return up_dir

    def get_export_dir(self) -> Path:
        exp_dir = self.export_dir.resolve() if self.export_dir else (self.get_storage_path() / "exports")
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_benchmark_dir(self) -> Path:
        b_dir = self.benchmark_dir.resolve()
        b_dir.mkdir(parents=True, exist_ok=True)
        return b_dir

    @staticmethod
    def _normalize_postgres_url(url: str) -> str:
        """Ensure postgres:// URLs map cleanly to SQLAlchemy's postgresql+psycopg2 dialect."""
        cleaned = url.strip()
        if cleaned.startswith("postgres://"):
            cleaned = "postgresql+psycopg2://" + cleaned[len("postgres://") :]
        elif cleaned.startswith("postgresql://"):
            cleaned = "postgresql+psycopg2://" + cleaned[len("postgresql://") :]
        return cleaned

    def get_database_url(self) -> str:
        """Return configured Neon/PostgreSQL URL or fallback local SQLite URL."""
        raw = self.database_url.strip() or os.environ.get("DATABASE_URL", "").strip()
        if raw:
            return self._normalize_postgres_url(raw)
        db_path = (self.get_storage_path() / "autospec.db").resolve()
        return f"sqlite:///{db_path.as_posix()}"

    def get_migration_database_url(self) -> str:
        """Prefer unpooled direct connection string for Alembic schema migrations."""
        raw = (
            self.database_url_unpooled.strip()
            or os.environ.get("DATABASE_URL_UNPOOLED", "").strip()
            or self.get_database_url()
        )
        return self._normalize_postgres_url(raw)

    def is_postgres(self) -> bool:
        return self.get_database_url().startswith("postgresql")


settings = Settings()

if settings.offline_models:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def configure_logging() -> None:
    """Configure structured logging with sensitive token redaction."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def redact_sensitive(text: str) -> str:
    """Strip credentials, API keys, and DB connection passwords from log strings."""
    if not text:
        return ""
    masked = re.sub(r"(postgres(?:ql)?(?:\+\w+)?://[^:]+:)[^@]+(@)", r"\1***\2", text)
    masked = re.sub(r"\b(sk-[A-Za-z0-9\-_]{10,}|gsk_[A-Za-z0-9\-_]{10,})\b", "***REDACTED_KEY***", masked)
    return masked


class DocumentValidationError(ValueError):
    """Raised when an input document fails runtime security or format validation."""


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_filename(filename: str, default: str = "uploaded_manual.pdf") -> str:
    """Sanitize a user-supplied filename to prevent path traversal and control-char injection."""
    if not filename or not filename.strip():
        return default
    base = filename.replace("\\", "/").split("/")[-1].strip()
    base = base.replace("\x00", "")
    base = _SAFE_FILENAME_RE.sub("_", base).strip(". ")
    if not base or base in (".", ".."):
        return default
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    return base[:160]


def sanitize_session_id(session_id: str | None) -> str:
    """Validate and normalize an anonymous/guest workspace session identifier."""
    if not session_id:
        return "public"
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", session_id.strip())
    return cleaned[:64] or "public"


_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
    re.compile(r"system\s*prompt\s*:", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|in)\s+", re.I),
    re.compile(r"</?(?:system|assistant|instruction|prompt)>", re.I),
    re.compile(r"override\s+(?:safety|verification|system)", re.I),
]


def sanitize_prompt_text(text: str) -> str:
    """Neutralize potential prompt-injection directives embedded inside untrusted manual text."""
    if not text:
        return ""
    sanitized = text
    for pat in _INJECTION_PATTERNS:
        sanitized = pat.sub("[UNTRUSTED_TEXT_REDACTED]", sanitized)
    return sanitized


def validate_pdf_bytes(
    data: bytes,
    filename: str = "upload.pdf",
    max_size_mb: int | None = None,
    max_pages: int | None = None,
    verify_structure: bool = True,
) -> None:
    """Validate raw PDF bytes for size, magic header, page limits, and structural integrity."""
    safe_name = sanitize_filename(filename)
    effective_max_mb = max_size_mb if max_size_mb is not None else settings.max_upload_size_mb
    effective_max_pages = max_pages if max_pages is not None else settings.max_pdf_pages

    if not data:
        raise DocumentValidationError(f"File '{safe_name}' is empty (0 bytes).")

    max_bytes = effective_max_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise DocumentValidationError(
            f"File '{safe_name}' exceeds maximum allowed size of {effective_max_mb} MB ({len(data)} bytes)."
        )

    prefix = data[:1024]
    if b"%PDF-" not in prefix:
        raise DocumentValidationError(
            f"File '{safe_name}' does not contain a valid PDF header (%PDF-)."
        )

    if verify_structure:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as err:
            raise DocumentValidationError(
                f"File '{safe_name}' is corrupted or unreadable as a PDF document."
            ) from err

        try:
            if doc.is_encrypted:
                raise DocumentValidationError(
                    f"File '{safe_name}' is password-encrypted. Please provide an unencrypted manual."
                )
            page_count = len(doc)
            if page_count <= 0:
                raise DocumentValidationError(f"File '{safe_name}' contains 0 pages.")
            if page_count > effective_max_pages:
                raise DocumentValidationError(
                    f"File '{safe_name}' has {page_count} pages, exceeding the limit of {effective_max_pages} pages."
                )
        finally:
            doc.close()


def validate_pdf_path(
    path_str: str | Path,
    max_size_mb: int | None = None,
    max_pages: int | None = None,
    allowed_roots: list[str] | None = None,
) -> Path:
    """Validate a local PDF file path for CLI/offline ingestion."""
    if not path_str:
        raise DocumentValidationError("No PDF path provided.")

    raw_str = str(path_str)
    if "\x00" in raw_str:
        raise DocumentValidationError("Invalid null byte in PDF path.")

    if ".." in Path(raw_str).parts:
        raise DocumentValidationError("Path traversal segments ('..') are not permitted.")

    path = Path(raw_str).expanduser()

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as err:
        raise DocumentValidationError(f"Document not found at path: {path}") from err
    except PermissionError as err:
        raise DocumentValidationError(f"Permission denied accessing path: {path}") from err

    if not resolved.is_file():
        raise DocumentValidationError(f"Path is not a regular file: {resolved}")

    if resolved.suffix.lower() != ".pdf":
        raise DocumentValidationError(f"File '{resolved.name}' must have a .pdf extension.")

    roots = allowed_roots if allowed_roots is not None else settings.allowed_local_roots
    if roots:
        in_allowed = False
        for root in roots:
            try:
                resolved.relative_to(Path(root).expanduser().resolve())
                in_allowed = True
                break
            except ValueError:
                continue
        if not in_allowed:
            raise DocumentValidationError(
                f"Access denied: '{resolved.name}' is outside configured allowed document directories."
            )

    effective_max_mb = max_size_mb if max_size_mb is not None else settings.max_upload_size_mb
    size_bytes = resolved.stat().st_size
    if size_bytes == 0:
        raise DocumentValidationError(f"PDF file is empty (0 bytes): {resolved}")

    max_bytes = effective_max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise DocumentValidationError(
            f"File exceeds maximum size limit of {effective_max_mb} MB ({size_bytes} bytes): {resolved}"
        )

    with resolved.open("rb") as f:
        header = f.read(1024)
        if b"%PDF-" not in header:
            raise DocumentValidationError(
                f"File '{resolved.name}' is not a valid PDF document (missing %PDF- header)."
            )

    effective_max_pages = max_pages if max_pages is not None else settings.max_pdf_pages
    try:
        doc = pymupdf.open(str(resolved))
        try:
            if doc.is_encrypted:
                raise DocumentValidationError(f"File '{resolved.name}' is password-encrypted.")
            if len(doc) <= 0:
                raise DocumentValidationError(f"File '{resolved.name}' contains 0 pages.")
            if len(doc) > effective_max_pages:
                raise DocumentValidationError(
                    f"File '{resolved.name}' exceeds {effective_max_pages} pages."
                )
        finally:
            doc.close()
    except DocumentValidationError:
        raise
    except Exception as err:
        raise DocumentValidationError(f"Invalid PDF structure in '{resolved.name}'.") from err

    return resolved
