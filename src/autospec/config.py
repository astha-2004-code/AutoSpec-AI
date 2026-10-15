"""Configuration and runtime document validation for AutoSpec AI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOSPEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage Paths
    storage_dir: Path = Field(
        default=Path("./storage"),
        description="Base directory for persistence of indexes, chunks, and metadata",
    )
    max_upload_size_mb: int = Field(
        default=100,
        description="Maximum permitted PDF upload size in megabytes",
    )
    default_pdf_path: str = Field(
        default="",
        description="Optional pre-configured local PDF file path",
    )

    # Embedding & Retrieval Settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace model identifier for local semantic embeddings",
    )
    embedding_dim: int = Field(
        default=384,
        description="Vector dimension for the embedding model",
    )
    device: str = Field(
        default="cpu",
        description="Compute device for embedding inference ('cpu' or 'cuda')",
    )
    top_k: int = Field(
        default=5,
        description="Default number of passages retrieved per query",
    )
    rrf_k: int = Field(
        default=60,
        description="Constant smoothing factor for Reciprocal Rank Fusion",
    )
    semantic_weight: float = Field(
        default=0.5,
        description="Weight of dense vector search in hybrid fusion",
    )
    keyword_weight: float = Field(
        default=0.5,
        description="Weight of sparse BM25 search in hybrid fusion",
    )

    # LLM Settings
    llm_provider: Literal["none", "openai", "anthropic", "gemini", "groq"] = Field(
        default="none",
        description="Configured LLM provider; 'none' activates high-precision deterministic fallback",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier if LLM provider is active",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for active LLM provider (never logged or exposed)",
    )
    llm_base_url: str = Field(
        default="",
        description="Optional custom API base URL for OpenAI-compatible providers",
    )

    # Server Settings
    api_host: str = Field(default="127.0.0.1", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    ui_port: int = Field(default=8501, description="Streamlit port")

    def get_storage_path(self) -> Path:
        """Return resolved base storage path and ensure it exists."""
        path = self.storage_dir.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_index_dir(self) -> Path:
        """Return directory for FAISS and BM25 index artifacts."""
        idx_dir = self.get_storage_path() / "indexes"
        idx_dir.mkdir(parents=True, exist_ok=True)
        return idx_dir

    def get_metadata_dir(self) -> Path:
        """Return directory for chunk and document metadata."""
        meta_dir = self.get_storage_path() / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        return meta_dir

    def get_uploads_dir(self) -> Path:
        """Return directory for runtime uploaded PDFs."""
        up_dir = self.get_storage_path() / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        return up_dir


# Global settings singleton
settings = Settings()


class DocumentValidationError(ValueError):
    """Raised when an input document fails runtime security or format validation."""


def validate_pdf_bytes(data: bytes, filename: str = "upload.pdf", max_size_mb: int = 100) -> None:
    """Validate raw PDF bytes for size and magic header."""
    if not data:
        raise DocumentValidationError(f"File '{filename}' is empty (0 bytes).")

    max_bytes = max_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise DocumentValidationError(
            f"File '{filename}' exceeds maximum allowed size of {max_size_mb} MB ({len(data)} bytes)."
        )

    # PDF specification requires %PDF- at or near the beginning
    prefix = data[:1024]
    if b"%PDF-" not in prefix:
        raise DocumentValidationError(
            f"File '{filename}' does not contain a valid PDF header (%PDF-)."
        )


def validate_pdf_path(path_str: str | Path, max_size_mb: int = 100) -> Path:
    """Validate a local PDF file path, ensuring security against traversal and valid format."""
    if not path_str:
        raise DocumentValidationError("No PDF path provided.")

    path = Path(path_str).expanduser()

    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as err:
        raise DocumentValidationError(f"Document not found at path: {path}") from err
    except PermissionError as err:
        raise DocumentValidationError(f"Permission denied accessing path: {path}") from err

    if not resolved.is_file():
        raise DocumentValidationError(f"Path is not a regular file: {resolved}")

    # File size validation
    size_bytes = resolved.stat().st_size
    if size_bytes == 0:
        raise DocumentValidationError(f"PDF file is empty (0 bytes): {resolved}")

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise DocumentValidationError(
            f"File exceeds maximum size limit of {max_size_mb} MB ({size_bytes} bytes): {resolved}"
        )

    # Header check
    with resolved.open("rb") as f:
        header = f.read(1024)
        if b"%PDF-" not in header:
            raise DocumentValidationError(
                f"File '{resolved.name}' is not a valid PDF document (missing %PDF- header)."
            )

    return resolved
