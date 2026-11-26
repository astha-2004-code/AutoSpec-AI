"""Initial AutoSpec AI 2.0 schema with pgvector support.

Revision ID: 0001_initial_pgvector_schema
Revises:
Create Date: 2026-11-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001_initial_pgvector_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    embedding_type = Vector(384) if is_postgres else sa.Text()

    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "vehicles",
        sa.Column("vehicle_id", sa.String(120), primary_key=True),
        sa.Column("manufacturer", sa.String(120), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("variant", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(64), primary_key=True),
        sa.Column("owner_session_id", sa.String(64), nullable=False),
        sa.Column("vehicle_id", sa.String(120), sa.ForeignKey("vehicles.vehicle_id", ondelete="SET NULL"), nullable=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("pdf_path", sa.Text(), nullable=False),
        sa.Column("pdf_data", sa.LargeBinary(), nullable=True),
        sa.Column("total_pages", sa.Integer(), nullable=False),
        sa.Column("indexed_chunks", sa.Integer(), nullable=False),
        sa.Column("spec_count", sa.Integer(), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("read_only", sa.Boolean(), nullable=False),
        sa.Column("manufacturer", sa.String(120), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("subsystems_json", sa.Text(), nullable=False),
        sa.Column("ocr_pages", sa.Integer(), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("freshness", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_owner_session_id", "documents", ["owner_session_id"])
    op.create_index("ix_documents_fingerprint", "documents", ["fingerprint"])
    op.create_index("ix_documents_session_fingerprint", "documents", ["owner_session_id", "fingerprint"])

    op.create_table(
        "document_pages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("ocr_used", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_document_pages_document_id", "document_pages", ["document_id"])
    op.create_index("ix_document_pages_doc_page", "document_pages", ["document_id", "page_number"], unique=True)

    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(128), primary_key=True),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(255), nullable=False),
        sa.Column("subsection", sa.String(255), nullable=False),
        sa.Column("subsystem", sa.String(120), nullable=False),
        sa.Column("chunk_type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("bbox_json", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])

    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.String(128), sa.ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("embedding", embedding_type, nullable=False),
    )
    op.create_index("ix_chunk_embeddings_document_id", "chunk_embeddings", ["document_id"])
    if is_postgres:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_cosine ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
        )

    op.create_table(
        "specifications",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("component", sa.String(255), nullable=False),
        sa.Column("subsystem", sa.String(120), nullable=False),
        sa.Column("spec_type", sa.String(120), nullable=False),
        sa.Column("value", sa.String(255), nullable=False),
        sa.Column("unit", sa.String(64), nullable=False),
        sa.Column("tolerance", sa.String(120), nullable=True),
        sa.Column("condition", sa.Text(), nullable=True),
        sa.Column("verification_status", sa.String(64), nullable=False),
        sa.Column("verification_reason", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("source_quote", sa.Text(), nullable=True),
        sa.Column("chunk_id", sa.String(128), nullable=True),
        sa.Column("vehicle_json", sa.Text(), nullable=False),
        sa.Column("bbox_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_specifications_document_id", "specifications", ["document_id"])
    op.create_index("ix_specifications_subsystem", "specifications", ["subsystem"])

    op.create_table(
        "citations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("specification_id", sa.Integer(), sa.ForeignKey("specifications.id", ondelete="CASCADE"), nullable=False),
        sa.Column("document_id", sa.String(64), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(128), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("bbox_json", sa.Text(), nullable=False),
    )
    op.create_index("ix_citations_specification_id", "citations", ["specification_id"])
    op.create_index("ix_citations_document_id", "citations", ["document_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("job_id", sa.String(64), primary_key=True),
        sa.Column("owner_session_id", sa.String(64), nullable=False),
        sa.Column("document_id", sa.String(64), nullable=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False),
        sa.Column("current_page", sa.Integer(), nullable=False),
        sa.Column("total_pages", sa.Integer(), nullable=False),
        sa.Column("progress_pct", sa.Float(), nullable=False),
        sa.Column("is_duplicate", sa.Boolean(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("source_pdf_path", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ingestion_jobs_owner_session_id", "ingestion_jobs", ["owner_session_id"])

    op.create_table(
        "search_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_session_id", sa.String(64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("document_ids_json", sa.Text(), nullable=False),
        sa.Column("subsystem_filter", sa.String(120), nullable=True),
        sa.Column("result_count", sa.Integer(), nullable=False),
        sa.Column("top_status", sa.String(64), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_search_history_owner_session_id", "search_history", ["owner_session_id"])

    op.create_table(
        "conversation_contexts",
        sa.Column("conversation_id", sa.String(64), primary_key=True),
        sa.Column("owner_session_id", sa.String(64), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_conversation_contexts_owner_session_id", "conversation_contexts", ["owner_session_id"])

    op.create_table(
        "evaluation_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("mode", sa.String(64), nullable=False),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("top_1_accuracy", sa.Float(), nullable=False),
        sa.Column("top_3_recall", sa.Float(), nullable=False),
        sa.Column("mrr", sa.Float(), nullable=False),
        sa.Column("exact_numeric_accuracy", sa.Float(), nullable=False),
        sa.Column("unit_accuracy", sa.Float(), nullable=False),
        sa.Column("citation_accuracy", sa.Float(), nullable=False),
        sa.Column("avg_latency_ms", sa.Float(), nullable=False),
        sa.Column("ablation_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("evaluation_runs.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("expected_value", sa.String(255), nullable=False),
        sa.Column("predicted_value", sa.String(255), nullable=False),
        sa.Column("verification_status", sa.String(64), nullable=False),
        sa.Column("is_hit_top1", sa.Boolean(), nullable=False),
        sa.Column("is_hit_top3", sa.Boolean(), nullable=False),
        sa.Column("exact_match", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
    )
    op.create_index("ix_evaluation_results_run_id", "evaluation_results", ["run_id"])


def downgrade() -> None:
    for table in (
        "evaluation_results",
        "evaluation_runs",
        "conversation_contexts",
        "search_history",
        "ingestion_jobs",
        "citations",
        "specifications",
        "chunk_embeddings",
        "document_chunks",
        "document_pages",
        "documents",
        "vehicles",
        "sessions",
    ):
        op.drop_table(table)

