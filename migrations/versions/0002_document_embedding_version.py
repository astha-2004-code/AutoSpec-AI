"""Track embedding model version and dimension per document for safe reindexing.

Revision ID: 0002_document_embedding_version
Revises: 0001_initial_pgvector_schema
Create Date: 2026-11-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_document_embedding_version"
down_revision = "0001_initial_pgvector_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("embedding_model", sa.String(120), nullable=False, server_default=""),
    )
    op.add_column(
        "documents",
        sa.Column("embedding_dim", sa.Integer(), nullable=False, server_default="384"),
    )


def downgrade() -> None:
    op.drop_column("documents", "embedding_dim")
    op.drop_column("documents", "embedding_model")
