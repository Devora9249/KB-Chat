"""Add documents and chunks (without the embedding vector, see 0004).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-15

- documents: one row per page; content_hash drives change detection.
- chunks: one row per chunk. product/version/doc_type are copied from the document so
  filtering needs no JOIN. Deleting a document deletes its chunks (ON DELETE CASCADE),
  which also makes "replace all chunks of a document" a delete + insert in one transaction.
- chunks.id is the primary key, unique across the WHOLE index (D12 working assumption):
  Search & Ranking fuses results by chunk_id (RRF), so equal ids from different documents
  would be merged by mistake.
- chunks.tsv is computed by Postgres from `text` with the 'simple' configuration (no
  stemming, no stop words), so exact technical names such as
  Set-CoraSeQuenceSiteAuthentication stay searchable; GIN index for full-text search.
- No CHECK constraints on product / version / doc_type on purpose: the allowed values are
  validated by Pydantic (contracts.common Literals), and DocType may grow after the full
  crawl, which would otherwise need a migration per new value.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "breadcrumb",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("product", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column("doc_type", sa.Text(), nullable=False),
        sa.Column("content_md", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("url", name=op.f("pk_documents")),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("document_url", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "heading_path",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("product", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column("doc_type", sa.Text(), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple'::regconfig, text)", persisted=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.ForeignKeyConstraint(
            ["document_url"],
            ["documents.url"],
            name=op.f("fk_chunks_document_url_documents"),
            ondelete="CASCADE",
        ),
    )

    # Replace / delete all chunks of a document quickly (also used by the CASCADE).
    op.create_index("ix_chunks_document_url", "chunks", ["document_url"])
    # Filters (contracts.retrieval.Filters): products, versions, doc_types / exclude_doc_types.
    op.create_index("ix_chunks_product", "chunks", ["product"])
    op.create_index("ix_chunks_version", "chunks", ["version"])
    op.create_index("ix_chunks_doc_type", "chunks", ["doc_type"])
    # Full-text search (text_search).
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_index("ix_chunks_doc_type", table_name="chunks")
    op.drop_index("ix_chunks_version", table_name="chunks")
    op.drop_index("ix_chunks_product", table_name="chunks")
    op.drop_index("ix_chunks_document_url", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
