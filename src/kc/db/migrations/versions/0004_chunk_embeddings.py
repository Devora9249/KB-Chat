"""Add chunks.embedding vector(1024) and its HNSW index.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-15

MODEL-DEPENDENT: 1024 is the dimension of BAAI/bge-m3 (see kc.indexing.bge_m3_embedder).
Switching to a model with a different dimension requires a NEW migration (change the
column type, rebuild this index) AND a full re-index; the embedding_config guard refuses
to index with a mismatching model until then. Everything model-independent is in 0003.

- vector_cosine_ops: cosine distance (`<=>`), matching the Embedder contract (unit-length
  vectors; similarity = 1 - distance).
- HNSW with pgvector's default build parameters (m=16, ef_construction=64); tuning later,
  with Search & Ranking, on real data.
- NOT NULL is safe here because 0003 creates `chunks` empty; a chunk without a vector
  cannot be searched by vector_search.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSION = 1024


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=False))
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_column("chunks", "embedding")
