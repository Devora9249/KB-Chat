"""SQLAlchemy ORM models owned by Storage & Indexing.

They mirror the migrations (0002 embedding_config, 0003 documents/chunks, 0004 embedding);
the migrations are the source of truth for the schema, these models are how code reads
and writes it.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Dimension of chunks.embedding (migration 0004, BAAI/bge-m3). Model-dependent.
EMBEDDING_DIMENSION = 1024

# Deterministic constraint/index names, so migrations are stable and reviewable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class EmbeddingConfig(Base):
    """Which embedding model built the vectors in this database (exactly one row).

    Vectors from different models (or dimensions) are not comparable, so before indexing
    we check this row; a mismatch means a full re-index is required.
    """

    __tablename__ = "embedding_config"
    __table_args__ = (CheckConstraint("id = 1", name="single_row"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False, default=1)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentRow(Base):
    """One indexed page (migration 0003)."""

    __tablename__ = "documents"

    url: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    breadcrumb: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    product: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_type: Mapped[str] = mapped_column(Text, nullable=False)
    content_md: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ChunkRow(Base):
    """One chunk with its vector (migrations 0003 + 0004). `tsv` is computed by Postgres."""

    __tablename__ = "chunks"

    # Chunk.id from the Chunker (D12, closed): 16 hex chars,
    # sha256(url | heading_path | part)[:16]. It already includes the URL, so it is unique
    # across the whole index. The primary key is only a safety net: a collision or a
    # Chunker bug fails the upsert (mapped to ValueError) instead of silently overwriting.
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_url: Mapped[str] = mapped_column(
        Text, ForeignKey("documents.url", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    product: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_type: Mapped[str] = mapped_column(Text, nullable=False)
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple'::regconfig, text)", persisted=True)
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
