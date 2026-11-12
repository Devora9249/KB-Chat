"""SQLAlchemy ORM models owned by Storage & Indexing.

Only tables that do NOT depend on the shared contracts live here for now.
`documents` / `chunks` are added after `Document`, `Chunk` and `Filters` are agreed.
"""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, MetaData, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
