"""Add embedding_config: which embedding model built the index (single row).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_config",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name=op.f("ck_embedding_config_single_row")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_embedding_config")),
    )


def downgrade() -> None:
    op.drop_table("embedding_config")
