"""Guard against mixing vectors from different embedding models in one index.

`decide()` is pure (unit-tested without a DB); `ensure_embedding_config()` applies it
to the database.
"""

from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kc.db.models import EmbeddingConfig


class EmbeddingConfigMismatchError(RuntimeError):
    """The database was indexed with a different embedding model or dimension."""


class Decision(StrEnum):
    INITIALIZE = "initialize"  # empty database: record the current model
    OK = "ok"  # same model and dimension as the existing index


def decide(stored: tuple[str, int] | None, model_name: str, dimension: int) -> Decision:
    """Compare the stored (model_name, dimension) with the configured one.

    Raises EmbeddingConfigMismatchError if they differ.
    """
    if not model_name:
        raise ValueError("model_name must not be empty")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if stored is None:
        return Decision.INITIALIZE
    if stored == (model_name, dimension):
        return Decision.OK
    stored_model, stored_dim = stored
    raise EmbeddingConfigMismatchError(
        f"index was built with {stored_model!r} (dim={stored_dim}) but the configured "
        f"embedder is {model_name!r} (dim={dimension}); a full re-index is required "
        "(coordinate with Evaluation & Quality before changing the embedding model)"
    )


async def ensure_embedding_config(session: AsyncSession, model_name: str, dimension: int) -> None:
    """Record the embedding model on first use; fail if it differs from the stored one.

    Commits only when it inserts the first row.
    """
    row = await session.scalar(select(EmbeddingConfig).where(EmbeddingConfig.id == 1))
    stored = (row.model_name, row.dimension) if row is not None else None
    if decide(stored, model_name, dimension) is Decision.INITIALIZE:
        session.add(EmbeddingConfig(id=1, model_name=model_name, dimension=dimension))
        await session.commit()
