import asyncio

import pytest
from db_test_support import require_database_url, require_test_database
from sqlalchemy import delete

from kc.db.embedding_config import (
    Decision,
    EmbeddingConfigMismatchError,
    decide,
    ensure_embedding_config,
)
from kc.db.engine import create_engine, create_session_factory
from kc.db.models import EmbeddingConfig
from kc.db.settings import db_settings_from_url


def test_empty_database_is_initialized() -> None:
    assert decide(None, "voyage-3", 1024) is Decision.INITIALIZE


def test_same_model_and_dimension_is_ok() -> None:
    assert decide(("voyage-3", 1024), "voyage-3", 1024) is Decision.OK


@pytest.mark.parametrize(
    ("stored", "model", "dim"),
    [
        (("voyage-3", 1024), "bge-m3", 1024),  # different model, same dim: still not comparable
        (("voyage-3", 1024), "voyage-3", 512),  # same model, different dim
    ],
)
def test_mismatch_requires_reindex(stored: tuple[str, int], model: str, dim: int) -> None:
    with pytest.raises(EmbeddingConfigMismatchError, match="re-index"):
        decide(stored, model, dim)


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        decide(None, "", 1024)
    with pytest.raises(ValueError):
        decide(None, "voyage-3", 0)


@pytest.mark.integration
def test_ensure_embedding_config_on_real_database() -> None:
    """DESTRUCTIVE (deletes embedding_config rows): runs only on a database whose name
    contains "test". Needs migrations applied (CI runs `alembic upgrade head` first)."""
    url = require_database_url()
    require_test_database(url)

    async def run() -> None:
        engine = create_engine(db_settings_from_url(url))
        factory = create_session_factory(engine)
        try:
            async with factory() as session:
                await session.execute(delete(EmbeddingConfig))
                await session.commit()

                await ensure_embedding_config(session, "fake", 256)  # first run: inserts
                await ensure_embedding_config(session, "fake", 256)  # same config: ok
                with pytest.raises(EmbeddingConfigMismatchError):
                    await ensure_embedding_config(session, "other-model", 256)

                await session.execute(delete(EmbeddingConfig))
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)
