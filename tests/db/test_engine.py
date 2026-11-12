import asyncio
import os

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kc.db.engine import create_engine, create_session_factory
from kc.db.settings import DbSettings


def _settings(url: str = "postgresql+psycopg://kc:s3cret@localhost:5432/kc") -> DbSettings:
    return DbSettings(_env_file=None, DATABASE_URL=SecretStr(url))  # type: ignore[call-arg]


def test_engine_is_created_lazily_with_psycopg_driver() -> None:
    # No database is needed: creating an engine does not open a connection.
    engine = create_engine(_settings())
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.url.database == "kc"
        assert "s3cret" not in str(engine.url)  # SQLAlchemy masks the password
    finally:
        asyncio.run(engine.dispose())


def test_session_factory_produces_async_sessions() -> None:
    engine = create_engine(_settings())
    try:
        factory = create_session_factory(engine)
        session = factory()
        assert isinstance(session, AsyncSession)
        assert factory.kw["expire_on_commit"] is False
    finally:
        asyncio.run(engine.dispose())


@pytest.mark.integration
def test_can_query_real_database() -> None:
    """Needs KC_INTEGRATION=1 and DATABASE_URL pointing at a running Postgres."""

    async def run() -> int:
        engine = create_engine(_settings(os.environ["DATABASE_URL"]))
        try:
            async with create_session_factory(engine)() as session:
                return int((await session.execute(text("SELECT 1"))).scalar_one())
        finally:
            await engine.dispose()

    # Selector loop: psycopg async does not support Windows' default Proactor loop.
    assert asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop) == 1
