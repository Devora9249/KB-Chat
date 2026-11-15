import asyncio
from typing import Any

import pytest
from db_test_support import require_database_url
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kc.db import engine as engine_module
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


def test_pool_settings_are_passed_to_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_async_engine(url: str, **kwargs: Any) -> str:
        captured.update(kwargs, url=url)
        return "engine"

    monkeypatch.setattr(engine_module, "create_async_engine", fake_create_async_engine)
    settings = DbSettings(  # type: ignore[call-arg]
        _env_file=None,
        DATABASE_URL=SecretStr("postgresql+psycopg://kc:kc@localhost/kc"),
        DB_POOL_SIZE=7,
        DB_MAX_OVERFLOW=2,
        DB_POOL_TIMEOUT=4.5,
    )

    create_engine(settings)

    assert (captured["pool_size"], captured["max_overflow"], captured["pool_timeout"]) == (
        7,
        2,
        4.5,
    )
    assert captured["pool_pre_ping"] is True


def test_default_engine_pool_matches_sqlalchemy_defaults() -> None:
    # Real (lazy) engine, no database needed: unset settings keep SQLAlchemy's behaviour.
    engine = create_engine(_settings())
    try:
        assert engine.pool.size() == 5  # type: ignore[attr-defined]
        assert engine.pool.timeout() == 30.0  # type: ignore[attr-defined]
        assert engine.pool._max_overflow == 10  # type: ignore[attr-defined]
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
    """Runs when DATABASE_URL is set (CI). Read-only, so no test-database guard needed."""
    url = require_database_url()

    async def run() -> int:
        engine = create_engine(_settings(url))
        try:
            async with create_session_factory(engine)() as session:
                return int((await session.execute(text("SELECT 1"))).scalar_one())
        finally:
            await engine.dispose()

    # Selector loop: psycopg async does not support Windows' default Proactor loop.
    assert asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop) == 1
