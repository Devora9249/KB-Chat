"""PgIndexStore-specific tests (the write contract itself is in tests/indexing/contract/).

- Fast tests (no database): error wrapping with a deliberately failing session, and that
  invalid input / a dimension mismatch is rejected BEFORE any database access.
- Integration test (CI): verify_embedding_config against the real embedding_config table.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import psycopg
import pytest
from db_test_support import require_database_url, require_test_database
from sqlalchemy import delete
from sqlalchemy import exc as sa_exc
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from contracts.storage import IndexStoreError
from kc.db.embedding_config import EmbeddingConfigMismatchError
from kc.db.engine import create_session_factory
from kc.db.models import EMBEDDING_DIMENSION, EmbeddingConfig
from kc.db.pg_index_store import PgIndexStore
from kc.indexing._provisional_contracts import Chunk, Document

MODEL = "BAAI/bge-m3@5617a9f"


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)


def _doc(url: str = "u1") -> Document:
    return Document(
        url=url,
        title="T",
        breadcrumb=[],
        product="sequence",
        version="9",
        doc_type="how-to",
        content_md="...",
        content_hash="h",
    )


def _chunk(chunk_id: str = "c1", url: str = "u1") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_url=url,
        text="t",
        heading_path=[],
        product="sequence",
        version="9",
        doc_type="how-to",
    )


def _vector() -> list[float]:
    return [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


# --- a session factory that fails on purpose ----------------------------------------------------


class _Transaction:
    async def __aenter__(self) -> "_Transaction":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class FailingSession:
    def __init__(self, error: BaseException, calls: list[str]) -> None:
        self.error = error
        self.calls = calls

    async def __aenter__(self) -> "FailingSession":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def begin(self) -> _Transaction:
        return _Transaction()

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("execute")
        raise self.error

    async def scalar(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("scalar")
        raise self.error


class FailingFactory:
    """Stands in for async_sessionmaker: every database call raises `error`."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls: list[str] = []
        self.sessions_opened = 0

    def __call__(self) -> FailingSession:
        self.sessions_opened += 1
        return FailingSession(self.error, self.calls)


def _store(error: BaseException) -> tuple[PgIndexStore, FailingFactory]:
    factory = FailingFactory(error)
    return PgIndexStore(factory), factory  # type: ignore[arg-type]


OPERATIONS: dict[str, Callable[[PgIndexStore], Coroutine[Any, Any, Any]]] = {
    "upsert_document": lambda s: s.upsert_document(_doc(), [_chunk()], [_vector()]),
    "delete_document": lambda s: s.delete_document("u1"),
    "get_document_hashes": lambda s: s.get_document_hashes(),
    "verify_embedding_config": lambda s: s.verify_embedding_config(MODEL, EMBEDDING_DIMENSION),
}

INFRASTRUCTURE_ERRORS = {
    "sqlalchemy.OperationalError": sa_exc.OperationalError("SELECT 1", {}, Exception("down")),
    "sqlalchemy.InterfaceError": sa_exc.InterfaceError("SELECT 1", {}, Exception("closed")),
    "sqlalchemy.DBAPIError": sa_exc.DBAPIError("SELECT 1", {}, Exception("db error")),
    "sqlalchemy.pool.TimeoutError": sa_exc.TimeoutError("no free connection"),
    "psycopg.OperationalError": psycopg.OperationalError("connection refused"),
    "psycopg.InterfaceError": psycopg.InterfaceError("connection closed"),
}


# --- error wrapping -----------------------------------------------------------------------------


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
@pytest.mark.parametrize("error_name", sorted(INFRASTRUCTURE_ERRORS))
def test_infrastructure_errors_are_wrapped_with_cause(operation: str, error_name: str) -> None:
    error = INFRASTRUCTURE_ERRORS[error_name]
    store, _ = _store(error)
    with pytest.raises(IndexStoreError, match=operation) as exc_info:
        run(OPERATIONS[operation](store))
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
@pytest.mark.parametrize(
    "error",
    [ValueError("from the database layer"), TypeError("a bug")],
    ids=["ValueError", "TypeError"],
)
def test_other_errors_are_not_wrapped(operation: str, error: Exception) -> None:
    # No generic `except Exception`: anything that is not an infrastructure failure
    # surfaces as itself.
    store, _ = _store(error)
    with pytest.raises(type(error)):
        run(OPERATIONS[operation](store))


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_cancellation_is_never_wrapped(operation: str) -> None:
    store, _ = _store(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run(OPERATIONS[operation](store))


# --- checks before any database access ----------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        ([_chunk("a"), _chunk("b")], [_vector()], "chunks but"),
        ([_chunk("a")], [[1.0, 0.0, 0.0]], "dimension"),
        ([_chunk("a", url="OTHER")], [_vector()], "url"),
    ],
    ids=["length-mismatch", "wrong-dimension", "wrong-document-url"],
)
def test_invalid_upsert_raises_value_error_without_touching_the_database(
    chunks: list[Chunk], vectors: list[list[float]], message: str
) -> None:
    store, factory = _store(AssertionError("the database must not be touched"))
    with pytest.raises(ValueError, match=message):
        run(store.upsert_document(_doc(), chunks, vectors))
    assert factory.sessions_opened == 0


def test_verify_rejects_a_dimension_that_does_not_fit_the_column_before_the_database() -> None:
    store, factory = _store(AssertionError("the database must not be touched"))
    with pytest.raises(EmbeddingConfigMismatchError, match="re-index"):
        run(store.verify_embedding_config("some-model", 512))
    assert factory.sessions_opened == 0


def test_constructor_rejects_a_dimension_other_than_the_column() -> None:
    with pytest.raises(ValueError, match=r"vector\(1024\)"):
        PgIndexStore(FailingFactory(AssertionError()), dimension=512)  # type: ignore[arg-type]


# --- integration: model-change guard against the real table -------------------------------------


@pytest.mark.integration
def test_verify_embedding_config_records_then_guards_against_a_model_change() -> None:
    """DESTRUCTIVE (clears embedding_config): only on a database whose name contains "test"."""
    url = require_database_url()
    require_test_database(url)

    async def scenario() -> None:
        engine = create_async_engine(url, poolclass=NullPool)
        factory = create_session_factory(engine)
        store = PgIndexStore(factory)
        try:
            async with factory() as session, session.begin():
                await session.execute(delete(EmbeddingConfig))

            await store.verify_embedding_config(MODEL, EMBEDDING_DIMENSION)  # first: records
            await store.verify_embedding_config(MODEL, EMBEDDING_DIMENSION)  # same: ok

            with pytest.raises(EmbeddingConfigMismatchError, match="re-index"):
                await store.verify_embedding_config("BAAI/bge-m3@0000000", EMBEDDING_DIMENSION)

            async with factory() as session:
                row = await session.get(EmbeddingConfig, 1)
                assert row is not None
                assert (row.model_name, row.dimension) == (MODEL, EMBEDDING_DIMENSION)
        finally:
            async with factory() as session, session.begin():
                await session.execute(delete(EmbeddingConfig))
            await engine.dispose()

    run(scenario())
