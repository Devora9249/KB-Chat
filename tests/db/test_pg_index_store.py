"""PgIndexStore-specific tests (the contract itself is in tests/indexing/contract/).

- Fast tests (no database): error wrapping with a deliberately failing session; invalid
  input, a dimension mismatch or a word-less text query never touch the database.
- Integration tests (CI): verify_embedding_config against the real embedding_config
  table; the vector query uses the HNSW index; a narrow filter still returns k results
  thanks to hnsw.iterative_scan (and loses results without it).
"""

import asyncio
import math
import random
from collections.abc import Callable, Coroutine
from typing import Any

import psycopg
import pytest
from db_test_support import require_database_url, require_test_database
from sqlalchemy import delete, event, text
from sqlalchemy import exc as sa_exc
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import NullPool
from sqlalchemy.sql.expression import ClauseElement, Executable

from contracts.retrieval import Filters
from contracts.storage import IndexStoreError
from kc.db.embedding_config import EmbeddingConfigMismatchError
from kc.db.engine import create_session_factory
from kc.db.models import EMBEDDING_DIMENSION, EmbeddingConfig
from kc.db.pg_index_store import (
    PgIndexStore,
    apply_settings_statement,
    build_vector_search_statement,
    ef_search_for,
    vector_search_settings,
)
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
    "vector_search": lambda s: s.vector_search(_vector(), k=5),
    "text_search": lambda s: s.text_search("configure SAML", k=5),
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


# --- D12: chunk id owned by another document (detected by the database) --------------------------


class _Diag:
    def __init__(self, constraint_name: str, message_detail: str) -> None:
        self.constraint_name = constraint_name
        self.message_detail = message_detail


class _DbError(Exception):
    """Stands in for psycopg.errors.UniqueViolation (carries .diag like psycopg does)."""

    def __init__(self, constraint_name: str) -> None:
        super().__init__("duplicate key value violates unique constraint")
        self.diag = _Diag(constraint_name, "Key (id)=(c1) already exists.")


def test_chunk_primary_key_violation_becomes_value_error_with_cause() -> None:
    error = sa_exc.IntegrityError("INSERT INTO chunks ...", {}, _DbError("pk_chunks"))
    store, _ = _store(error)
    with pytest.raises(ValueError, match=r"already used by another document.*\(c1\)") as info:
        run(store.upsert_document(_doc(), [_chunk()], [_vector()]))
    assert info.value.__cause__ is error


def test_other_integrity_errors_stay_infrastructure_errors() -> None:
    error = sa_exc.IntegrityError("INSERT ...", {}, _DbError("some_other_constraint"))
    store, _ = _store(error)
    with pytest.raises(IndexStoreError) as info:
        run(store.upsert_document(_doc(), [_chunk()], [_vector()]))
    assert info.value.__cause__ is error


# --- checks before any database access ----------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        ([_chunk("a"), _chunk("b")], [_vector()], "chunks but"),
        ([_chunk("a")], [[1.0, 0.0, 0.0]], "dimension"),
        ([_chunk("a", url="OTHER")], [_vector()], "url"),
        ([_chunk("a"), _chunk("a")], [_vector(), _vector()], "duplicate chunk ids"),
    ],
    ids=["length-mismatch", "wrong-dimension", "wrong-document-url", "duplicate-id"],
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


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda s: s.vector_search(_vector(), k=0), "k must be positive"),
        (lambda s: s.vector_search(_vector(), k=-1), "k must be positive"),
        (lambda s: s.vector_search([1.0, 0.0, 0.0], k=5), "dimension"),
        (lambda s: s.text_search("saml", k=0), "k must be positive"),
    ],
    ids=["vector-k-0", "vector-k-negative", "vector-wrong-dimension", "text-k-0"],
)
def test_invalid_search_raises_value_error_without_touching_the_database(
    call: Callable[[PgIndexStore], Coroutine[Any, Any, Any]], message: str
) -> None:
    store, factory = _store(AssertionError("the database must not be touched"))
    with pytest.raises(ValueError, match=message):
        run(call(store))
    assert factory.sessions_opened == 0


@pytest.mark.parametrize("query", ["", "   ", "!!! ???", "-- ;"])
def test_text_search_without_words_returns_empty_without_touching_the_database(
    query: str,
) -> None:
    store, factory = _store(AssertionError("the database must not be touched"))
    assert run(store.text_search(query)) == []
    assert factory.sessions_opened == 0


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, 40), (5, 40), (30, 40), (31, 41), (100, 110), (995, 1000), (5000, 1000)],
)
def test_ef_search_follows_k_within_pgvector_limits(k: int, expected: int) -> None:
    assert ef_search_for(k) == expected


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


# --- integration: HNSW index usage and iterative scan --------------------------------------------


def _axis(axis: int) -> list[float]:
    """Unit vector along `axis` (for tests that only need SOME valid vector)."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[axis] = 1.0
    return vector


def _unit(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector]


def _narrow_filter_vectors() -> tuple[list[float], list[list[float]], list[list[float]]]:
    """(query, 200 orchestration vectors, 5 sequence vectors), deterministic (fixed seed).

    Realistic, NOT degenerate: every vector is different. Each one is the query plus random
    noise of a given size (noise components ~ N(0, sigma^2 / dim), so the noise norm is
    about sigma and cos(vector, query) is about 1 / sqrt(1 + sigma^2)):
    - orchestration: sigma 0.3 -> cos ~ 0.96 (close to the query, all distinct)
    - sequence:      sigma 1.3 -> cos ~ 0.61 (clearly farther than ANY orchestration one)
    So the 40 nearest HNSW candidates are all orchestration (a sequence filter rejects
    them), while distances are graded like real embeddings and the graph stays connected.
    Earlier data (200 near-identical vectors plus 5 near-identical ones exactly orthogonal
    to them) built a degenerate HNSW graph in which seq-0 was unreachable.
    """
    rng = random.Random(1234)
    scale = 1 / math.sqrt(EMBEDDING_DIMENSION)
    query = _unit([rng.gauss(0, 1) for _ in range(EMBEDDING_DIMENSION)])

    def around(sigma: float) -> list[float]:
        return _unit([q + rng.gauss(0, sigma * scale) for q in query])

    return query, [around(0.3) for _ in range(200)], [around(1.3) for _ in range(5)]


NARROW_QUERY, ORCHESTRATION_VECTORS, SEQUENCE_VECTORS = _narrow_filter_vectors()


def _engine_that_must_use_indexes(url: str) -> AsyncEngine:
    """NullPool engine whose connections disable sequential scans.

    On a small test table Postgres rightly prefers a sequential scan; disabling it (for
    THESE connections only) shows whether the query CAN be answered by the HNSW index,
    i.e. that operator (<=>) and operator class (vector_cosine_ops) match.
    """
    engine = create_async_engine(url, poolclass=NullPool)

    @event.listens_for(engine.sync_engine, "connect")
    def _no_seqscan(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("SET enable_seqscan = off")
        cursor.close()

    return engine


class Explain(Executable, ClauseElement):
    """`EXPLAIN <statement>` as a SQLAlchemy construct (the @compiles recipe).

    Executed like any statement, so SQLAlchemy binds the parameters itself: expanding IN
    lists, pgvector's Vector type, ... - no hand-made parameter handling.
    """

    inherit_cache = False

    def __init__(self, statement: Any) -> None:
        self.statement = statement


@compiles(Explain, "postgresql")
def _compile_explain(element: Explain, compiler: Any, **kw: Any) -> str:
    return "EXPLAIN " + compiler.process(element.statement, **kw)


async def _plan(connection: AsyncConnection, statement: Any) -> str:
    result = await connection.execute(Explain(statement))
    return "\n".join(row[0] for row in result.all())


async def _load_narrow_filter_data(store: PgIndexStore) -> None:
    """200 orchestration chunks close to NARROW_QUERY, 5 sequence chunks clearly farther."""
    orchestration = [
        Chunk(
            id=f"orch-{i:03d}",
            document_url="test://orch",
            text="saml",
            heading_path=[],
            product="orchestration",
            version="10",
            doc_type="how-to",
        )
        for i in range(200)
    ]
    await store.upsert_document(
        _doc("test://orch").model_copy(update={"product": "orchestration"}),
        orchestration,
        ORCHESTRATION_VECTORS,
    )
    sequence = [
        Chunk(
            id=f"seq-{i}",
            document_url="test://seq",
            text="saml",
            heading_path=[],
            product="sequence",
            version="9",
            doc_type="how-to",
        )
        for i in range(5)
    ]
    await store.upsert_document(_doc("test://seq"), sequence, SEQUENCE_VECTORS)


async def _truncate(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE documents, chunks"))


def _assert_all_five_sequence_chunks_best_first(results: list[tuple[str, float]]) -> None:
    """k=5 results, every one matching the filter (so: all 5 sequence chunks), best first."""
    ids = [chunk_id for chunk_id, _ in results]
    scores = [score for _, score in results]
    assert sorted(ids) == [f"seq-{i}" for i in range(5)], results
    assert scores == sorted(scores, reverse=True), results


@pytest.mark.integration
def test_vector_search_query_uses_the_hnsw_index() -> None:
    """DESTRUCTIVE (writes and empties documents/chunks): test databases only."""
    url = require_database_url()
    require_test_database(url)

    async def scenario() -> str:
        engine = _engine_that_must_use_indexes(url)
        try:
            await _truncate(engine)
            await _load_narrow_filter_data(PgIndexStore(create_session_factory(engine)))
            async with engine.connect() as connection, connection.begin():
                await connection.execute(apply_settings_statement(vector_search_settings(5)))
                return await _plan(connection, build_vector_search_statement(NARROW_QUERY, None, 5))
        finally:
            await _truncate(engine)
            await engine.dispose()

    plan = run(scenario())
    assert "ix_chunks_embedding_hnsw" in plan, plan
    assert "Seq Scan on chunks" not in plan, plan


@pytest.mark.integration
def test_narrow_filter_returns_k_through_vector_search() -> None:
    """DESTRUCTIVE (writes and empties documents/chunks): test databases only.

    The user-facing guarantee: 5 of 205 chunks match the filter, and vector_search returns
    exactly those 5, whatever plan Postgres picks (HNSW + iterative scan, or the B-tree
    index on product followed by an exact sort).
    """
    url = require_database_url()
    require_test_database(url)

    async def scenario() -> list[tuple[str, float]]:
        engine = create_async_engine(url, poolclass=NullPool)
        store = PgIndexStore(create_session_factory(engine))
        try:
            await _truncate(engine)
            await _load_narrow_filter_data(store)
            results = await store.vector_search(NARROW_QUERY, Filters(products=["sequence"]), k=5)
            return [(r.chunk_id, r.score) for r in results]
        finally:
            await _truncate(engine)
            await engine.dispose()

    _assert_all_five_sequence_chunks_best_first(run(scenario()))


@pytest.mark.integration
def test_iterative_scan_is_what_keeps_k_results_when_hnsw_is_the_only_path() -> None:
    """DESTRUCTIVE (writes and empties documents/chunks): test databases only.

    For a narrow filter Postgres may rightly answer from the B-tree index on product
    (exact, no HNSW involved). To test iterative_scan itself, this transaction temporarily
    DROPs that B-tree index (and sequential scans are off), so HNSW is the only path. DDL in
    Postgres is transactional: the transaction ends with ROLLBACK and the index is back.

    The 40 HNSW candidates nearest to the query are all orchestration: without iterative
    scan the sequence filter loses results; with vector_search's settings it returns all 5.
    Checks what the contract promises (k results, all matching, best first), not one exact
    id order, because HNSW is approximate.
    """
    url = require_database_url()
    require_test_database(url)
    narrow = Filters(products=["sequence"])
    statement = build_vector_search_statement(NARROW_QUERY, narrow, 5)
    without_iterative = {**vector_search_settings(5), "hnsw.iterative_scan": "off"}

    async def scenario() -> tuple[list[tuple[str, float]], int, str]:
        engine = _engine_that_must_use_indexes(url)
        try:
            await _truncate(engine)
            await _load_narrow_filter_data(PgIndexStore(create_session_factory(engine)))
            async with engine.connect() as connection:
                transaction = await connection.begin()
                try:
                    await connection.execute(text("DROP INDEX ix_chunks_product"))

                    await connection.execute(apply_settings_statement(without_iterative))
                    count_without = len((await connection.execute(statement)).all())

                    await connection.execute(apply_settings_statement(vector_search_settings(5)))
                    rows = (await connection.execute(statement)).all()
                    with_iterative = [(row.id, float(row.score)) for row in rows]
                    plan = await _plan(connection, statement)
                finally:
                    await transaction.rollback()  # the B-tree index comes back
            return with_iterative, count_without, plan
        finally:
            await _truncate(engine)
            await engine.dispose()

    with_iterative, count_without, plan = run(scenario())

    assert "ix_chunks_embedding_hnsw" in plan, plan  # HNSW really answered the query
    assert count_without < 5, "without iterative_scan the narrow filter should lose results"
    _assert_all_five_sequence_chunks_best_first(with_iterative)


@pytest.mark.integration
def test_b_tree_index_on_product_still_exists_after_the_rollback() -> None:
    """Guards the previous test: its DROP INDEX must have been rolled back."""
    url = require_database_url()

    async def scenario() -> list[tuple[Any, ...]]:
        engine = create_async_engine(url, poolclass=NullPool)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(
                    text("SELECT indexname FROM pg_indexes WHERE indexname = 'ix_chunks_product'")
                )
                return [tuple(row) for row in result.all()]
        finally:
            await engine.dispose()

    assert run(scenario()) == [("ix_chunks_product",)]
