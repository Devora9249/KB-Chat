"""Fixtures for the storage contract test suites (owner: Raizy).

- `make_search_index`: a SearchIndex already loaded with given documents. Search tests use
  ONLY vector_search / text_search on it (contracts.storage.SearchIndex).
- `store` / `make_store`: a fresh, empty full IndexStore (search + write), e.g. for the
  concurrency tests. Registry: IMPLEMENTATIONS.
- `writer`: a fresh, empty IndexWriter plus an observer that reads back what it stored
  WITHOUT the search side. Write tests use it. Registry: WRITER_IMPLEMENTATIONS.
- `embedder`: one per entry in EMBEDDER_IMPLEMENTATIONS (contracts.storage.Embedder).

Every test runs once per registered implementation. Implementations that need Postgres
are marked `integration`: they run when DATABASE_URL is set (CI) and only against a
database whose name contains "test" (tables are emptied between tests).
Other teams are welcome to add contract tests via PR.
"""

import re
from collections.abc import Callable, Iterator

import pytest
from index_store_helpers import (
    DIM,
    IndexedDocument,
    SearchIndexFactory,
    StoredChunk,
    StoreFactory,
    WriterUnderTest,
    run,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from contracts.storage import Embedder, SearchIndex
from kc.db.engine import create_session_factory
from kc.db.models import EMBEDDING_DIMENSION
from kc.db.pg_index_store import PgIndexStore
from kc.indexing._provisional_contracts import IndexStore
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from tests.db.db_test_support import (
    SKIP_REASON,
    database_url,
    migrate_if_test_database,
    require_test_database,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs a real Postgres + pgvector; runs only when DATABASE_URL is set",
    )


# --- full IndexStore (search + write) ----------------------------------------------------------


def _in_memory(dimension: int) -> IndexStore:
    return InMemoryIndexStore(dimension=dimension, model_name="contract-tests")


# TODO(PgIndexStore search side, step 4): add "pg" here too.
IMPLEMENTATIONS: dict[str, StoreFactory] = {
    "in_memory": _in_memory,
}


@pytest.fixture(params=sorted(IMPLEMENTATIONS))
def make_store(request: pytest.FixtureRequest) -> StoreFactory:
    """Factory for a fresh, empty store of the current implementation (any dimension)."""
    factory: StoreFactory = IMPLEMENTATIONS[request.param]
    return factory


@pytest.fixture
def store(make_store: StoreFactory) -> IndexStore:
    """A fresh, empty store with dimension DIM (3)."""
    return make_store(DIM)


@pytest.fixture
def make_search_index(make_store: StoreFactory) -> SearchIndexFactory:
    """Factory for a SearchIndex that already contains the given documents.

    Loading goes through the implementation's own writer; the returned object is typed
    as SearchIndex so tests can only search it.
    """

    def factory(documents: list[IndexedDocument], dimension: int = DIM) -> SearchIndex:
        store = make_store(dimension)
        for document, chunks, vectors in documents:
            run(store.upsert_document(document, chunks, vectors))
        return store

    return factory


# --- IndexWriter (write side) ------------------------------------------------------------------


def _in_memory_writer() -> WriterUnderTest:
    store = InMemoryIndexStore(dimension=DIM, model_name="contract-tests")

    def stored() -> list[StoredChunk]:
        # Test-only peek at the internal state, so write tests do not depend on search.
        return sorted(
            (chunk.id, chunk.document_url, chunk.text, vector)
            for entry in store._documents.values()
            for chunk, vector in entry.entries
        )

    return WriterUnderTest(writer=store, dimension=DIM, stored=stored)


WRITER_IMPLEMENTATIONS: dict[str, Callable[[pytest.FixtureRequest], WriterUnderTest]] = {
    "in_memory": lambda request: _in_memory_writer(),
    "pg": lambda request: request.getfixturevalue("pg_writer"),
}
INTEGRATION_WRITERS = {"pg"}  # need Postgres: run only with DATABASE_URL (CI)


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.integration) if name in INTEGRATION_WRITERS else name
        for name in sorted(WRITER_IMPLEMENTATIONS)
    ]
)
def writer(request: pytest.FixtureRequest) -> WriterUnderTest:
    """A fresh, empty IndexWriter of the current implementation, with its observer."""
    return WRITER_IMPLEMENTATIONS[request.param](request)


# --- Postgres -----------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_test_database_url() -> str:
    """DATABASE_URL of a TEST database (name contains "test"), migrated to head once."""
    url = database_url()
    if url is None:
        pytest.skip(SKIP_REASON)
    require_test_database(url)  # tests empty the tables: never on a shared database
    migrate_if_test_database(url)
    return url


async def _truncate(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE documents, chunks"))


def _parse_vector(value: str) -> tuple[float, ...]:
    return tuple(float(x) for x in re.findall(r"-?[\d.e+-]+", value))


async def _stored_chunks(engine: AsyncEngine) -> list[StoredChunk]:
    async with engine.connect() as connection:
        rows = await connection.execute(
            text("SELECT id, document_url, text, embedding::text FROM chunks ORDER BY id")
        )
        return [(i, url, body, _parse_vector(vector)) for i, url, body, vector in rows.all()]


@pytest.fixture
def pg_writer(pg_test_database_url: str) -> Iterator[WriterUnderTest]:
    """PgIndexStore on emptied tables.

    NullPool: the contract tests call asyncio.run() per operation, i.e. a new event loop
    each time, and pooled async connections cannot move between event loops. Production
    uses the normal pool from DbSettings.
    """
    engine = create_async_engine(pg_test_database_url, poolclass=NullPool)
    run(_truncate(engine))
    store = PgIndexStore(create_session_factory(engine))
    try:
        yield WriterUnderTest(
            writer=store,
            dimension=EMBEDDING_DIMENSION,
            stored=lambda: run(_stored_chunks(engine)),
        )
    finally:
        run(_truncate(engine))
        run(engine.dispose())


# --- Embedder -----------------------------------------------------------------------------------

# TODO(BgeM3Embedder): add "bge_m3" here, marked real_model (runs only with KC_REAL_EMBEDDER=1).
EMBEDDER_IMPLEMENTATIONS: dict[str, Callable[[], Embedder]] = {
    "fake": lambda: FakeEmbedder(dimension=64),
}


@pytest.fixture(params=sorted(EMBEDDER_IMPLEMENTATIONS))
def embedder(request: pytest.FixtureRequest) -> Embedder:
    """A fresh embedder of the current implementation."""
    return EMBEDDER_IMPLEMENTATIONS[request.param]()
