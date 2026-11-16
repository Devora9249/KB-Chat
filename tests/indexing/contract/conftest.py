"""Fixtures for the storage contract test suites (owner: Raizy).

- `make_search_index`: a SearchIndex already loaded with given documents. Search tests use
  ONLY vector_search / text_search on it (contracts.storage.SearchIndex).
- `store` / `make_store`: a fresh, empty IndexStore for write tests (provisional
  IndexStore = SearchIndex + IndexWriter).
- `embedder`: one per entry in EMBEDDER_IMPLEMENTATIONS (contracts.storage.Embedder).

Every test runs once per registered implementation. To add one (e.g. PgIndexStore,
BgeM3Embedder), add one entry. Other teams are welcome to add contract tests via PR.
"""

from collections.abc import Callable

import pytest
from index_store_helpers import DIM, IndexedDocument, SearchIndexFactory, StoreFactory, run

from contracts.storage import Embedder, SearchIndex
from kc.indexing._provisional_contracts import IndexStore
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore


def _in_memory(dimension: int) -> IndexStore:
    return InMemoryIndexStore(dimension=dimension, model_name="contract-tests")


# TODO(PgIndexStore): add "pg" here (needs Postgres; mark/skip like tests/db integration tests).
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


# Each entry gets the pytest request, so real models can reuse the session-wide instance
# (fixture `bge_m3_embedder` in tests/indexing/conftest.py) instead of loading per test.
EMBEDDER_IMPLEMENTATIONS: dict[str, Callable[[pytest.FixtureRequest], Embedder]] = {
    "fake": lambda request: FakeEmbedder(dimension=64),
    "bge_m3": lambda request: request.getfixturevalue("bge_m3_embedder"),
    "e5": lambda request: request.getfixturevalue("e5_embedder"),
}
# Run only with KC_REAL_EMBEDDER=1 (marker handled in tests/indexing/conftest.py).
REAL_MODEL_EMBEDDERS = {"bge_m3", "e5"}


@pytest.fixture(
    params=[
        pytest.param(name, marks=pytest.mark.real_model) if name in REAL_MODEL_EMBEDDERS else name
        for name in sorted(EMBEDDER_IMPLEMENTATIONS)
    ]
)
def embedder(request: pytest.FixtureRequest) -> Embedder:
    """An embedder of the current implementation."""
    return EMBEDDER_IMPLEMENTATIONS[request.param](request)
