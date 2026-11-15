"""Fixtures for the IndexStore and Embedder contract test suites (owner: Raizy).

Every test using `store` / `make_store` runs once per entry in IMPLEMENTATIONS, and every
test using `embedder` runs once per entry in EMBEDDER_IMPLEMENTATIONS.
To add an implementation (e.g. PgIndexStore, BgeM3Embedder), add one entry.
Other teams are welcome to add contract tests via PR.
"""

from collections.abc import Callable

import pytest
from index_store_helpers import DIM, StoreFactory

from kc.indexing._provisional_contracts import Embedder, IndexStore
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
    """Factory for a fresh store of the current implementation (any dimension)."""
    factory: StoreFactory = IMPLEMENTATIONS[request.param]
    return factory


@pytest.fixture
def store(make_store: StoreFactory) -> IndexStore:
    """A fresh, empty store with dimension DIM (3)."""
    return make_store(DIM)


# TODO(BgeM3Embedder): add "bge_m3" here, marked real_model (runs only with KC_REAL_EMBEDDER=1).
EMBEDDER_IMPLEMENTATIONS: dict[str, Callable[[], Embedder]] = {
    "fake": lambda: FakeEmbedder(dimension=64),
}


@pytest.fixture(params=sorted(EMBEDDER_IMPLEMENTATIONS))
def embedder(request: pytest.FixtureRequest) -> Embedder:
    """A fresh embedder of the current implementation."""
    return EMBEDDER_IMPLEMENTATIONS[request.param]()
