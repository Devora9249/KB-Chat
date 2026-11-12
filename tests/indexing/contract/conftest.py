"""Fixtures for the IndexStore contract test suite (owner: Raizy).

Every test in this folder runs once per implementation in IMPLEMENTATIONS.
To add an implementation (e.g. PgIndexStore), add one entry that returns a NEW, EMPTY store.
Other teams are welcome to add contract tests via PR.
"""

import pytest
from index_store_helpers import DIM, StoreFactory

from kc.indexing._provisional_contracts import IndexStore
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
