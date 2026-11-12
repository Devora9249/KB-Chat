"""Shared test setup for tests/db (Storage & Indexing).

Integration tests need a real Postgres + pgvector. Mark them with
`@pytest.mark.integration`. They run automatically when DATABASE_URL is set in the
environment (as in CI) and are skipped with "DATABASE_URL not set; runs in CI" otherwise.
Before the FIRST integration test of a run, the database is migrated once
(`alembic upgrade head`, idempotent), so tests do not depend on a CI step or on order.
Destructive integration tests must also call `require_test_database(url)`.

The marker is registered here (not in pyproject.toml) to stay inside this domain's
folders. The skip applies only to tests collected under tests/db.
"""

import pytest
from db_test_support import SKIP_REASON, integration_enabled, require_database_url, upgrade_to_head


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs a real Postgres + pgvector; runs only when DATABASE_URL is set",
    )


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Runs `alembic upgrade head` once per test session; returns the DATABASE_URL used."""
    url = require_database_url()
    upgrade_to_head(url)
    return url


@pytest.fixture(autouse=True)
def _integration_setup(request: pytest.FixtureRequest) -> None:
    """Skip integration tests without DATABASE_URL; otherwise make sure the DB is migrated."""
    if not request.node.get_closest_marker("integration"):
        return
    if not integration_enabled():
        pytest.skip(SKIP_REASON)
    request.getfixturevalue("migrated_database")
