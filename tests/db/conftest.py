"""Shared test setup for tests/db (Storage & Indexing).

Integration tests need a real Postgres + pgvector. Mark them with
`@pytest.mark.integration`. They run automatically when DATABASE_URL is set in the
environment (as in CI) and are skipped with "DATABASE_URL not set; runs in CI" otherwise.
Destructive integration tests must also call `require_test_database(url)`.

The marker is registered here (not in pyproject.toml) to stay inside this domain's
folders. The skip applies only to tests collected under tests/db.
"""

import pytest
from db_test_support import SKIP_REASON, integration_enabled


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: needs a real Postgres + pgvector; runs only when DATABASE_URL is set",
    )


@pytest.fixture(autouse=True)
def _skip_integration_unless_enabled(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("integration") and not integration_enabled():
        pytest.skip(SKIP_REASON)
