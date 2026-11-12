"""Shared test setup for tests/db (Storage & Indexing).

Integration tests need a real Postgres + pgvector. Mark them with
`@pytest.mark.integration`; they are skipped unless KC_INTEGRATION=1 is set:

    $env:KC_INTEGRATION = "1"; uv run pytest tests/db      # PowerShell
    KC_INTEGRATION=1 uv run pytest tests/db                # bash

The marker is registered here (not in pyproject.toml) to stay inside this domain's
folders. The skip applies only to tests collected under tests/db.
"""

import pytest
from db_test_support import INTEGRATION_ENV_VAR, integration_enabled


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"integration: needs a real Postgres + pgvector; runs only when {INTEGRATION_ENV_VAR}=1",
    )


@pytest.fixture(autouse=True)
def _skip_integration_unless_enabled(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("integration") and not integration_enabled():
        pytest.skip(f"integration test: set {INTEGRATION_ENV_VAR}=1 to run")
