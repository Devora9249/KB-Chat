"""Helpers shared by tests/db (kept out of conftest.py: several folders have a conftest.py,
so `from conftest import ...` could pick up the wrong one depending on collection order).

Integration tests run automatically when DATABASE_URL is set in the environment (as in CI)
and are skipped otherwise. pytest does NOT read `.env`, so a local `.env` never turns them on.

Destructive tests (dropping tables, deleting rows) additionally require the database NAME
to contain "test", so they can never wipe a shared development database by accident.
"""

import os

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

DATABASE_URL_ENV_VAR = "DATABASE_URL"
SKIP_REASON = "DATABASE_URL not set; runs in CI"
DESTRUCTIVE_SKIP_REASON = "destructive test: database name must contain 'test'"


def database_url() -> str | None:
    """DATABASE_URL from the environment only; None if unset or blank."""
    value = os.environ.get(DATABASE_URL_ENV_VAR, "").strip()
    return value or None


def integration_enabled() -> bool:
    return database_url() is not None


def require_database_url() -> str:
    """For use inside integration tests (the autouse fixture has already skipped if unset)."""
    url = database_url()
    if url is None:
        pytest.skip(SKIP_REASON)
    return url


def is_test_database(url: str) -> bool:
    """True only if the database NAME (not user/host) contains "test". Unparsable -> False."""
    try:
        name = make_url(url).database
    except ArgumentError:
        return False
    return name is not None and "test" in name


def require_test_database(url: str) -> None:
    """Skip the calling test unless `url` points at a database whose name contains "test"."""
    if not is_test_database(url):
        pytest.skip(DESTRUCTIVE_SKIP_REASON)
