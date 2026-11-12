"""Helpers shared by tests/db (kept out of conftest.py: several folders have a conftest.py,
so `from conftest import ...` could pick up the wrong one depending on collection order).

Integration tests run automatically when DATABASE_URL is set in the environment (as in CI)
and are skipped otherwise. pytest does NOT read `.env`, so a local `.env` never turns them on.
Before the first integration test, a TEST database (name contains "test") is migrated once
(`alembic upgrade head`); any other database is left untouched.

Destructive tests (dropping tables, deleting rows) additionally require the database NAME
to contain "test", so they can never wipe a shared development database by accident.
"""

import io
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

DATABASE_URL_ENV_VAR = "DATABASE_URL"
SKIP_REASON = "DATABASE_URL not set; runs in CI"
DESTRUCTIVE_SKIP_REASON = "destructive test: database name must contain 'test'"
ALEMBIC_INI = Path(__file__).resolve().parents[2] / "src" / "kc" / "db" / "alembic.ini"


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


def alembic_config(
    *, database_url: str | None = None, output_buffer: io.StringIO | None = None
) -> Config:
    """Alembic config for tests. An explicit `database_url` bypasses DbSettings/.env."""
    config = Config(str(ALEMBIC_INI), output_buffer=output_buffer)
    config.attributes["configure_logger"] = False  # keep pytest's logging setup intact
    if database_url is not None:
        config.attributes["database_url"] = database_url
    return config


def upgrade_to_head(url: str) -> None:
    """Idempotent: Alembic applies only migrations missing from alembic_version."""
    command.upgrade(alembic_config(database_url=url), "head")


def migrate_if_test_database(url: str, upgrade: Callable[[str], None] = upgrade_to_head) -> bool:
    """Run `upgrade(url)` only if the database NAME contains "test"; return whether it ran.

    A shared/non-test database is never migrated by tests (an unmerged migration on a
    branch must not change everyone's schema). `upgrade` is injectable for unit tests.
    """
    if not is_test_database(url):
        return False
    upgrade(url)
    return True
