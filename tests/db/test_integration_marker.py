"""Checks the integration switch and the destructive-test guard (no database needed)."""

import pytest
from db_test_support import (
    DATABASE_URL_ENV_VAR,
    DESTRUCTIVE_SKIP_REASON,
    database_url,
    integration_enabled,
    is_test_database,
    require_test_database,
)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_integration_disabled_when_database_url_missing_or_blank(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(DATABASE_URL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(DATABASE_URL_ENV_VAR, value)
    assert database_url() is None
    assert integration_enabled() is False


def test_integration_enabled_when_database_url_set(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql+psycopg://kc:kc@localhost:5432/kc_test"
    monkeypatch.setenv(DATABASE_URL_ENV_VAR, f"  {url}  ")
    assert database_url() == url
    assert integration_enabled() is True


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql+psycopg://kc:kc@localhost:5432/kc_test", True),
        ("postgresql+psycopg://kc:kc@localhost:5432/test", True),
        ("postgresql+psycopg://kc:kc@localhost:5432/kc", False),
        # "test" only in user / password / host must NOT count
        ("postgresql+psycopg://test:test@test-host:5432/kc", False),
        ("postgresql+psycopg://kc@localhost", False),  # no database name
        ("not a url", False),
    ],
)
def test_is_test_database_checks_only_the_database_name(url: str, expected: bool) -> None:
    assert is_test_database(url) is expected


def test_require_test_database_skips_for_non_test_database() -> None:
    with pytest.raises(pytest.skip.Exception, match=DESTRUCTIVE_SKIP_REASON):
        require_test_database("postgresql+psycopg://kc:kc@localhost:5432/kc")


def test_require_test_database_allows_test_database() -> None:
    require_test_database("postgresql+psycopg://kc:kc@localhost:5432/kc_test")  # no skip


@pytest.mark.integration
def test_integration_marker_runs_only_when_enabled() -> None:
    # If we got here, the autouse fixture did not skip us, so DATABASE_URL must be set.
    assert integration_enabled()
