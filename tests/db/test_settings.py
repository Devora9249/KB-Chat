import inspect

import pytest
from pydantic import ValidationError
from sqlalchemy.pool import QueuePool

from kc.db.settings import DbSettings

URL = "postgresql+psycopg://kc:s3cret@localhost:5432/kc"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DATABASE_URL",
        "DB_ECHO_SQL",
        "DB_POOL_SIZE",
        "DB_MAX_OVERFLOW",
        "DB_POOL_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def _load() -> DbSettings:
    # _env_file=None: ignore any developer's local .env so tests are reproducible.
    return DbSettings(_env_file=None)  # type: ignore[call-arg]


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("DB_ECHO_SQL", "true")
    monkeypatch.setenv("DB_POOL_SIZE", "10")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "3")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "2.5")
    settings = _load()
    assert settings.database_url.get_secret_value() == URL
    assert settings.db_echo_sql is True
    assert settings.db_pool_size == 10
    assert settings.db_max_overflow == 3
    assert settings.db_pool_timeout == 2.5


def test_defaults_match_sqlalchemy_queue_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset -> SQLAlchemy's own defaults, so adding the settings changes no behaviour.
    monkeypatch.setenv("DATABASE_URL", URL)
    settings = _load()
    assert settings.db_echo_sql is False
    assert (settings.db_pool_size, settings.db_max_overflow, settings.db_pool_timeout) == (
        5,
        10,
        30.0,
    )
    defaults = inspect.signature(QueuePool.__init__).parameters
    assert settings.db_pool_size == defaults["pool_size"].default
    assert settings.db_max_overflow == defaults["max_overflow"].default
    assert settings.db_pool_timeout == defaults["timeout"].default


def test_missing_url_raises() -> None:
    with pytest.raises(ValidationError):
        _load()


def test_wrong_driver_raises_without_leaking_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://kc:s3cret@localhost/kc")
    with pytest.raises(ValidationError) as exc_info:
        _load()
    assert "postgresql+psycopg://" in str(exc_info.value)
    assert "s3cret" not in str(exc_info.value)


def test_password_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    assert "s3cret" not in repr(_load())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DB_POOL_SIZE", "0"),
        ("DB_MAX_OVERFLOW", "-1"),  # SQLAlchemy's "unlimited" is rejected on purpose
        ("DB_POOL_TIMEOUT", "0"),
        ("DB_POOL_TIMEOUT", "-5"),
    ],
)
def test_invalid_pool_settings_raise(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        _load()
