import pytest
from pydantic import ValidationError

from kc.db.settings import DbSettings

URL = "postgresql+psycopg://kc:s3cret@localhost:5432/kc"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "DB_ECHO_SQL", "DB_POOL_SIZE"):
        monkeypatch.delenv(name, raising=False)


def _load() -> DbSettings:
    # _env_file=None: ignore any developer's local .env so tests are reproducible.
    return DbSettings(_env_file=None)  # type: ignore[call-arg]


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("DB_ECHO_SQL", "true")
    monkeypatch.setenv("DB_POOL_SIZE", "10")
    settings = _load()
    assert settings.database_url.get_secret_value() == URL
    assert settings.db_echo_sql is True
    assert settings.db_pool_size == 10


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    settings = _load()
    assert settings.db_echo_sql is False
    assert settings.db_pool_size == 5


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


def test_invalid_pool_size_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("DB_POOL_SIZE", "0")
    with pytest.raises(ValidationError):
        _load()
