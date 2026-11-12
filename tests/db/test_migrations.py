"""Migration checks that run without a database (Alembic offline / `--sql` mode)."""

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from db_test_support import require_database_url, require_test_database

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "src" / "kc" / "db" / "alembic.ini"
FAKE_URL = "postgresql+psycopg://kc:kc@localhost:5432/kc"


def _config(buffer: io.StringIO | None = None) -> Config:
    config = Config(str(ALEMBIC_INI), output_buffer=buffer)
    config.attributes["configure_logger"] = False  # keep pytest's logging setup intact
    return config


def _offline_sql(monkeypatch: pytest.MonkeyPatch, *, upgrade: bool) -> str:
    monkeypatch.setenv("DATABASE_URL", FAKE_URL)
    buffer = io.StringIO()
    config = _config(buffer)
    if upgrade:
        command.upgrade(config, "head", sql=True)
    else:
        command.downgrade(config, "head:base", sql=True)
    return buffer.getvalue()


def test_single_linear_history() -> None:
    script = ScriptDirectory.from_config(_config())
    assert script.get_heads() == ["0002"]
    revisions = [rev.revision for rev in script.walk_revisions("base", "heads")]
    assert revisions == ["0002", "0001"]


def test_upgrade_sql_enables_pgvector_then_creates_embedding_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql = _offline_sql(monkeypatch, upgrade=True)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE embedding_config" in sql
    assert "CONSTRAINT ck_embedding_config_single_row CHECK (id = 1)" in sql
    assert "SERIAL" not in sql  # single-row table: no auto-increment sequence
    assert sql.index("CREATE EXTENSION") < sql.index("CREATE TABLE embedding_config")


def test_downgrade_sql_reverses_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _offline_sql(monkeypatch, upgrade=False)
    assert "DROP TABLE embedding_config" in sql
    assert "DROP EXTENSION IF EXISTS vector" in sql
    assert sql.index("DROP TABLE embedding_config") < sql.index("DROP EXTENSION")


def test_offline_sql_does_not_contain_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://kc:s3cret@localhost/kc")
    buffer = io.StringIO()
    command.upgrade(_config(buffer), "head", sql=True)
    assert "s3cret" not in buffer.getvalue()


def test_explicit_url_attribute_is_used_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # With DATABASE_URL unset, env.py would fail validation; an explicit URL must be enough.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    buffer = io.StringIO()
    config = _config(buffer)
    config.attributes["database_url"] = FAKE_URL
    command.upgrade(config, "head", sql=True)
    assert "CREATE TABLE embedding_config" in buffer.getvalue()


@pytest.mark.integration
def test_upgrade_and_downgrade_on_real_database() -> None:
    """DESTRUCTIVE (drops all tables): runs only on a database whose name contains "test".

    Uses DATABASE_URL from the environment explicitly (never `.env`) and always finishes
    with `upgrade head`, so tests that run afterwards find the tables.
    """
    url = require_database_url()
    require_test_database(url)
    config = _config()
    config.attributes["database_url"] = url
    command.upgrade(config, "head")
    try:
        command.downgrade(config, "base")
    finally:
        command.upgrade(config, "head")  # leave the DB migrated, even if downgrade failed
