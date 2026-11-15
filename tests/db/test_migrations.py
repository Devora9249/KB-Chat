"""Migration checks that run without a database (Alembic offline / `--sql` mode)."""

import io
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

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


@pytest.mark.integration
def test_upgrade_and_downgrade_on_real_database() -> None:
    """Needs KC_INTEGRATION=1 and DATABASE_URL pointing at Postgres with pgvector available."""
    assert os.environ.get("DATABASE_URL"), "set DATABASE_URL for integration tests"
    config = _config()
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")  # leave the DB migrated
