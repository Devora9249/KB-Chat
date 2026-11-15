"""Schema of migrations 0003 (documents, chunks) and 0004 (embedding + HNSW).

- Offline tests (no database): the SQL that Alembic would run (`--sql` mode).
- Integration tests (CI, DATABASE_URL set): the real tables, columns and indexes in
  Postgres, the CASCADE and the computed tsvector, and the pgvector version.
"""

import asyncio
import io
import re
import warnings
from typing import Any

import pytest
from alembic import command
from db_test_support import alembic_config, require_database_url, require_test_database
from sqlalchemy import text

from kc.db.engine import create_engine
from kc.db.settings import db_settings_from_url

FAKE_URL = "postgresql+psycopg://kc:kc@localhost:5432/kc"
DIMENSION = 1024


def _sql(monkeypatch: pytest.MonkeyPatch, revisions: str, *, upgrade: bool = True) -> str:
    monkeypatch.setenv("DATABASE_URL", FAKE_URL)
    buffer = io.StringIO()
    config = alembic_config(output_buffer=buffer)
    if upgrade:
        command.upgrade(config, revisions, sql=True)
    else:
        command.downgrade(config, revisions, sql=True)
    return " ".join(buffer.getvalue().split())  # normalize whitespace for matching


# --- offline: 0003 -----------------------------------------------------------------------------


def test_0003_creates_documents_table(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _sql(monkeypatch, "0002:0003")
    for column in (
        "url TEXT NOT NULL",
        "title TEXT NOT NULL",
        "breadcrumb TEXT[] DEFAULT '{}'::text[] NOT NULL",
        "product TEXT NOT NULL",
        "version TEXT,",  # nullable
        "doc_type TEXT NOT NULL",
        "content_md TEXT NOT NULL",
        "content_hash TEXT NOT NULL",
        "indexed_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL",
        "CONSTRAINT pk_documents PRIMARY KEY (url)",
    ):
        assert column in sql, column


def test_0003_creates_chunks_with_cascade_and_computed_simple_tsvector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql = _sql(monkeypatch, "0002:0003")
    assert "CREATE TABLE chunks" in sql
    assert "CONSTRAINT pk_chunks PRIMARY KEY (id)" in sql  # id unique across the index (D12)
    assert "FOREIGN KEY(document_url) REFERENCES documents (url) ON DELETE CASCADE" in sql
    assert "tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, text)) STORED" in sql
    assert "embedding" not in sql  # the vector belongs to 0004
    assert "CHECK" not in sql  # metadata values are validated by Pydantic, not the database


@pytest.mark.parametrize(
    "index",
    [
        "CREATE INDEX ix_chunks_document_url ON chunks (document_url)",
        "CREATE INDEX ix_chunks_product ON chunks (product)",
        "CREATE INDEX ix_chunks_version ON chunks (version)",
        "CREATE INDEX ix_chunks_doc_type ON chunks (doc_type)",
        "CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)",
    ],
)
def test_0003_creates_indexes(monkeypatch: pytest.MonkeyPatch, index: str) -> None:
    assert index in _sql(monkeypatch, "0002:0003")


def test_0003_downgrade_drops_everything_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _sql(monkeypatch, "0003:0002", upgrade=False)
    assert "DROP TABLE chunks" in sql
    assert "DROP TABLE documents" in sql
    assert sql.index("DROP TABLE chunks") < sql.index("DROP TABLE documents")  # FK order
    for index in ("document_url", "product", "version", "doc_type", "tsv"):
        assert f"DROP INDEX ix_chunks_{index}" in sql


# --- offline: 0004 -----------------------------------------------------------------------------


def test_0004_adds_vector_1024_with_hnsw_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _sql(monkeypatch, "0003:0004")
    assert f"ALTER TABLE chunks ADD COLUMN embedding VECTOR({DIMENSION}) NOT NULL" in sql
    assert (
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
        in sql
    )


def test_0004_downgrade_drops_index_then_column(monkeypatch: pytest.MonkeyPatch) -> None:
    sql = _sql(monkeypatch, "0004:0003", upgrade=False)
    assert sql.index("DROP INDEX ix_chunks_embedding_hnsw") < sql.index(
        "ALTER TABLE chunks DROP COLUMN embedding"
    )


# --- integration (CI): the real schema ---------------------------------------------------------


def _query(url: str, sql: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
    async def run() -> list[tuple[Any, ...]]:
        engine = create_engine(db_settings_from_url(url))
        try:
            async with engine.begin() as connection:
                result = await connection.execute(text(sql), params or {})
                return [tuple(row) for row in result.all()] if result.returns_rows else []
        finally:
            await engine.dispose()

    # Selector loop: psycopg async does not support Windows' default Proactor loop.
    return asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)


@pytest.mark.integration
def test_pgvector_version_supports_hnsw_and_reports_iterative_scan() -> None:
    url = require_database_url()
    ((version,),) = _query(url, "SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    major, minor = (int(part) for part in re.findall(r"\d+", version)[:2])
    iterative_scan = (major, minor) >= (0, 8)
    # Shown in the "warnings summary" of the CI log even though the test passes.
    warnings.warn(
        f"pgvector {version} in this database; hnsw.iterative_scan available: {iterative_scan}",
        UserWarning,
        stacklevel=1,
    )
    assert (major, minor) >= (0, 5), f"HNSW needs pgvector >= 0.5, got {version}"


@pytest.mark.integration
def test_tables_and_columns_exist_with_expected_types() -> None:
    url = require_database_url()
    rows = _query(
        url,
        """
        SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull,
               a.attgenerated
        FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname IN ('documents', 'chunks') AND a.attnum > 0 AND NOT a.attisdropped
        """,
    )
    columns = {
        (table, name): (type_, not_null, generated)
        for table, name, type_, not_null, generated in rows
    }

    assert columns[("documents", "url")][:2] == ("text", True)
    assert columns[("documents", "breadcrumb")][:2] == ("text[]", True)
    assert columns[("documents", "version")][:2] == ("text", False)
    assert columns[("chunks", "id")][:2] == ("text", True)
    assert columns[("chunks", "heading_path")][:2] == ("text[]", True)
    assert columns[("chunks", "embedding")][:2] == (f"vector({DIMENSION})", True)
    assert columns[("chunks", "tsv")] == ("tsvector", True, "s")  # 's' = generated, stored


@pytest.mark.integration
def test_indexes_exist_with_expected_methods() -> None:
    url = require_database_url()
    indexes = dict(
        _query(url, "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks'")
    )
    assert "USING hnsw (embedding vector_cosine_ops)" in indexes["ix_chunks_embedding_hnsw"]
    assert "USING gin (tsv)" in indexes["ix_chunks_tsv"]
    for column in ("document_url", "product", "version", "doc_type"):
        assert f"USING btree ({column})" in indexes[f"ix_chunks_{column}"]
    assert "pk_chunks" in indexes


@pytest.mark.integration
def test_cascade_and_computed_tsvector_work() -> None:
    """DESTRUCTIVE (writes and deletes rows): only on a database whose name contains "test"."""
    url = require_database_url()
    require_test_database(url)
    vector = "[" + ",".join(["1"] + ["0"] * (DIMENSION - 1)) + "]"
    _query(
        url,
        """
        INSERT INTO documents (url, title, product, version, doc_type, content_md, content_hash)
        VALUES ('test://schema', 'T', 'sequence', '9', 'how-to', '...', 'h');
        """,
    )
    try:
        _query(
            url,
            """
            INSERT INTO chunks (id, document_url, text, product, version, doc_type, embedding)
            VALUES ('test-schema-1', 'test://schema', :text,
                    'sequence', '9', 'how-to', CAST(:vector AS vector))
            """,
            {"text": "Configure Set-CoraSeQuenceSiteAuthentication", "vector": vector},
        )
        ((tsv,),) = _query(url, "SELECT tsv::text FROM chunks WHERE id = 'test-schema-1'")
        assert "'configure'" in tsv  # computed by Postgres, lower-cased, no stemming
        assert (
            "'set-corasequencesiteauthentication'" in tsv
            or "'corasequencesiteauthentication'" in tsv
        )
    finally:
        _query(url, "DELETE FROM documents WHERE url = 'test://schema'")
    assert _query(url, "SELECT count(*) FROM chunks WHERE id = 'test-schema-1'") == [(0,)]
