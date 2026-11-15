"""PostgreSQL + pgvector implementation of the index: IndexStore = SearchIndex + IndexWriter.

Sessions: receives an async_sessionmaker (the Engine and its pool are created and disposed
by the application) and opens a SHORT session per call, so one instance is safe for
concurrent async calls.

upsert_document runs in ONE transaction:
  1. validate the input (ValueError before touching the database),
  2. INSERT ... ON CONFLICT (url) DO UPDATE the document row,
  3. DELETE all its chunks,
  4. INSERT the new chunks (one executemany),
  5. COMMIT - only now other sessions see the change; any failure rolls everything back,
     so the previous version stays. Concurrent searches see the old or the new version,
     never a mix (Postgres MVCC).

vector_search (cosine; score = 1 - cosine distance):
- inner query: `ORDER BY embedding <=> :q LIMIT k + TIE_MARGIN` with the filters in WHERE,
  answered by the HNSW index; outer query: JOIN documents (title), ORDER BY distance,
  chunk_id, LIMIT k. The margin lets equal distances at the cut-off be ordered by chunk_id.
- hnsw.iterative_scan = strict_order: if the filters reject the first ef_search candidates,
  pgvector keeps scanning the index until it has enough matching rows (up to
  hnsw.max_scan_tuples), so a narrow filter still returns k results when they exist.
  strict_order (not relaxed_order) keeps results exactly ordered by distance, which the
  contract requires.
- hnsw.ef_search = max(40, k + TIE_MARGIN), capped at 1000; both settings are
  transaction-local (set_config(..., is_local => true) == SET LOCAL), so other requests are
  not affected.

text_search: tsv @@ websearch_to_tsquery('simple', :q), ORDER BY ts_rank DESC, chunk_id.
A query without any word returns [] without touching the database.

Filters (contracts.retrieval.Filters): OR within a field, AND across fields,
exclude_doc_types removes, and `versions` never matches version IS NULL (SQL `IN` is never
true for NULL).

Embedding-model guard: call `await store.verify_embedding_config(model_name, dimension)`
once at startup. It reuses kc.db.embedding_config (records the model on first use, raises
EmbeddingConfigMismatchError if the index was built with another model or dimension).

Errors:
- ValueError: invalid input, never wrapped. Raised before any database access, except
  one case the database detects: a chunk id already used by a chunk of ANOTHER document
  (chunk.id is unique across the whole index, D12). That IntegrityError on the chunks
  primary key (pk_chunks) becomes ValueError (`raise ... from e`); the transaction is
  rolled back, so nothing changes.
- IndexStoreError (`raise ... from e`): ONLY infrastructure failures - SQLAlchemy
  DBAPIError and its OperationalError / InterfaceError, psycopg OperationalError /
  InterfaceError, and SQLAlchemy's pool TimeoutError (no free connection in time).
  Any other IntegrityError (not pk_chunks) is a DBAPIError and stays wrapped.
- No retries here: the caller decides. asyncio.CancelledError is never caught.
"""

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from sqlalchemy import ColumnElement, Select, delete, insert, literal, select
from sqlalchemy import exc as sa_exc
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from contracts.retrieval import Filters, SearchResult
from contracts.storage import IndexStoreError
from kc.db.embedding_config import EmbeddingConfigMismatchError, ensure_embedding_config
from kc.db.models import EMBEDDING_DIMENSION, ChunkRow, DocumentRow
from kc.indexing._provisional_contracts import Chunk, Document
from kc.indexing.upsert_validation import validate_upsert

# Extra rows fetched by the inner (index) query, so equal distances / ranks at the k-th
# position are still ordered by chunk_id in the outer query.
TIE_MARGIN = 10
DEFAULT_EF_SEARCH = 40  # pgvector's default
MAX_EF_SEARCH = 1000  # pgvector's upper limit
_WORD_RE = re.compile(r"\w", re.UNICODE)


def ef_search_for(k: int) -> int:
    """HNSW candidate list size for a top-k query (transaction-local setting)."""
    return min(MAX_EF_SEARCH, max(DEFAULT_EF_SEARCH, k + TIE_MARGIN))


def vector_search_settings(k: int) -> dict[str, str]:
    """pgvector settings vector_search applies, transaction-local, for a top-k query.

    Single source of truth, also used by tests that run the statement directly.
    """
    return {"hnsw.iterative_scan": "strict_order", "hnsw.ef_search": str(ef_search_for(k))}


def apply_settings_statement(settings: dict[str, str]) -> Select[Any]:
    """SELECT set_config(name, value, true), ... - i.e. SET LOCAL with bound parameters."""
    return select(*(func.set_config(name, value, True) for name, value in settings.items()))


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def _filter_clauses(filters: Filters | None) -> list[ColumnElement[bool]]:
    """Filters semantics: OR within a field (IN), AND across fields (list of clauses)."""
    if filters is None:
        return []
    clauses: list[ColumnElement[bool]] = []
    if filters.products is not None:
        clauses.append(ChunkRow.product.in_(filters.products))
    if filters.versions is not None:
        clauses.append(ChunkRow.version.in_(filters.versions))  # NULL never matches IN
    if filters.doc_types is not None:
        clauses.append(ChunkRow.doc_type.in_(filters.doc_types))
    if filters.exclude_doc_types is not None:
        clauses.append(ChunkRow.doc_type.not_in(filters.exclude_doc_types))
    return clauses


_RESULT_COLUMNS = (
    ChunkRow.id,
    ChunkRow.document_url,
    ChunkRow.text,
    ChunkRow.heading_path,
    ChunkRow.product,
    ChunkRow.version,
    ChunkRow.doc_type,
)


def build_vector_search_statement(
    vector: list[float], filters: Filters | None, k: int
) -> Select[Any]:
    """Inner query uses the HNSW index (ORDER BY distance LIMIT); outer adds title and the
    chunk_id tie-break. Public so tests can EXPLAIN exactly what vector_search runs."""
    distance = ChunkRow.embedding.cosine_distance(vector).label("distance")
    inner = (
        select(*_RESULT_COLUMNS, distance)
        .where(*_filter_clauses(filters))
        .order_by(distance)
        .limit(k + TIE_MARGIN)
        .subquery("nearest")
    )
    return (
        select(inner, DocumentRow.title, (1 - inner.c.distance).label("score"))
        .join(DocumentRow, DocumentRow.url == inner.c.document_url)
        .order_by(inner.c.distance, inner.c.id)
        .limit(k)
    )


def build_text_search_statement(query: str, filters: Filters | None, k: int) -> Select[Any]:
    tsquery = func.websearch_to_tsquery(literal("simple", REGCONFIG), query)
    rank = func.ts_rank(ChunkRow.tsv, tsquery).label("score")
    return (
        select(*_RESULT_COLUMNS, DocumentRow.title, rank)
        .join(DocumentRow, DocumentRow.url == ChunkRow.document_url)
        .where(ChunkRow.tsv.op("@@")(tsquery), *_filter_clauses(filters))
        .order_by(rank.desc(), ChunkRow.id)
        .limit(k)
    )


def _to_result(row: Any) -> SearchResult:
    return SearchResult(
        chunk_id=row.id,
        url=row.document_url,
        title=row.title,
        text=row.text,
        score=float(row.score),
        product=row.product,
        version=row.version,
        doc_type=row.doc_type,
        heading_path=list(row.heading_path),
    )


# Infrastructure failures that are wrapped in IndexStoreError. Nothing broader on purpose.
INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (
    sa_exc.DBAPIError,  # includes sa_exc.OperationalError and sa_exc.InterfaceError
    sa_exc.TimeoutError,  # connection pool exhausted for longer than pool_timeout
    psycopg.OperationalError,
    psycopg.InterfaceError,
)


# Name of the chunks primary key (migration 0003, op.f("pk_chunks")).
CHUNKS_PRIMARY_KEY = "pk_chunks"


def _is_chunk_id_conflict(error: sa_exc.IntegrityError) -> bool:
    """True if the database rejected a chunk id that is already used (pk_chunks)."""
    diagnostics = getattr(error.orig, "diag", None)  # psycopg error details
    return getattr(diagnostics, "constraint_name", None) == CHUNKS_PRIMARY_KEY


@asynccontextmanager
async def _infrastructure_errors(operation: str) -> AsyncIterator[None]:
    try:
        yield
    except INFRASTRUCTURE_ERRORS as e:
        raise IndexStoreError(f"PgIndexStore.{operation} failed") from e


class PgIndexStore:
    """IndexStore (SearchIndex + IndexWriter) backed by PostgreSQL + pgvector."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        if dimension != EMBEDDING_DIMENSION:
            # The column is vector(EMBEDDING_DIMENSION) (migration 0004); another dimension
            # needs a new migration first.
            raise ValueError(
                f"dimension {dimension} does not match the chunks.embedding column "
                f"(vector({EMBEDDING_DIMENSION}))"
            )
        self._session_factory = session_factory
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def verify_embedding_config(self, model_name: str, dimension: int) -> None:
        """Call once at startup. Records the model on first use; raises
        EmbeddingConfigMismatchError if the index was built with another model/dimension
        (a full re-index is then required)."""
        if dimension != self._dimension:
            raise EmbeddingConfigMismatchError(
                f"embedder {model_name!r} has dimension {dimension}, but the index column is "
                f"vector({self._dimension}); a new migration and a full re-index are required"
            )
        async with _infrastructure_errors("verify_embedding_config"):
            async with self._session_factory() as session:
                await ensure_embedding_config(session, model_name, dimension)

    async def upsert_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        validate_upsert(document, chunks, vectors, self._dimension)  # before any DB access

        document_values = {
            "url": document.url,
            "title": document.title,
            "breadcrumb": list(document.breadcrumb),
            "product": document.product,
            "version": document.version,
            "doc_type": document.doc_type,
            "content_md": document.content_md,
            "content_hash": document.content_hash,
        }
        upsert_document = (
            pg_insert(DocumentRow)
            .values(**document_values)
            .on_conflict_do_update(
                index_elements=[DocumentRow.url],
                set_={
                    **{k: v for k, v in document_values.items() if k != "url"},
                    "indexed_at": func.now(),
                },
            )
        )
        chunk_rows = [
            {
                "id": chunk.id,
                "document_url": chunk.document_url,
                "text": chunk.text,
                "heading_path": list(chunk.heading_path),
                "product": chunk.product,
                "version": chunk.version,
                "doc_type": chunk.doc_type,
                "embedding": list(vector),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        async with _infrastructure_errors("upsert_document"):
            try:
                async with self._session_factory() as session, session.begin():
                    await session.execute(upsert_document)
                    await session.execute(
                        delete(ChunkRow).where(ChunkRow.document_url == document.url)
                    )
                    if chunk_rows:
                        await session.execute(insert(ChunkRow), chunk_rows)
                # session.begin() commits here, or rolls back if anything above raised.
            except sa_exc.IntegrityError as e:
                if not _is_chunk_id_conflict(e):
                    raise  # any other integrity error: wrapped in IndexStoreError below
                detail = getattr(getattr(e.orig, "diag", None), "message_detail", None)
                raise ValueError(
                    f"chunk id already used by another document while upserting "
                    f"{document.url!r} (chunk.id must be unique across the whole index, D12)"
                    + (f": {detail}" if detail else "")
                ) from e

    async def delete_document(self, url: str) -> None:
        """No-op if the document does not exist; its chunks go with it (ON DELETE CASCADE)."""
        async with _infrastructure_errors("delete_document"):
            async with self._session_factory() as session, session.begin():
                await session.execute(delete(DocumentRow).where(DocumentRow.url == url))

    async def get_document_hashes(self) -> dict[str, str]:
        async with _infrastructure_errors("get_document_hashes"):
            async with self._session_factory() as session:
                rows = await session.execute(select(DocumentRow.url, DocumentRow.content_hash))
                return {url: content_hash for url, content_hash in rows.all()}

    # --- SearchIndex ------------------------------------------------------------------------

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        _check_k(k)
        if len(vector) != self._dimension:
            raise ValueError(
                f"query vector has dimension {len(vector)}, index dimension is {self._dimension}"
            )
        statement = build_vector_search_statement(list(vector), filters, k)
        async with _infrastructure_errors("vector_search"):
            async with self._session_factory() as session, session.begin():
                # Transaction-local (== SET LOCAL): never leaks into other requests.
                await session.execute(apply_settings_statement(vector_search_settings(k)))
                rows = await session.execute(statement)
                return [_to_result(row) for row in rows.all()]

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        _check_k(k)
        if not _WORD_RE.search(query):
            return []  # nothing to search for; do not touch the database
        statement = build_text_search_statement(query, filters, k)
        async with _infrastructure_errors("text_search"):
            async with self._session_factory() as session:
                rows = await session.execute(statement)
                return [_to_result(row) for row in rows.all()]
