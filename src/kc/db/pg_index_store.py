"""PostgreSQL + pgvector implementation of the index - WRITE side (IndexWriter).

The search side (vector_search, text_search) comes in the next step.

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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from sqlalchemy import delete, insert, select
from sqlalchemy import exc as sa_exc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from contracts.storage import IndexStoreError
from kc.db.embedding_config import EmbeddingConfigMismatchError, ensure_embedding_config
from kc.db.models import EMBEDDING_DIMENSION, ChunkRow, DocumentRow
from kc.indexing._provisional_contracts import Chunk, Document
from kc.indexing.upsert_validation import validate_upsert

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
    """IndexWriter backed by PostgreSQL + pgvector (search side: next step)."""

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
