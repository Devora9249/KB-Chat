"""TEMPORARY contract types that still wait for Document/Chunk in contracts/ (D11, Devora).

Everything else is official and must be imported from contracts/ directly:
- contracts.storage: Embedder, SearchIndex, InputType, EmbeddingError, IndexStoreError
- contracts.retrieval: Filters, SearchResult
- contracts.common: Product, DocType, Version

TODO(D11): once Devora adds Document and Chunk to contracts/, move IndexWriter and
IndexStore to contracts/storage.py and delete this file.

Rules:
- Import ONLY from inside `src/kc/indexing/` and from tests/indexing (Raizy's code).
  Other domains must use contracts/ only (consumers that only search need SearchIndex).
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from contracts.common import DocType, Product, Version
from contracts.storage import SearchIndex

__all__ = ["Chunk", "Document", "IndexStore", "IndexWriter"]


# =====================================================================================
# Document / Chunk: fields as in CLAUDE.md, types aligned with contracts.common.
# Aligned with contracts.common; still provisional until Devora adds Chunk/Document to
# contracts/ (D11).
# =====================================================================================
class Document(BaseModel):
    url: str
    title: str
    breadcrumb: list[str]
    product: Product
    version: Version | None
    doc_type: DocType
    content_md: str
    content_hash: str


class Chunk(BaseModel):
    id: str
    document_url: str
    text: str
    heading_path: list[str]
    product: Product
    version: Version | None
    doc_type: DocType


# =====================================================================================
# Write side of the index (draft D6-D7). Moves to contracts/storage.py with Document/Chunk.
# =====================================================================================


@runtime_checkable
class IndexWriter(Protocol):
    """Write side of the index.

    - upsert_document atomically replaces ALL chunks of the document (chunks that
      disappeared on re-crawl are removed) and keeps title / content_hash. Raises
      ValueError if len(chunks) != len(vectors), if a vector has the wrong dimension, or if
      a chunk's document_url != document.url; on ValueError nothing is changed.
    - chunk.id must be unique across the WHOLE index (D12): Search & Ranking fuses results
      by chunk_id. A duplicate id inside one call, or an id already used by a chunk of
      ANOTHER document, is a data error -> ValueError, and nothing is changed. Re-upserting
      a document with its own existing ids is allowed (that is a normal update).
    - delete_document is a no-op if the document does not exist.
    - get_document_hashes returns url -> content_hash of everything indexed.

    Concurrency and errors: same rules as contracts.storage.SearchIndex (safe for
    concurrent async calls; ValueError not wrapped; infrastructure failures wrapped in
    IndexStoreError with `raise ... from e`; no retries; CancelledError never wrapped).

    Future PgIndexStore: receives an async_sessionmaker in its constructor (the Engine
    is created and disposed by the application), opens a short session per call, and
    runs upsert_document in a single transaction.
    """

    async def upsert_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None: ...

    async def delete_document(self, url: str) -> None: ...

    async def get_document_hashes(self) -> dict[str, str]: ...


@runtime_checkable
class IndexStore(SearchIndex, IndexWriter, Protocol):
    """Full index: search (contracts.storage.SearchIndex) + write (IndexWriter).

    A search running during upsert_document of the same document sees either the old or
    the new version, never a mix.
    """
