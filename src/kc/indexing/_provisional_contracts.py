"""Single import point for contract types used by Storage & Indexing.

Some types already come from the official `contracts/` package; the rest are TEMPORARY
stand-ins until their owners add them there.

TODO(D11): once Document/Chunk (Devora) and Embedder/IndexStore/EmbeddingError (Raizy)
are in `contracts/`, replace the definitions below with imports and delete this file.

Rules:
- Import ONLY from inside `src/kc/indexing/` and from tests/indexing (Raizy's code).
  Other domains must import from `contracts/` directly.
- All Storage & Indexing code imports contract types from HERE, so each switch to
  `contracts/` is a one-line change per name.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

# =====================================================================================
# Official contracts (already in contracts/), re-exported
# =====================================================================================
from contracts.common import DocType, Product, Version
from contracts.retrieval import Filters, SearchResult

__all__ = [
    "Chunk",
    "DocType",
    "Document",
    "Embedder",
    "EmbeddingError",
    "Filters",
    "IndexStore",
    "IndexStoreError",
    "InputType",
    "Product",
    "SearchResult",
    "Version",
]


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
# From my storage contract draft (D1-D9), not yet approved by the team
# =====================================================================================

# Required, keyword-only argument of Embedder.embed (D2).
InputType = Literal["query", "document"]


class EmbeddingError(RuntimeError):
    """The embedding provider failed after the embedder's internal retries were exhausted."""


class IndexStoreError(RuntimeError):
    """Infrastructure failure inside an IndexStore (connection, timeout, database error).

    Implementations wrap the original error with `raise IndexStoreError(...) from e`, so it
    stays available in `__cause__`.
    """


@runtime_checkable
class Embedder(Protocol):
    """See storage contract draft D1-D5: one unit-length vector per text, same order.

    Concurrency: one instance must be safe to use from several concurrent async calls.
    An implementation may run the calls one after another (e.g. with a Lock), but must
    never mix up their results.

    Errors:
    - EmbeddingError: raised only after the embedder's internal retries have failed.
    - ValueError: invalid input (a bug in the caller); NOT wrapped.
    - asyncio.CancelledError: never wrapped (cancellation must propagate).
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]: ...


@runtime_checkable
class IndexStore(Protocol):
    """See storage contract draft D6-D9.

    Concurrency: one instance must be safe to use from several concurrent async calls
    (one IndexStore lives for the whole application). A search running during an
    upsert_document of the same document sees either the old or the new version, never
    a mix.

    Future PgIndexStore: receives an async_sessionmaker in its constructor (the Engine
    is created and disposed by the application), opens a short session per call, and
    runs upsert_document in a single transaction.

    Errors:
    - ValueError: invalid input (a bug in the caller, e.g. k <= 0, wrong vector
      dimension, chunk of another document); NOT wrapped.
    - IndexStoreError: infrastructure failure (connection, timeout, database error).
      Implementations wrap it with `raise IndexStoreError(...) from e`, so the original
      error is kept in `__cause__`.
    - No retries inside an IndexStore: whether to retry is the caller's decision.
    - asyncio.CancelledError: never wrapped (cancellation must propagate).
    """

    async def upsert_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None: ...

    async def delete_document(self, url: str) -> None: ...

    async def get_document_hashes(self) -> dict[str, str]: ...

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]: ...

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]: ...
