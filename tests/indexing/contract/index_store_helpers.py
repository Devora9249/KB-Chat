"""Shared helpers for the SearchIndex / IndexStore contract tests (owner: Raizy)."""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

from contracts.common import DocType, Product, Version
from contracts.storage import SearchIndex
from kc.indexing._provisional_contracts import Chunk, Document, IndexStore, IndexWriter

DIM = 3

# A document with its chunks and one vector per chunk, ready to be indexed.
IndexedDocument = tuple[Document, list[Chunk], list[list[float]]]

# A factory creates a NEW, EMPTY store with the given vector dimension.
StoreFactory = Callable[[int], IndexStore]


class SearchIndexFactory(Protocol):
    """Creates a NEW SearchIndex that already contains `documents` (read-only use)."""

    def __call__(self, documents: list[IndexedDocument], dimension: int = DIM) -> SearchIndex: ...


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    # Selector loop: psycopg async (PgIndexStore) does not support Windows' default Proactor
    # loop; on Linux this is the default loop anyway.
    return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)


# What a writer actually stored, read back WITHOUT the search side:
# (chunk id, document_url, text, vector).
StoredChunk = tuple[str, str, str, tuple[float, ...]]


@dataclass
class WriterUnderTest:
    """An IndexWriter plus an observer that reads back what it stored (sorted by id)."""

    writer: IndexWriter
    dimension: int
    stored: Callable[[], list[StoredChunk]]

    def unit(self, axis: int = 0) -> list[float]:
        """A unit vector of this writer's dimension (1.0 at `axis`)."""
        vector = [0.0] * self.dimension
        vector[axis] = 1.0
        return vector


def make_doc(url: str, *, title: str = "Title", content_hash: str = "h1") -> Document:
    return Document(
        url=url,
        title=title,
        breadcrumb=["Home"],
        product="sequence",
        version="9",
        doc_type="how-to",
        content_md="...",
        content_hash=content_hash,
    )


def make_chunk(
    chunk_id: str,
    url: str,
    text: str = "text",
    *,
    product: Product = "sequence",
    version: Version | None = "9",
    doc_type: DocType = "how-to",
    heading_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_url=url,
        text=text,
        heading_path=heading_path if heading_path is not None else ["H1"],
        product=product,
        version=version,
        doc_type=doc_type,
    )


def snapshot(w: WriterUnderTest) -> tuple[dict[str, str], list[StoredChunk]]:
    """Everything a writer stored: document hashes + every chunk with its vector."""
    return run(w.writer.get_document_hashes()), w.stored()
