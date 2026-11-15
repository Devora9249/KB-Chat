"""Shared helpers for IndexStore contract tests (owner: Raizy)."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from kc.indexing._provisional_contracts import (
    Chunk,
    DocType,
    Document,
    IndexStore,
    Product,
    SearchResult,
    Version,
)

# A factory creates a NEW, EMPTY store with the given vector dimension.
StoreFactory = Callable[[int], IndexStore]

DIM = 3


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


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


def snapshot(store: IndexStore) -> tuple[dict[str, str], list[SearchResult]]:
    """Everything observable from outside: hashes + all chunks via a broad search."""
    return run(store.get_document_hashes()), run(store.vector_search([1.0, 0.0, 0.0], k=1000))
