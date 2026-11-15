"""Indexing pipeline: Document + Chunks -> vectors -> IndexStore.

Per document:
1. Pre-check (cheap): every chunk belongs to the document. Done BEFORE embedding so we
   never pay for embeddings the store would reject.
2. Embed all chunk texts with input_type="document" (batching/retries live inside the
   Embedder, contract D5).
3. upsert_document: atomically replaces the document in the index (contract D6).

If embedding fails (EmbeddingError) step 3 never runs, so the PREVIOUS version of the
document stays in the index unchanged: a slightly stale answer beats a missing page.

Dependencies are passed in (Dependency Injection): this module never imports a concrete
Embedder or IndexStore, only the protocols.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from kc.indexing._provisional_contracts import (
    Chunk,
    Document,
    Embedder,
    EmbeddingError,
    IndexStore,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """Outcome of index_documents: which URLs were indexed and which failed (url -> reason)."""

    indexed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


async def index_document(
    document: Document,
    chunks: list[Chunk],
    *,
    embedder: Embedder,
    store: IndexStore,
) -> None:
    """Embed and store one document. Raises EmbeddingError / ValueError; the index is then
    left exactly as it was before the call."""
    for chunk in chunks:
        if chunk.document_url != document.url:
            raise ValueError(
                f"chunk {chunk.id!r} has document_url {chunk.document_url!r}, "
                f"expected {document.url!r}"
            )
    vectors = await embedder.embed([chunk.text for chunk in chunks], input_type="document")
    await store.upsert_document(document, chunks, vectors)


async def index_documents(
    items: Iterable[tuple[Document, list[Chunk]]],
    *,
    embedder: Embedder,
    store: IndexStore,
) -> IndexReport:
    """Index many documents sequentially. A failing document is recorded and skipped, so one
    bad page does not stop the run. Only EmbeddingError and ValueError are caught; anything
    else is a bug and propagates."""
    report = IndexReport()
    for document, chunks in items:
        try:
            await index_document(document, chunks, embedder=embedder, store=store)
        except (EmbeddingError, ValueError) as error:
            logger.warning("indexing failed for %s: %s", document.url, error)
            report.failed[document.url] = f"{type(error).__name__}: {error}"
        else:
            report.indexed.append(document.url)
    return report
