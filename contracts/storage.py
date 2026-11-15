"""Storage & Indexing contracts (owner: Raizy): Embedder, SearchIndex and their errors.

Only the READ side of the index lives here. It depends on Filters and SearchResult
(contracts.retrieval), which are already agreed.

The WRITE side (IndexWriter: upsert_document, delete_document, get_document_hashes) and
IndexStore = SearchIndex + IndexWriter depend on Document and Chunk. They will be added
here once Document and Chunk are in contracts/ (owner: Devora, D11). Until then they live
in src/kc/indexing/_provisional_contracts.py. This split keeps the open decision about
Document/Chunk away from consumers that only search (HybridRetriever).
"""

from typing import Literal, Protocol, runtime_checkable

from contracts.retrieval import Filters, SearchResult

# What a text is: a user question ("query") or indexed content ("document").
InputType = Literal["query", "document"]


class EmbeddingError(RuntimeError):
    """The embedding provider failed after the embedder's internal retries were exhausted."""


class IndexStoreError(RuntimeError):
    """Infrastructure failure inside an index (connection, timeout, database error).

    Implementations wrap the original error with `raise IndexStoreError(...) from e`, so it
    stays available in `__cause__`.
    """


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors.

    Guarantees for every implementation:
    - returns exactly one vector per input text, in the same order;
    - every vector has length `dimension` and is L2-normalized (unit length), so cosine
      similarity equals the dot product;
    - `embed([])` returns `[]`;
    - batching, retries and rate limits are handled INSIDE the embedder; callers may pass
      any number of texts;
    - never returns partial results.

    `model_name` must identify the model AND its version unambiguously (e.g.
    "BAAI/bge-m3@5617a9f"), because changing it requires a full re-index; callers that use
    it as a key for a cache or for the re-index guard can rely on that.

    `input_type`: callers MUST pass it (it is required and keyword-only, so mypy catches
    a missing one): "query" when searching, "document" when indexing. An implementation
    MAY ignore it if its model does not distinguish the two (e.g. bge-m3).

    Concurrency: one instance must be safe to use from several concurrent async calls.
    An implementation may run the calls one after another (e.g. with a Lock), but must
    never mix up their results.

    Errors:
    - EmbeddingError: raised only after the embedder's internal retries have failed.
    - ValueError: invalid input (a bug in the caller); NOT wrapped.
    - asyncio.CancelledError: never wrapped (cancellation must propagate).
    - Implementations always raise a single exception, never an ExceptionGroup.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]: ...


@runtime_checkable
class SearchIndex(Protocol):
    """Read side of the index: vector search and full-text search over chunks.

    Results (both methods):
    - at most `k` results, sorted by `score` descending; equal scores are ordered by
      `chunk_id` ascending, so the order is deterministic;
    - higher score is better, but scores are comparable ONLY within one method
      (vector_search: cosine similarity in [-1, 1]; text_search: text rank), so fusion
      must use ranks (e.g. RRF), not raw scores;
    - `filters` are applied BEFORE choosing the top k, so k counts only matching chunks;
      a search may return fewer than k results;
    - `k` is keyword-only; k <= 0 raises ValueError.

    vector_search: `vector` must have the index dimension, otherwise ValueError.
    text_search: `query` is raw user text; parsing it is the index's job.

    Concurrency: one instance must be safe to use from several concurrent async calls
    (one instance lives for the whole application). A search running while a document is
    being updated sees either its old or its new version, never a mix.

    No consistency guarantee across separate calls: each call on its own sees a complete
    version of every document, but two calls (e.g. vector_search and text_search in the
    same request) may see different versions if an update happened between them. Every
    SearchResult carries its own text and URL, so it stands on its own.

    Errors:
    - ValueError: invalid input (a bug in the caller, e.g. k <= 0, wrong vector
      dimension); NOT wrapped.
    - IndexStoreError: infrastructure failure (connection, timeout, database error),
      wrapped with `raise IndexStoreError(...) from e`.
    - No retries inside the index: whether to retry is the caller's decision.
    - asyncio.CancelledError: never wrapped (cancellation must propagate).
    - Implementations always raise a single exception, never an ExceptionGroup.
    """

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]: ...

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]: ...
