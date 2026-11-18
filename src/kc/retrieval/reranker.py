"""Reranker protocol (internal to Search & Ranking).

A reranker takes the query and the candidates that retrieval already found, scores each
candidate against the query again (a real one uses a cross-encoder, e.g. Voyage rerank or
bge-reranker, which reads the query and the text TOGETHER and is more precise than
embeddings), and returns the best ones in a new order.

Internal on purpose (like `kc.generation.answerer.Answerer`): only Search & Ranking calls
it; other domains keep using `contracts.retrieval.Retriever`. Not wired into
HybridRetriever yet. Planned for F9 PR B (not implemented yet): on RerankError or
TimeoutError, search falls back to the RRF ranking.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contracts.retrieval import SearchResult


class RerankError(RuntimeError):
    """Reranking failed because of infrastructure, not because of the caller's input.

    Real rerankers raise it for known infrastructure failures (provider unavailable,
    network error, rate limit, ...), with `raise RerankError(...) from e` so the original
    error stays in `__cause__`. Never used for programming errors.
    """


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        k: int,
    ) -> list[SearchResult]:
        """Return the best `k` of `results`, re-scored against `query`, best first.

        A reranker REORDERS candidates: it never drops a candidate because of its score;
        only the cut to `k` removes results.

        Guarantees for every implementation:
        - only results from the input (matched by `chunk_id`);
        - no duplicate `chunk_id` in the output (which copy is kept when the input
          repeats a `chunk_id` is up to the implementation; Retriever output never does);
        - exactly min(k, number of unique input `chunk_id`s) results, so with
          k >= that number the output is a full permutation of the unique input;
        - sorted by `score` descending, ties by `chunk_id` ascending (deterministic);
        - `score` is the reranker's OWN score, finite and >= 0 (comparable only within
          this one output list); the incoming `score` is never passed through;
        - every field except `score` is preserved as in the input;
        - returns new objects; never modifies `results` or the objects in it;
        - empty `results` -> [].

        Errors:
        - `k` must be a positive int: a non-int `k` (including bool) raises TypeError;
          k < 1 raises ValueError. Checked before anything else.
        - Known infrastructure failure (provider, network, rate limit) -> RerankError,
          original error in `__cause__`.
        - Timeout -> TimeoutError. A provider/network timeout (e.g. httpx.TimeoutException,
          which is NOT a TimeoutError subclass) may be raised as TimeoutError or wrapped in
          RerankError; both mean an infrastructure failure.
        - Programming errors (ValueError, TypeError, ...) are NOT wrapped.
        - asyncio.CancelledError is never caught or wrapped.
        """
        ...
