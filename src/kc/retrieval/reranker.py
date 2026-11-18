"""Reranker protocol (internal to Search & Ranking).

A reranker takes the query and the candidates that retrieval already found, scores each
candidate against the query again (a real one uses a cross-encoder, e.g. Voyage rerank or
bge-reranker, which reads the query and the text TOGETHER and is more precise than
embeddings), and returns the best ones in a new order.

Internal on purpose (like `kc.generation.answerer.Answerer`): only Search & Ranking calls
it; other domains keep using `contracts.retrieval.Retriever`. Not wired into
HybridRetriever yet; error handling (infrastructure failures, timeouts, fallback) is
decided in the integration step.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contracts.retrieval import SearchResult


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        k: int,
    ) -> list[SearchResult]:
        """Return up to `k` of `results`, re-scored against `query`, best first.

        Guarantees for every implementation:
        - only results from the input (matched by `chunk_id`);
        - no duplicate `chunk_id` in the output (which copy is kept when the input
          repeats a `chunk_id` is up to the implementation; Retriever output never does);
        - at most `k` results (fewer when the input is shorter);
        - sorted by `score` descending, ties by `chunk_id` ascending (deterministic);
        - every field except `score` is preserved as in the input; `score` is the
          reranker's score (comparable only within this one output list);
        - returns new objects; never modifies `results` or the objects in it;
        - empty `results` -> [].

        Errors:
        - `k` must be a positive int: a non-int `k` (including bool) raises TypeError;
          k < 1 raises ValueError. Checked before anything else.
        - asyncio.CancelledError is never caught or wrapped.
        - Infrastructure failures of real implementations: to be defined when the
          reranker is integrated (not part of this protocol yet).
        """
        ...
