"""Deterministic, offline Reranker for tests and local development.

Satisfies `kc.retrieval.reranker.Reranker` structurally (no inheritance).

Score source:
- default: token overlap with the query, same convention as FakeRetriever: tokens are
  `\\w+` on lowercased text, query terms are a set, and the score is the number of tokens
  in the result's `text` that are query terms. Results with score 0 are KEPT (a reranker
  reorders candidates; it does not filter them).
- `scores` mapping (chunk_id -> score): used instead of the overlap, to force a specific
  order in tests; a chunk_id missing from the mapping gets 0.0.
The incoming `score` of a result is ignored.

FakeReranker decisions (NOT required by the Reranker protocol):
- duplicate chunk_id in the input: the FIRST occurrence is kept, later ones are dropped;
- a query with no `\\w+` token does not return [] early (unlike FakeRetriever): every
  result gets score 0.0 and the normal tie-break (chunk_id ascending) decides the order.
"""

import math
import re
from collections.abc import Mapping, Sequence

from contracts.retrieval import SearchResult

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


class FakeReranker:
    """Reranks by query/text token overlap, or by an explicit chunk_id -> score mapping."""

    def __init__(self, scores: Mapping[str, float] | None = None) -> None:
        self._scores: dict[str, float] | None = None
        if scores is not None:
            checked: dict[str, float] = {}
            for chunk_id, score in scores.items():
                if isinstance(score, bool) or not isinstance(score, int | float):
                    raise TypeError(
                        f"score for {chunk_id!r} must be an int or float, "
                        f"got {type(score).__name__}"
                    )
                try:
                    # An int too large for a float (e.g. 10**400) raises OverflowError here.
                    as_float = float(score)
                except OverflowError:
                    raise ValueError(
                        f"score for {chunk_id!r} must be finite, got an int too large for a float"
                    ) from None
                if not math.isfinite(as_float):
                    raise ValueError(f"score for {chunk_id!r} must be finite, got {score}")
                checked[chunk_id] = as_float
            # A private copy: later changes to the caller's mapping have no effect.
            self._scores = checked

    async def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        k: int,
    ) -> list[SearchResult]:
        """See `Reranker.rerank`. `k` is checked first, before the input is looked at."""
        _check_k(k)

        # FakeReranker decision: first occurrence of each chunk_id only (the output must
        # not repeat an id; the protocol leaves the choice of copy to the implementation).
        unique: dict[str, SearchResult] = {}
        for result in results:
            unique.setdefault(result.chunk_id, result)

        query_terms = set(_tokens(query))
        scored = [(self._score(result, query_terms), result) for result in unique.values()]
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        # New deep copies: the caller's objects are never shared or modified.
        # model_copy(update=...) skips validation; safe here, every score is finite.
        return [result.model_copy(update={"score": s}, deep=True) for s, result in scored[:k]]

    def _score(self, result: SearchResult, query_terms: set[str]) -> float:
        if self._scores is not None:
            return self._scores.get(result.chunk_id, 0.0)
        return float(sum(1 for token in _tokens(result.text) if token in query_terms))
