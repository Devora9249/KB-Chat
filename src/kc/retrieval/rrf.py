"""Reciprocal Rank Fusion (RRF): merge vector-search and text-search results by rank.

Vector scores (cosine) and text scores (ts_rank) live on different scales and cannot be
added. RRF ignores them and uses only each item's position: every list an item appears in
contributes 1 / (rrf_k + rank), with rank starting at 1. An item ranked well in both lists
beats an item that tops only one of them.
"""

from collections.abc import Sequence
from fractions import Fraction
from typing import Final

from contracts.retrieval import SearchResult

RRF_K: Final = 60


def reciprocal_rank_fusion(
    vector_results: Sequence[SearchResult],
    text_results: Sequence[SearchResult],
    *,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse two ranked result lists into one list with RRF scores.

    Behaviour:
    - Rank = 1-based position in the input list as given (inputs must already be sorted
      best first, per the Retriever/IndexStore contracts). Input `score` values are ignored.
    - Score = sum of 1 / (rrf_k + rank) over the lists the chunk appears in.
    - Each chunk_id appears once in the output.
    - Duplicate chunk_id inside ONE list: only the first occurrence is scored; original
      positions are kept (the duplicate still occupies its slot), so [A, A, B] scores
      A = 1/(rrf_k+1) and B = 1/(rrf_k+3).
    - Same chunk_id with different metadata: the output keeps the FIRST SearchResult from
      `vector_results`; if the chunk is not there, the first one from `text_results`.
      Other occurrences only contribute to the score.
    - Sorted by score descending, ties by chunk_id ascending. Scores are summed exactly
      (Fraction), so true ties are detected and the order is deterministic.
    - No cut to k (the caller does that). Outputs are new deep copies; inputs are not
      modified.

    Raises:
        TypeError: `rrf_k` is not an int (bool is rejected too).
        ValueError: `rrf_k` < 1.
    """
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
        raise TypeError(f"rrf_k must be an int, got {type(rrf_k).__name__}")
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

    scores: dict[str, Fraction] = {}
    kept: dict[str, SearchResult] = {}  # first occurrence wins; vector list is read first

    for results in (vector_results, text_results):
        seen_in_list: set[str] = set()
        for rank, item in enumerate(results, start=1):
            if item.chunk_id in seen_in_list:
                continue  # duplicate within this list: keeps its slot, adds no score
            seen_in_list.add(item.chunk_id)
            scores[item.chunk_id] = scores.get(item.chunk_id, Fraction(0)) + Fraction(
                1, rrf_k + rank
            )
            kept.setdefault(item.chunk_id, item)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        kept[chunk_id].model_copy(update={"score": float(scores[chunk_id])}, deep=True)
        for chunk_id in ordered
    ]
