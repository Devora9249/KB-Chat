"""Retrieval metrics: pure functions, no I/O, so they are easy to test and reuse.

Metrics are computed at the *article* (URL) level, not the chunk level: the Golden Set
lists the articles that answer a question (`EvalCase.expected_urls`), while the Retriever
returns chunks, and several chunks may come from the same article. So the ranked chunk
list is first collapsed into a ranked list of unique URLs (see `ranked_unique_urls`).

All functions take that ranked URL list and the expected URLs. They are defined only for
positive cases (non-empty `expected_urls`); a negative case has nothing to find, so
calling them with an empty expected list raises ValueError instead of returning a
misleading number.
"""

from collections.abc import Iterable, Sequence

from contracts.retrieval import SearchResult


def ranked_unique_urls(results: Iterable[SearchResult]) -> list[str]:
    """Article URLs in rank order, each once (first occurrence keeps its rank)."""
    return list(dict.fromkeys(result.url for result in results))


def _require_expected(expected_urls: Sequence[str]) -> set[str]:
    if not expected_urls:
        raise ValueError("metric is undefined for a negative case (empty expected_urls)")
    return set(expected_urls)


def hit(ranked_urls: Sequence[str], expected_urls: Sequence[str]) -> bool:
    """True if at least one expected article was retrieved (averaged over cases: Hit Rate)."""
    expected = _require_expected(expected_urls)
    return any(url in expected for url in ranked_urls)


def recall(ranked_urls: Sequence[str], expected_urls: Sequence[str]) -> float:
    """Fraction of the expected articles that were retrieved (0.0 to 1.0)."""
    expected = _require_expected(expected_urls)
    return len(expected & set(ranked_urls)) / len(expected)


def reciprocal_rank(ranked_urls: Sequence[str], expected_urls: Sequence[str]) -> float:
    """1 / rank of the first expected article (rank starts at 1); 0.0 if none was retrieved.

    Averaged over cases this is MRR (Mean Reciprocal Rank): it rewards putting the right
    article near the top, not just somewhere in the results.
    """
    expected = _require_expected(expected_urls)
    for rank, url in enumerate(ranked_urls, start=1):
        if url in expected:
            return 1.0 / rank
    return 0.0
