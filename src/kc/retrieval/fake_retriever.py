"""Deterministic, offline Retriever for tests and local development.

Satisfies the `Retriever` protocol from `contracts.retrieval` structurally (no inheritance):
    async def search(self, query, filters=None, k=5) -> list[SearchResult]

The default corpus is MOCK DATA ONLY (see `_fake_data.py`): invented content,
not real Knowledge Center pages.

Scoring mirrors `InMemoryIndexStore.text_search` (Raizy,
src/kc/indexing/in_memory_index_store.py) so both fakes behave the same.
If that algorithm changes, update this one too:
- tokens: `\\w+` on lowercased text;
- query terms are a set (a word repeated in the query counts once);
- score = number of tokens in the item's `text` (not title / heading_path) that are
  query terms; items with score 0 are left out;
- sorted by score descending, ties by chunk_id ascending, then cut to k.
The only intended difference: FakeRetriever applies `Filters` (contract semantics).
"""

import re
from collections.abc import Sequence

from contracts.retrieval import Filters, SearchResult
from kc.retrieval._fake_data import DEFAULT_CORPUS

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _passes_filters(item: SearchResult, filters: Filters | None) -> bool:
    """AND across fields, OR within a field. None means no filter on that field."""
    if filters is None:
        return True
    if filters.products is not None and item.product not in filters.products:
        return False
    # An item with version=None never matches, because None is not in the list.
    if filters.versions is not None and item.version not in filters.versions:
        return False
    if filters.doc_types is not None and item.doc_type not in filters.doc_types:
        return False
    return not (
        filters.exclude_doc_types is not None and item.doc_type in filters.exclude_doc_types
    )


class FakeRetriever:
    """Keyword-overlap retriever over an in-memory corpus of SearchResult items."""

    def __init__(self, corpus: Sequence[SearchResult] | None = None) -> None:
        source = DEFAULT_CORPUS if corpus is None else corpus
        seen: set[str] = set()
        for item in source:
            if item.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id in corpus: {item.chunk_id}")
            seen.add(item.chunk_id)
        # Deep copies: later changes to the caller's objects do not leak into the corpus.
        self._corpus: tuple[SearchResult, ...] = tuple(
            item.model_copy(deep=True) for item in source
        )

    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        """Return up to `k` results, best first (see module docstring for scoring).

        `k` must be a positive int (contract): a non-int `k` (including bool) raises
        TypeError; k < 1 raises ValueError. Checked first, before the query is looked at.

        Fake-specific decision (NOT required by the shared contract): a query with no
        word tokens (e.g. "" or "  ?! ") returns []. Same as InMemoryIndexStore.text_search.
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        query_terms = set(_tokens(query))
        if not query_terms:
            return []

        scored: list[tuple[float, SearchResult]] = []
        for item in self._corpus:
            if not _passes_filters(item, filters):
                continue
            score = float(sum(1 for token in _tokens(item.text) if token in query_terms))
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        # New deep copies per call: mutating a returned result never affects later searches.
        # model_copy(update=...) skips validation; safe here, score is a finite float.
        return [item.model_copy(update={"score": score}, deep=True) for score, item in scored[:k]]
