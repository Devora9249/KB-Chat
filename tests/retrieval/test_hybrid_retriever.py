"""Tests for HybridRetriever (Search & Ranking)."""

import asyncio
import logging
import math
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from contracts.common import DocType
from contracts.retrieval import Filters, RetrievalError, Retriever, SearchResult
from contracts.storage import EmbeddingError, IndexStoreError, InputType
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RELEASE_NOTES_WEIGHT,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_TIMEOUT_S,
    HybridRetriever,
    _raise_translated,
)
from kc.retrieval.reranker import RerankError
from kc.retrieval.rrf import reciprocal_rank_fusion

# --- Local async stubs (no imports of other domains' implementations) --------------------

QUERY_VECTOR = [1.0, 0.0, 0.0]
LONG = 10.0  # "forever" for a test; such waits are always cancelled


def _r(chunk_id: str, doc_type: DocType = "how-to") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        url=f"https://knowledgecenter.gcora.genpact.com/help/{chunk_id}",
        title=f"Title {chunk_id}",
        text=f"Text {chunk_id}",
        score=0.5,
        product="sequence",
        version="9",
        doc_type=doc_type,
        heading_path=["Section"],
    )


def _rn(chunk_id: str) -> SearchResult:
    return _r(chunk_id, "release-notes")


class Behavior:
    """What one stub call does: wait (gate / delay), then raise `error` or return.

    `error_on_cancel`: raised INSTEAD of CancelledError when the call is cancelled.
    This forces "both branches failed": the other branch fails first, the TaskGroup
    cancels this one, and it fails while being cancelled.
    """

    def __init__(
        self,
        *,
        error: Exception | None = None,
        delay: float = 0.0,
        gate: asyncio.Event | None = None,
        error_on_cancel: Exception | None = None,
        cancel_delay: float = 0.0,
    ) -> None:
        self.error = error
        self.delay = delay
        self.gate = gate
        self.error_on_cancel = error_on_cancel
        self.cancel_delay = cancel_delay  # slow cleanup while being cancelled
        self.started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def run(self) -> None:
        self.started.set()
        try:
            try:
                if self.gate is not None:
                    await self.gate.wait()
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled = True
                self.cleanup_started.set()
                if self.cancel_delay:
                    await asyncio.sleep(self.cancel_delay)
                if self.error_on_cancel is not None:
                    raise self.error_on_cancel from None
                raise
            if self.error is not None:
                raise self.error
        finally:
            self.finished = True


class StubEmbedder:
    """Records every call; behavior configurable (error, delay, gate, error_on_cancel)."""

    def __init__(self, vectors: list[list[float]] | None = None, **behavior: Any) -> None:
        self._vectors = [QUERY_VECTOR] if vectors is None else vectors
        self.behavior = Behavior(**behavior)
        self.calls: list[tuple[list[str], InputType]] = []

    @property
    def model_name(self) -> str:
        return "stub-embedder"

    @property
    def dimension(self) -> int:
        return 3

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        self.calls.append((list(texts), input_type))
        await self.behavior.run()
        return [list(v) for v in self._vectors]


class StubIndex:
    """Records every call (with the exact filters object) and returns fixed lists."""

    def __init__(
        self,
        vector_results: list[SearchResult] | None = None,
        text_results: list[SearchResult] | None = None,
        *,
        vector: Behavior | None = None,
        text: Behavior | None = None,
    ) -> None:
        self._vector_results = vector_results or []
        self._text_results = text_results or []
        self.vector = vector or Behavior()
        self.text = text or Behavior()
        self.vector_calls: list[tuple[list[float], Filters | None, int]] = []
        self.text_calls: list[tuple[str, Filters | None, int]] = []

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        self.vector_calls.append((list(vector), filters, k))
        await self.vector.run()
        return [r.model_copy(deep=True) for r in self._vector_results]

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        self.text_calls.append((query, filters, k))
        await self.text.run()
        return [r.model_copy(deep=True) for r in self._text_results]


class StubReranker:
    """Records every call. Scores from `scores` (missing id -> 0.0), or a forced `output`.

    Follows the Reranker protocol unless `output` is given (used to break it on purpose).
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        behavior: Behavior | None = None,
        output: Callable[[list[SearchResult]], list[SearchResult]] | None = None,
    ) -> None:
        self._scores = scores or {}
        self.behavior = behavior or Behavior()
        self._output = output
        self.calls: list[tuple[str, list[SearchResult], int]] = []

    async def rerank(
        self, query: str, results: Sequence[SearchResult], *, k: int
    ) -> list[SearchResult]:
        pool = [r.model_copy(deep=True) for r in results]
        self.calls.append((query, pool, k))
        await self.behavior.run()
        if self._output is not None:
            return self._output(pool)
        scored = [r.model_copy(update={"score": self._scores.get(r.chunk_id, 0.0)}) for r in pool]
        return sorted(scored, key=lambda r: (-r.score, r.chunk_id))[:k]


def _make(**kwargs: Any) -> HybridRetriever:
    return HybridRetriever(StubEmbedder(), StubIndex(), **kwargs)


def _search(
    retriever: HybridRetriever, query: str, filters: Filters | None = None, k: int = 5
) -> list[SearchResult]:
    return asyncio.run(retriever.search(query, filters, k))


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.chunk_id for r in results]


def _other_tasks() -> set[asyncio.Task[Any]]:
    return asyncio.all_tasks() - {asyncio.current_task()}


def _run_and_capture(retriever: HybridRetriever, query: str = "saml") -> BaseException:
    """Run one search that must fail; return the error and check no task is left."""

    async def scenario() -> BaseException:
        try:
            result = await retriever.search(query)
        except Exception as exc:
            assert _other_tasks() == set()  # the TaskGroup waited for both branches
            return exc
        pytest.fail(f"expected an error, got {result!r}")

    return asyncio.run(scenario())


def _reachable(start: BaseException | None) -> list[BaseException]:
    """Every exception reachable via __cause__, __context__ and group members."""
    seen: dict[int, BaseException] = {}
    stack = [start]
    while stack:
        exc = stack.pop()
        if exc is None or id(exc) in seen:
            continue
        seen[id(exc)] = exc
        stack += [exc.__cause__, exc.__context__]
        if isinstance(exc, BaseExceptionGroup):
            stack += list(exc.exceptions)
    return list(seen.values())


def _group_leaves(exc: BaseException) -> list[BaseException]:
    """Leaf members of a (possibly nested) group, following group membership only."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in _group_leaves(inner)]
    return [exc]


def _contains(objects: list[BaseException], target: BaseException) -> bool:
    return any(obj is target for obj in objects)


def _assert_formats(exc: BaseException) -> None:
    text = "".join(traceback.format_exception(exc))
    assert type(exc).__name__ in text


# --- Construction ------------------------------------------------------------------------


def test_defaults() -> None:
    embedder = StubEmbedder()
    index = StubIndex()
    retriever = HybridRetriever(embedder, index)
    assert retriever._embedder is embedder
    assert retriever._index is index
    assert retriever._candidate_k == DEFAULT_CANDIDATE_K == 50
    assert retriever._timeout_s == DEFAULT_TIMEOUT_S == 10.0
    assert retriever._release_notes_weight == DEFAULT_RELEASE_NOTES_WEIGHT == 1.0
    assert retriever._reranker is None
    assert retriever._rerank_candidates == DEFAULT_RERANK_CANDIDATES == 20
    assert retriever._rerank_timeout_s == DEFAULT_RERANK_TIMEOUT_S == 3.0


def test_custom_settings_are_kept() -> None:
    retriever = _make(candidate_k=7, timeout_s=2.5)
    assert retriever._candidate_k == 7
    assert retriever._timeout_s == 2.5


# --- candidate_k -------------------------------------------------------------------------


@pytest.mark.parametrize("candidate_k", [1, 50, 1000])
def test_valid_candidate_k(candidate_k: int) -> None:
    assert _make(candidate_k=candidate_k)._candidate_k == candidate_k


@pytest.mark.parametrize("candidate_k", [True, False, 5.0, "5", None])
def test_candidate_k_wrong_type_raises_type_error(candidate_k: Any) -> None:
    with pytest.raises(TypeError, match="candidate_k"):
        _make(candidate_k=candidate_k)


@pytest.mark.parametrize("candidate_k", [0, -1])
def test_candidate_k_below_one_raises_value_error(candidate_k: int) -> None:
    with pytest.raises(ValueError, match="candidate_k"):
        _make(candidate_k=candidate_k)


# --- timeout_s ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeout_s", [1, 0.5, 10.0])
def test_valid_timeout_s(timeout_s: float) -> None:
    assert _make(timeout_s=timeout_s)._timeout_s == timeout_s


@pytest.mark.parametrize("timeout_s", [True, False, "1", None])
def test_timeout_s_wrong_type_raises_type_error(timeout_s: Any) -> None:
    with pytest.raises(TypeError, match="timeout_s"):
        _make(timeout_s=timeout_s)


@pytest.mark.parametrize(
    "timeout_s",
    [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_timeout_s_not_finite_or_not_positive_raises_value_error(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        _make(timeout_s=timeout_s)


# --- release_notes_weight ----------------------------------------------------------------


@pytest.mark.parametrize("weight", [1, 1.0, 0.5, 1e-9])
def test_valid_release_notes_weight(weight: float) -> None:
    assert _make(release_notes_weight=weight)._release_notes_weight == weight


@pytest.mark.parametrize("weight", [True, False, "0.5", None])
def test_release_notes_weight_wrong_type_raises_type_error(weight: Any) -> None:
    with pytest.raises(TypeError, match="release_notes_weight"):
        _make(release_notes_weight=weight)


@pytest.mark.parametrize(
    "weight",
    [0, 0.0, -0.1, 1.0000001, 2, float("nan"), float("inf"), float("-inf")],
)
def test_release_notes_weight_out_of_range_raises_value_error(weight: float) -> None:
    with pytest.raises(ValueError, match="release_notes_weight"):
        _make(release_notes_weight=weight)


# --- search: contract and input ----------------------------------------------------------


def test_satisfies_retriever_protocol() -> None:
    retriever: Retriever = HybridRetriever(StubEmbedder(), StubIndex([_r("a")]))
    assert _ids(asyncio.run(retriever.search("saml"))) == ["a"]


@pytest.mark.parametrize("k", [True, False, 5.0, "5", None])
def test_k_wrong_type_raises_type_error_without_calls(k: Any) -> None:
    embedder, index = StubEmbedder(), StubIndex()
    with pytest.raises(TypeError, match="k must be an int"):
        _search(HybridRetriever(embedder, index), "saml", k=k)
    assert embedder.calls == []
    assert index.vector_calls == [] and index.text_calls == []


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises_value_error_without_calls(k: int) -> None:
    embedder, index = StubEmbedder(), StubIndex()
    with pytest.raises(ValueError, match="k must be >= 1"):
        _search(HybridRetriever(embedder, index), "saml", k=k)
    assert embedder.calls == []
    assert index.vector_calls == [] and index.text_calls == []


@pytest.mark.parametrize("query", ["", "   ", "?! ...", "-- //"])
def test_query_without_tokens_returns_empty_without_calls(query: str) -> None:
    embedder, index = StubEmbedder(), StubIndex([_r("a")], [_r("b")])
    assert _search(HybridRetriever(embedder, index), query) == []
    assert embedder.calls == []
    assert index.vector_calls == [] and index.text_calls == []


@pytest.mark.parametrize("query", ["_", "SAML", "é"])
def test_single_token_query_runs_full_search(query: str) -> None:
    # "_" is a token: `\w` includes the underscore (same as the other implementations).
    embedder, index = StubEmbedder(), StubIndex([_r("a")], [_r("b")])
    results = _search(HybridRetriever(embedder, index), query)
    assert sorted(_ids(results)) == ["a", "b"]
    assert len(embedder.calls) == 1
    assert len(index.vector_calls) == 1 and len(index.text_calls) == 1


# --- search: flow and parameters ---------------------------------------------------------


def test_embed_called_once_with_query_input_type() -> None:
    embedder = StubEmbedder()
    _search(HybridRetriever(embedder, StubIndex()), "configure saml")
    assert embedder.calls == [(["configure saml"], "query")]


def test_vector_search_gets_the_embedding_and_text_search_gets_the_query() -> None:
    vector = [0.0, 0.6, 0.8]
    index = StubIndex()
    _search(HybridRetriever(StubEmbedder([vector]), index), "configure saml")
    assert index.vector_calls[0][0] == vector
    assert index.text_calls[0][0] == "configure saml"


@pytest.mark.parametrize("filters", [None, Filters(products=["sequence"], versions=["9"])])
def test_same_filters_object_is_passed_to_both_searches(filters: Filters | None) -> None:
    index = StubIndex()
    _search(HybridRetriever(StubEmbedder(), index), "saml", filters)
    assert index.vector_calls[0][1] is filters
    assert index.text_calls[0][1] is filters


@pytest.mark.parametrize(
    ("candidate_k", "k", "expected_n"),
    [(50, 5, 50), (50, 80, 80), (10, 3, 10), (1, 1, 1)],
)
def test_candidates_requested_are_max_of_candidate_k_and_k(
    candidate_k: int, k: int, expected_n: int
) -> None:
    index = StubIndex()
    _search(HybridRetriever(StubEmbedder(), index, candidate_k=candidate_k), "saml", k=k)
    assert index.vector_calls[0][2] == expected_n
    assert index.text_calls[0][2] == expected_n


def test_text_search_starts_while_embed_is_still_blocked() -> None:
    async def scenario() -> list[SearchResult]:
        release_embed = asyncio.Event()
        embedder = StubEmbedder(gate=release_embed)
        index = StubIndex([_r("a")], [_r("b")])
        task = asyncio.create_task(HybridRetriever(embedder, index).search("saml"))

        await asyncio.wait_for(index.text.started.wait(), timeout=1.0)
        assert len(embedder.calls) == 1  # embed has started ...
        assert index.vector_calls == []  # ... and is still blocked
        assert len(index.text_calls) == 1  # text_search did not wait for it

        release_embed.set()
        return await asyncio.wait_for(task, timeout=1.0)

    assert sorted(_ids(asyncio.run(scenario()))) == ["a", "b"]


# --- search: fusion and cut to k ---------------------------------------------------------


def test_results_are_fused_with_rrf() -> None:
    vector = [_r("a"), _r("b"), _r("c")]
    text = [_r("c"), _r("d"), _r("a")]
    results = _search(HybridRetriever(StubEmbedder(), StubIndex(vector, text)), "saml", k=10)
    # a: vector 1 + text 3; c: vector 3 + text 1 (tie -> chunk_id); then b, d (tie).
    assert _ids(results) == ["a", "c", "b", "d"]
    assert results == reciprocal_rank_fusion(vector, text)


def test_item_in_both_lists_beats_top_of_one_list() -> None:
    vector = [_r("x"), _r("both")]
    text = [_r("y"), _r("both")]
    results = _search(HybridRetriever(StubEmbedder(), StubIndex(vector, text)), "saml", k=1)
    assert _ids(results) == ["both"]


def test_returns_top_k_in_rrf_order() -> None:
    vector = [_r("a"), _r("b"), _r("c")]
    text = [_r("c"), _r("d"), _r("a")]
    results = _search(HybridRetriever(StubEmbedder(), StubIndex(vector, text)), "saml", k=2)
    assert _ids(results) == ["a", "c"]
    assert results == reciprocal_rank_fusion(vector, text)[:2]


def test_fewer_candidates_than_k_returns_all() -> None:
    index = StubIndex([_r("a")], [_r("b")])
    assert sorted(_ids(_search(HybridRetriever(StubEmbedder(), index), "saml", k=10))) == [
        "a",
        "b",
    ]


def test_both_searches_empty_returns_empty_list() -> None:
    assert _search(HybridRetriever(StubEmbedder(), StubIndex()), "saml") == []


def test_output_is_sorted_with_unique_chunk_ids() -> None:
    vector = [_r("b"), _r("a"), _r("c")]
    text = [_r("a"), _r("b"), _r("d")]
    results = _search(HybridRetriever(StubEmbedder(), StubIndex(vector, text)), "saml", k=10)
    assert len(_ids(results)) == len(set(_ids(results)))
    assert results == sorted(results, key=lambda r: (-r.score, r.chunk_id))


# --- release notes weighting (after RRF, before the cut to k) ------------------------------
#
# Single-list RRF scores: rank 1 -> 1/61 ~ 0.01639, rank 2 -> 1/62 ~ 0.01613.


def _weighted(
    vector: list[SearchResult], text: list[SearchResult], weight: float, k: int = 10
) -> list[SearchResult]:
    index = StubIndex(vector, text)
    return _search(HybridRetriever(StubEmbedder(), index, release_notes_weight=weight), "q", k=k)


@pytest.mark.parametrize("weight", [None, 1.0])
def test_weight_one_keeps_the_rrf_result_exactly(weight: float | None) -> None:
    vector = [_rn("a"), _r("b"), _r("c")]
    text = [_r("c"), _rn("a"), _rn("d")]
    retriever = (
        HybridRetriever(StubEmbedder(), StubIndex(vector, text))  # default weight
        if weight is None
        else HybridRetriever(StubEmbedder(), StubIndex(vector, text), release_notes_weight=weight)
    )
    assert _search(retriever, "q", k=10) == reciprocal_rank_fusion(vector, text)


def test_only_release_note_scores_are_multiplied() -> None:
    vector = [_rn("a"), _r("b"), _rn("c")]
    text = [_r("b"), _r("d")]
    fused = {r.chunk_id: r for r in reciprocal_rank_fusion(vector, text)}

    results = _weighted(vector, text, weight=0.5)

    assert len(results) == len(fused)
    for r in results:
        original = fused[r.chunk_id]
        if r.doc_type == "release-notes":
            assert r.score == original.score * 0.5
        else:
            assert r.score == original.score
        assert r.model_dump(exclude={"score"}) == original.model_dump(exclude={"score"})


def test_lower_ranked_how_to_moves_above_a_release_note() -> None:
    vector = [_rn("rn"), _r("howto")]  # rn 1/61, howto 1/62
    assert _ids(_weighted(vector, [], weight=1.0)) == ["rn", "howto"]  # precondition
    # 0.9/61 ~ 0.01475 < 1/62 ~ 0.01613
    assert _ids(_weighted(vector, [], weight=0.9)) == ["howto", "rn"]


def test_highly_ranked_release_note_can_still_rank_first() -> None:
    vector = [_rn("rn"), _r("x")]
    text = [_r("y"), _rn("rn")]
    # rn: (1/61 + 1/62) * 0.6 ~ 0.01951 beats y, the best single-list result: 1/61 ~ 0.01639
    assert _ids(_weighted(vector, text, weight=0.6)) == ["rn", "y", "x"]


def test_weighting_happens_before_the_cut_to_k() -> None:
    vector = [_rn("rn"), _r("howto")]
    assert _ids(_weighted(vector, [], weight=1.0, k=1)) == ["rn"]  # precondition
    assert _ids(_weighted(vector, [], weight=0.9, k=1)) == ["howto"]


def test_equal_weighted_scores_are_ordered_by_chunk_id() -> None:
    # b (vector rank 1) and a (text rank 1) both get 1/61, then the same weight 0.5;
    # c (vector rank 2, how-to) keeps 1/62 and moves above both.
    results = _weighted([_rn("b"), _r("c")], [_rn("a")], weight=0.5)
    scores = {r.chunk_id: r.score for r in results}
    assert scores["a"] == scores["b"]
    assert _ids(results) == ["c", "a", "b"]  # the tie a / b is broken by chunk_id


def test_release_notes_only_keep_the_rrf_order() -> None:
    vector = [_rn("a"), _rn("b"), _rn("c")]
    text = [_rn("c"), _rn("d")]
    fused = reciprocal_rank_fusion(vector, text)

    results = _weighted(vector, text, weight=0.5)

    assert _ids(results) == _ids(fused)
    assert [r.score for r in results] == [r.score * 0.5 for r in fused]


# --- errors: one infrastructure failure -> RetrievalError(cause=original) ----------------


def test_embedding_error_becomes_retrieval_error_without_fallback() -> None:
    err = EmbeddingError("provider down")
    index = StubIndex(text_results=[_r("b")], text=Behavior(delay=LONG))
    exc = _run_and_capture(HybridRetriever(StubEmbedder(error=err), index), "secret-query")
    assert type(exc) is RetrievalError
    assert exc.__cause__ is err
    assert "secret-query" not in str(exc)  # the query text is never put in the message
    assert index.vector_calls == []  # no vector search without a vector
    assert index.text.cancelled  # the text branch was cancelled, not returned alone


def test_index_error_in_vector_search_becomes_retrieval_error() -> None:
    err = IndexStoreError("vector down")
    index = StubIndex([_r("a")], [_r("b")], vector=Behavior(error=err), text=Behavior(delay=LONG))
    exc = _run_and_capture(HybridRetriever(StubEmbedder(), index))
    assert type(exc) is RetrievalError
    assert exc.__cause__ is err
    assert "vector" in str(exc)
    assert index.text.cancelled


def test_index_error_in_text_search_becomes_retrieval_error() -> None:
    err = IndexStoreError("text down")
    embedder = StubEmbedder(delay=LONG)
    index = StubIndex([_r("a")], [_r("b")], text=Behavior(error=err))
    exc = _run_and_capture(HybridRetriever(embedder, index))
    assert type(exc) is RetrievalError
    assert exc.__cause__ is err
    assert "text" in str(exc)
    assert embedder.behavior.cancelled


@pytest.mark.parametrize("branch", ["embed", "vector", "text"])
def test_timeout_error_from_a_dependency_becomes_retrieval_error(branch: str) -> None:
    err = TimeoutError(f"{branch} timed out internally")
    embedder = StubEmbedder(error=err) if branch == "embed" else StubEmbedder()
    index = StubIndex(
        vector=Behavior(error=err) if branch == "vector" else None,
        text=Behavior(error=err) if branch == "text" else Behavior(delay=LONG),
    )
    # Large budget: this TimeoutError is the dependency's own, not ours.
    exc = _run_and_capture(HybridRetriever(embedder, index, timeout_s=LONG))
    assert type(exc) is RetrievalError
    assert exc.__cause__ is err


# --- errors: our own time budget ---------------------------------------------------------


@pytest.mark.parametrize("slow", ["embed", "vector", "text"])
def test_budget_expiry_cancels_the_slow_call_and_becomes_retrieval_error(slow: str) -> None:
    # The slow call waits on an Event that is never set, so it can ONLY end by being
    # cancelled: the budget always expires on it. The other calls are fast; they may
    # finish before the budget expires or be cancelled with it, and the test accepts both.
    def gate(name: str) -> asyncio.Event | None:
        return asyncio.Event() if name == slow else None

    embedder = StubEmbedder(gate=gate("embed"))
    index = StubIndex(
        [_r("a")],
        [_r("b")],
        vector=Behavior(gate=gate("vector")),
        text=Behavior(gate=gate("text")),
    )
    # _run_and_capture also asserts that no task is left running.
    exc = _run_and_capture(HybridRetriever(embedder, index, timeout_s=0.2))
    assert type(exc) is RetrievalError
    assert "timed out after 0.2s" in str(exc)
    assert isinstance(exc.__cause__, TimeoutError)  # created by asyncio.timeout ...
    assert not isinstance(exc.__cause__, BaseExceptionGroup)

    calls = {"embed": embedder.behavior, "vector": index.vector, "text": index.text}
    assert calls[slow].started.is_set()  # the slow call was reached ...
    assert calls[slow].cancelled  # ... and ended by cancellation
    for call in calls.values():
        if call.started.is_set():
            assert call.finished  # nothing is still running


def test_budget_does_not_apply_to_input_checks() -> None:
    retriever = _make(timeout_s=1e-9)
    assert _search(retriever, "?!") == []
    with pytest.raises(ValueError, match="k must be >= 1"):
        _search(retriever, "saml", k=0)


# --- errors: both branches failed -> ExceptionGroup cause in V, T order ------------------


def test_both_infra_failures_are_kept_in_order_vector_then_text() -> None:
    # Text fails FIRST; vector fails while being cancelled. Completion order is T, V,
    # but the cause must still be ordered V, T.
    v_err = EmbeddingError("embed down")
    t_err = IndexStoreError("text down")
    embedder = StubEmbedder(delay=LONG, error_on_cancel=v_err)
    index = StubIndex(text=Behavior(error=t_err))
    exc = _run_and_capture(HybridRetriever(embedder, index))

    assert type(exc) is RetrievalError
    assert str(exc) == "vector branch (embedding or vector search) and text search failed"
    cause = exc.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert len(cause.exceptions) == 2
    assert cause.exceptions[0] is v_err
    assert cause.exceptions[1] is t_err
    # The TaskGroup's own group is kept in __context__ (completion order, untouched).
    assert isinstance(exc.__context__, ExceptionGroup)
    assert {id(e) for e in exc.__context__.exceptions} == {id(v_err), id(t_err)}
    _assert_formats(exc)


# --- errors: programming errors are never wrapped ----------------------------------------


@pytest.mark.parametrize(
    "error",
    [ValueError("bad k"), TypeError("bad arg"), OSError("not a timeout"), KeyError("x")],
)
def test_non_infra_error_is_raised_unchanged(error: Exception) -> None:
    index = StubIndex(vector=Behavior(error=error), text=Behavior(delay=LONG))
    exc = _run_and_capture(HybridRetriever(StubEmbedder(), index))
    assert exc is error
    assert not isinstance(exc, (RetrievalError, BaseExceptionGroup))
    assert isinstance(exc.__context__, ExceptionGroup)
    assert _contains(list(exc.__context__.exceptions), error)
    _assert_formats(exc)


@pytest.mark.parametrize("vectors", [[], [QUERY_VECTOR, QUERY_VECTOR]])
def test_embedder_returning_wrong_number_of_vectors_raises_value_error(
    vectors: list[list[float]],
) -> None:
    index = StubIndex([_r("a")], [_r("b")], text=Behavior(delay=LONG))
    exc = _run_and_capture(HybridRetriever(StubEmbedder(vectors), index))
    assert type(exc) is ValueError
    assert "vectors for 1 query" in str(exc)
    assert index.vector_calls == []


def test_bug_wins_over_infra_failure_and_infra_error_stays_reachable() -> None:
    bug = TypeError("bug in text search")
    infra = IndexStoreError("vector down")
    embedder = StubEmbedder(delay=LONG, error_on_cancel=infra)
    index = StubIndex(text=Behavior(error=bug))
    exc = _run_and_capture(HybridRetriever(embedder, index))
    assert exc is bug
    assert _contains(_reachable(exc.__context__), infra)
    _assert_formats(exc)


def test_two_bugs_raise_the_vector_one_regardless_of_completion_order() -> None:
    # Text fails first, vector fails while being cancelled: still the VECTOR bug wins.
    v_bug = ValueError("vector bug")
    t_bug = TypeError("text bug")
    embedder = StubEmbedder(delay=LONG, error_on_cancel=v_bug)
    index = StubIndex(text=Behavior(error=t_bug))
    exc = _run_and_capture(HybridRetriever(embedder, index))
    assert exc is v_bug
    assert isinstance(exc.__context__, ExceptionGroup)
    assert {id(e) for e in exc.__context__.exceptions} == {id(v_bug), id(t_bug)}
    _assert_formats(exc)


# --- errors: nested ExceptionGroups (flattened only to classify) -------------------------


def test_nested_group_of_infra_errors_is_the_cause_itself() -> None:
    a = IndexStoreError("shard 1 down")
    inner = ExceptionGroup("inner", [a])
    index = StubIndex(text=Behavior(error=inner))
    exc = _run_and_capture(HybridRetriever(StubEmbedder(delay=LONG), index))
    assert type(exc) is RetrievalError
    assert exc.__cause__ is inner  # not unwrapped, not rebuilt
    assert inner.exceptions == (a,)


def test_nested_group_with_several_infra_errors_keeps_every_object() -> None:
    a = IndexStoreError("shard 1 down")
    b = TimeoutError("shard 2 timed out")
    inner = ExceptionGroup("inner", [a, b])
    c = EmbeddingError("embed down")
    embedder = StubEmbedder(delay=LONG, error_on_cancel=c)
    index = StubIndex(text=Behavior(error=inner))
    exc = _run_and_capture(HybridRetriever(embedder, index))

    assert type(exc) is RetrievalError
    cause = exc.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert cause.exceptions[0] is c  # vector branch first
    assert cause.exceptions[1] is inner  # text branch: the original nested group
    assert inner.exceptions[0] is a and inner.exceptions[1] is b  # unchanged
    assert {id(e) for e in _group_leaves(cause)} == {id(a), id(b), id(c)}
    _assert_formats(exc)


def test_nested_group_with_a_bug_raises_that_bug_and_keeps_the_graph() -> None:
    a = IndexStoreError("shard 1 down")
    v = ValueError("bad shard request")
    inner = ExceptionGroup("inner", [a, v])
    index = StubIndex(text=Behavior(error=inner))
    exc = _run_and_capture(HybridRetriever(StubEmbedder(delay=LONG), index))

    assert exc is v
    graph = _reachable(exc.__context__)
    assert _contains(graph, inner)
    assert _contains(graph, a)
    assert inner.exceptions[0] is a and inner.exceptions[1] is v  # unchanged
    _assert_formats(exc)


def test_unmatched_group_is_reraised_unchanged() -> None:
    # Defensive path: a group that cannot be matched to the branch tasks is not swallowed.
    group = ExceptionGroup("unexpected", [IndexStoreError("x")])
    with pytest.raises(ExceptionGroup) as exc_info:
        try:
            raise group
        except ExceptionGroup as caught:
            _raise_translated(caught, (("vector", None), ("text", None)))
    assert exc_info.value is group


# --- cancellation ------------------------------------------------------------------------


def test_external_cancel_propagates_and_cancels_both_branches() -> None:
    async def scenario() -> tuple[StubEmbedder, StubIndex]:
        embedder = StubEmbedder(delay=LONG)
        index = StubIndex(text=Behavior(delay=LONG))
        task = asyncio.create_task(HybridRetriever(embedder, index).search("saml"))
        await asyncio.wait_for(embedder.behavior.started.wait(), timeout=1.0)
        await asyncio.wait_for(index.text.started.wait(), timeout=1.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()  # CancelledError, not RetrievalError
        assert _other_tasks() == set()
        return embedder, index

    embedder, index = asyncio.run(scenario())
    assert embedder.behavior.cancelled
    assert index.text.cancelled


def test_external_cancel_is_not_converted_with_a_long_budget() -> None:
    async def scenario() -> None:
        embedder = StubEmbedder(delay=LONG)
        task = asyncio.create_task(
            HybridRetriever(embedder, StubIndex(), timeout_s=LONG).search("saml")
        )
        await asyncio.wait_for(embedder.behavior.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _other_tasks() == set()

    asyncio.run(scenario())


def test_external_cancel_during_a_branch_failure_still_ends_in_cancelled_error() -> None:
    # Regression: T fails first, so TaskGroup is aborting and waits for V's (slow)
    # cleanup. An external cancel arriving in that window used to be swallowed by
    # TaskGroup (errors have priority) and turned into RetrievalError.
    async def scenario() -> tuple[asyncio.Task[list[SearchResult]], StubEmbedder, StubIndex]:
        t_err = IndexStoreError("text down")
        embedder = StubEmbedder(delay=LONG, cancel_delay=0.05)
        index = StubIndex(text=Behavior(error=t_err))
        task = asyncio.create_task(HybridRetriever(embedder, index).search("saml"))

        # V has been cancelled by the TaskGroup (because T failed) and is cleaning up.
        await asyncio.wait_for(embedder.behavior.cleanup_started.wait(), timeout=1.0)
        assert index.text.finished and not embedder.behavior.finished

        task.cancel()  # external cancel, inside the window
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        assert task.cancelled()
        # The infrastructure error that happened meanwhile is not lost.
        assert isinstance(exc_info.value.__cause__, RetrievalError)
        assert exc_info.value.__cause__.__cause__ is t_err
        assert _other_tasks() == set()
        return task, embedder, index

    _, embedder, index = asyncio.run(scenario())
    assert embedder.behavior.finished and index.text.finished  # both branches ended


def test_branch_failure_does_not_leave_a_cancel_request_on_the_caller() -> None:
    # TaskGroup's own internal cancel request must not reach the caller's task:
    # otherwise a later await (or asyncio.timeout) in the caller could misbehave.
    async def scenario() -> None:
        me = asyncio.current_task()
        assert me is not None
        before = me.cancelling()
        embedder = StubEmbedder(delay=LONG)
        index = StubIndex(text=Behavior(error=IndexStoreError("text down")))
        with pytest.raises(RetrievalError):
            await HybridRetriever(embedder, index).search("saml")
        assert me.cancelling() == before
        await asyncio.sleep(0)  # no pending cancellation is delivered here
        assert embedder.behavior.cancelled  # internal cancel of the other branch happened
        assert _other_tasks() == set()

    asyncio.run(scenario())


def _cancel_search_when_runner_finishes(
    monkeypatch: pytest.MonkeyPatch, retriever: HybridRetriever
) -> tuple[asyncio.CancelledError, asyncio.Task[Any]]:
    """Cancel `search` exactly after `runner` finished and BEFORE `search` resumes.

    The done-callback is added to `runner` when it is created, so it runs before the
    callback that wakes `search` up. No sleeps: the order is fixed by asyncio.
    """
    original = asyncio.create_task
    tasks: dict[str, asyncio.Task[Any]] = {}

    def create_task(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = original(coro, **kwargs)
        if getattr(coro, "__name__", "") == "_run_branches":
            tasks["runner"] = task
            task.add_done_callback(lambda _: tasks["search"].cancel())
        return task

    monkeypatch.setattr(asyncio, "create_task", create_task)

    async def scenario() -> asyncio.CancelledError:
        tasks["search"] = original(retriever.search("saml"))
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await tasks["search"]  # raises: nothing is returned after the cancel
        assert tasks["search"].cancelled()
        assert _other_tasks() == set()
        return exc_info.value

    cancel = asyncio.run(scenario())
    return cancel, tasks["runner"]


def test_cancel_after_runner_failed_with_a_bug_keeps_the_bug_in_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bug = ValueError("bug in text search")
    index = StubIndex([_r("a")], text=Behavior(error=bug))
    cancel, runner = _cancel_search_when_runner_finishes(
        monkeypatch, HybridRetriever(StubEmbedder(), index)
    )
    assert runner.done() and not runner.cancelled()
    assert runner.exception() is bug
    assert cancel.__cause__ is bug  # not lost
    assert isinstance(bug.__context__, ExceptionGroup)  # bug's own chain unchanged
    _assert_formats(cancel)


def test_cancel_after_runner_failed_with_retrieval_error_keeps_it_in_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = IndexStoreError("text down")
    index = StubIndex([_r("a")], text=Behavior(error=err))
    cancel, runner = _cancel_search_when_runner_finishes(
        monkeypatch, HybridRetriever(StubEmbedder(), index)
    )
    cause = cancel.__cause__
    assert isinstance(cause, RetrievalError)
    assert cause is runner.exception()
    assert cause.__cause__ is err
    _assert_formats(cancel)


def test_cancel_after_runner_succeeded_has_no_cause_and_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = StubIndex([_r("a")], [_r("b")])
    cancel, runner = _cancel_search_when_runner_finishes(
        monkeypatch, HybridRetriever(StubEmbedder(), index)
    )
    assert runner.done() and not runner.cancelled() and runner.exception() is None
    assert cancel.__cause__ is None


def test_cancel_while_runner_is_running_keeps_existing_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = asyncio.create_task
    runners: list[asyncio.Task[Any]] = []

    def create_task(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = original(coro, **kwargs)
        if getattr(coro, "__name__", "") == "_run_branches":
            runners.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", create_task)

    async def scenario() -> asyncio.CancelledError:
        embedder = StubEmbedder(delay=LONG)
        search = original(HybridRetriever(embedder, StubIndex()).search("saml"))
        await asyncio.wait_for(embedder.behavior.started.wait(), timeout=1.0)
        search.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await search
        assert search.cancelled()
        assert _other_tasks() == set()
        return exc_info.value

    cancel = asyncio.run(scenario())
    (runner,) = runners
    assert runner.cancelled()  # the cancel reached runner (cancelled, not failed)
    assert cancel.__cause__ is None


# --- state: search never changes the instance --------------------------------------------


@pytest.mark.parametrize("scenario", ["success", "rerank_fallback", "infra_failure", "timeout"])
def test_search_does_not_change_instance_state(scenario: str) -> None:
    if scenario == "success":
        retriever = HybridRetriever(StubEmbedder(), StubIndex([_r("a")], [_r("b")]))
    elif scenario == "rerank_fallback":
        reranker = StubReranker(behavior=Behavior(error=RerankError("down")))
        index = StubIndex([_r("a")], [_r("b")])
        retriever = HybridRetriever(StubEmbedder(), index, reranker=reranker)
    elif scenario == "infra_failure":
        index = StubIndex(text=Behavior(error=IndexStoreError("text down")))
        retriever = HybridRetriever(StubEmbedder(), index)
    else:
        retriever = HybridRetriever(StubEmbedder(delay=LONG), StubIndex(), timeout_s=0.05)

    before = dict(vars(retriever))  # snapshot: names and the exact objects

    if scenario in ("success", "rerank_fallback"):
        assert _ids(_search(retriever, "saml")) == ["a", "b"]
    else:
        exc = _run_and_capture(retriever)  # also checks that no task is left running
        assert type(exc) is RetrievalError
        if scenario == "timeout":
            assert "timed out" in str(exc)

    after = vars(retriever)
    assert after.keys() == before.keys()  # nothing added or removed
    for name, value in before.items():
        assert after[name] is value, f"attribute {name!r} was replaced"


# --- concurrency: 20 searches on one instance never mix their results --------------------


def _query_number(query: str) -> int:
    return int(query.removeprefix("query-"))


class QueryAwareEmbedder:
    """Encodes the query number in the vector, so vector_search can tell queries apart."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], InputType]] = []

    @property
    def model_name(self) -> str:
        return "query-aware-stub"

    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        self.calls.append((list(texts), input_type))
        await asyncio.sleep(0)
        return [[float(_query_number(text)), 1.0] for text in texts]


class BarrierIndex:
    """Results identify their query; every call waits until ALL expected calls are in flight.

    vector_search reads the query number from the vector, text_search from the query text,
    so each arm answers for its own request only. No call returns before `expected` calls
    are running at the same time: that only happens if the searches really run concurrently.
    """

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.release = asyncio.Event()
        self.in_flight = 0
        self.max_in_flight = 0
        self.vector_queries: list[int] = []
        self.text_queries: list[int] = []

    async def _barrier(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight == self.expected:
            self.release.set()
        await self.release.wait()
        self.in_flight -= 1

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        number = int(vector[0])
        self.vector_queries.append(number)
        await self._barrier()
        return [_r(f"q{number:02d}-vector")]

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        number = _query_number(query)
        self.text_queries.append(number)
        await self._barrier()
        return [_r(f"q{number:02d}-text")]


def test_twenty_concurrent_searches_each_get_their_own_results() -> None:
    queries = [f"query-{i}" for i in range(20)]

    async def scenario() -> tuple[list[list[SearchResult]], QueryAwareEmbedder, BarrierIndex]:
        embedder = QueryAwareEmbedder()
        index = BarrierIndex(expected=2 * len(queries))  # both arms of every search
        retriever = HybridRetriever(embedder, index)  # ONE shared instance
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(retriever.search(q) for q in queries)), timeout=2.0
            )
        except TimeoutError:
            pytest.fail(
                f"barrier never released: only {index.max_in_flight} of {index.expected} "
                "index calls were in flight at the same time"
            )
        assert _other_tasks() == set()
        return results, embedder, index

    results, embedder, index = asyncio.run(scenario())

    # All 40 index calls were running at the same time (that is what released the barrier).
    assert index.max_in_flight == 2 * len(queries)
    # Every query was embedded once, as a query, and reached both arms exactly once.
    assert sorted(_query_number(texts[0]) for texts, _ in embedder.calls) == list(range(20))
    assert all(input_type == "query" for _, input_type in embedder.calls)
    assert sorted(index.vector_queries) == list(range(20))
    assert sorted(index.text_queries) == list(range(20))
    # Each search got exactly its own two results, nothing from another query.
    assert len(results) == len(queries)
    for i, result in enumerate(results):
        assert sorted(_ids(result)) == [f"q{i:02d}-text", f"q{i:02d}-vector"]


# --- concurrency with a one-at-a-time embedder (bge-m3 queue simulation) -----------------


class SerialEmbedder:
    """SIMULATION of an embedder that serves one call at a time, using asyncio.Lock.

    This is NOT a test of threading.Lock or of the real BgeM3Embedder. There, the model
    runs in a worker thread: a cancelled await does not stop the thread, which keeps
    holding its lock and delays the next requests (open item O5). Here, a cancelled call
    leaves the queue or the lock immediately.
    """

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s  # simulated compute time per call, inside the lock
        self.lock = asyncio.Lock()
        self.inside = 0
        self.max_inside = 0
        self.completed = 0
        self.cancelled = 0

    @property
    def model_name(self) -> str:
        return "serial-stub"

    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        try:
            async with self.lock:
                self.inside += 1
                self.max_inside = max(self.max_inside, self.inside)
                try:
                    await asyncio.sleep(self.hold_s)  # work time, not synchronization
                finally:
                    self.inside -= 1
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.completed += 1
        return [[float(_query_number(text)), 1.0] for text in texts]


class QueryIndex:
    """Like BarrierIndex (results identify their query), but without the barrier."""

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        await asyncio.sleep(0)
        return [_r(f"q{int(vector[0]):02d}-vector")]

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        await asyncio.sleep(0)
        return [_r(f"q{_query_number(query):02d}-text")]


def _run_twenty_with_serial_embedder(
    hold_s: float, timeout_s: float
) -> tuple[list[list[SearchResult] | BaseException], SerialEmbedder]:
    queries = [f"query-{i}" for i in range(20)]

    async def scenario() -> tuple[list[list[SearchResult] | BaseException], SerialEmbedder]:
        embedder = SerialEmbedder(hold_s)
        retriever = HybridRetriever(embedder, QueryIndex(), timeout_s=timeout_s)  # ONE instance
        try:
            outcomes = await asyncio.wait_for(  # protective timeout against a deadlock
                asyncio.gather(*(retriever.search(q) for q in queries), return_exceptions=True),
                timeout=10.0,
            )
        except TimeoutError:
            pytest.fail("the 20 searches did not finish (deadlock?)")
        assert not embedder.lock.locked()  # the lock is released at the end
        assert embedder.inside == 0
        assert _other_tasks() == set()
        return outcomes, embedder

    return asyncio.run(scenario())


def _own_ids(i: int) -> list[str]:
    return [f"q{i:02d}-text", f"q{i:02d}-vector"]


def test_twenty_searches_with_serial_embedder_all_succeed_with_enough_budget() -> None:
    # 20 calls x 0.01s in the lock = ~0.2s in total, far below the 5s budget.
    outcomes, embedder = _run_twenty_with_serial_embedder(hold_s=0.01, timeout_s=5.0)

    assert embedder.max_inside == 1  # the embeddings really ran one at a time
    assert embedder.completed == 20 and embedder.cancelled == 0
    for i, outcome in enumerate(outcomes):
        assert isinstance(outcome, list), f"search {i} failed: {outcome!r}"
        assert sorted(_ids(outcome)) == _own_ids(i)


def test_twenty_searches_with_serial_embedder_and_short_budget_fail_independently() -> None:
    # 20 calls x 0.05s = ~1.0s of queue, budget 0.3s per search: the first ~6 in the queue
    # fit, the last ones cannot. Wide margins on both sides; the exact split is not assumed.
    outcomes, embedder = _run_twenty_with_serial_embedder(hold_s=0.05, timeout_s=0.3)

    successes = [(i, o) for i, o in enumerate(outcomes) if isinstance(o, list)]
    failures = [o for o in outcomes if not isinstance(o, list)]
    assert successes, "expected at least one search to succeed"
    assert failures, "expected at least one search to time out"
    assert embedder.max_inside == 1

    # Every failure is our own budget expiring: no partial list, no CancelledError leaking
    # from another request's failure, no other error type.
    for failure in failures:
        assert type(failure) is RetrievalError
        assert "timed out" in str(failure)
        assert isinstance(failure.__cause__, TimeoutError)
    # Every success got exactly its own two results.
    for i, result in successes:
        assert sorted(_ids(result)) == _own_ids(i)
    # Every embedding either completed or was cancelled (waiting for or holding the lock).
    # A search can still time out AFTER its embedding completed (budget ends during the
    # index calls), so completions may exceed successes, never the other way round.
    assert embedder.completed + embedder.cancelled == 20
    assert embedder.completed >= len(successes)
    assert 1 <= embedder.cancelled <= len(failures)


# --- output isolation: mutating a result never leaks into a later search ------------------


class SharedObjectsIndex:
    """Returns the SAME SearchResult objects on every call (like a cache): no copies.

    With this index, the only thing that can keep two searches apart is HybridRetriever
    itself (via RRF's deep copies). StubIndex copies its results, so it would hide a leak.
    """

    def __init__(self, vector_results: list[SearchResult], text_results: list[SearchResult]):
        self.vector_results = vector_results
        self.text_results = text_results

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return list(self.vector_results)  # new list, same objects

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return list(self.text_results)  # new list, same objects


def _mutate_first_then_search_again(
    retriever: HybridRetriever,
) -> tuple[SearchResult, SearchResult]:
    async def scenario() -> tuple[SearchResult, SearchResult]:
        (first,) = await retriever.search("saml", k=1)
        first.text = "changed"  # attribute replaced
        first.heading_path.append("Injected")  # list mutated in place
        (second,) = await retriever.search("saml", k=1)
        return first, second

    return asyncio.run(scenario())


def _leaked_fields(second: SearchResult) -> set[str]:
    leaked = set()
    if second.text == "changed":
        leaked.add("text")
    if "Injected" in second.heading_path:
        leaked.add("heading_path")
    return leaked


def test_mutating_a_returned_result_does_not_affect_a_later_search() -> None:
    item = _r("a")  # the same object in both lists
    original = item.model_dump()
    retriever = HybridRetriever(StubEmbedder(), SharedObjectsIndex([item], [item]))

    first, second = _mutate_first_then_search_again(retriever)

    assert first is not item and second is not item and second is not first
    assert _leaked_fields(second) == set()
    assert second.text == "Text a"
    assert second.heading_path == ["Section"]
    assert item.model_dump() == original  # the stub's own objects are untouched


@pytest.mark.parametrize(
    ("copy_mode", "expected_leaks"),
    [("none", {"text", "heading_path"}), ("shallow", {"heading_path"})],
)
def test_isolation_check_detects_a_fusion_without_deep_copies(
    monkeypatch: pytest.MonkeyPatch, copy_mode: str, expected_leaks: set[str]
) -> None:
    # Sensitivity check for the test above: replace RRF (in hybrid_retriever only, rrf.py
    # is untouched) with a version that does NOT deep-copy, and show the leak is caught.
    def leaky_fusion(
        vector_results: list[SearchResult], text_results: list[SearchResult], *, rrf_k: int
    ) -> list[SearchResult]:
        kept: dict[str, SearchResult] = {}
        for result in [*vector_results, *text_results]:
            kept.setdefault(result.chunk_id, result if copy_mode == "none" else result.model_copy())
        return list(kept.values())

    monkeypatch.setattr("kc.retrieval.hybrid_retriever.reciprocal_rank_fusion", leaky_fusion)
    item = _r("a")
    retriever = HybridRetriever(StubEmbedder(), SharedObjectsIndex([item], [item]))

    _, second = _mutate_first_then_search_again(retriever)

    assert _leaked_fields(second) == expected_leaks


def test_no_task_is_left_after_success() -> None:
    async def scenario() -> None:
        await HybridRetriever(StubEmbedder(), StubIndex([_r("a")])).search("saml")
        assert _other_tasks() == set()

    asyncio.run(scenario())


# === F9: optional reranker (after RRF + release-notes weighting, before the cut) =========

FIVE = [_r("a"), _r("b"), _r("c"), _r("d"), _r("e")]  # vector list only: RRF keeps a..e
LOGGER = "kc.retrieval.hybrid_retriever"


def _with_reranker(
    reranker: StubReranker,
    vector: list[SearchResult],
    text: list[SearchResult] | None = None,
    **kwargs: Any,
) -> HybridRetriever:
    return HybridRetriever(StubEmbedder(), StubIndex(vector, text), reranker=reranker, **kwargs)


def _no_reranker(
    vector: list[SearchResult], text: list[SearchResult] | None = None, **kwargs: Any
) -> HybridRetriever:
    return HybridRetriever(StubEmbedder(), StubIndex(vector, text), **kwargs)


def _scores(results: list[SearchResult]) -> dict[str, float]:
    return {r.chunk_id: r.score for r in results}


# --- construction ------------------------------------------------------------------------


@pytest.mark.parametrize("rerank_candidates", [1, 20, 1000])
def test_valid_rerank_candidates(rerank_candidates: int) -> None:
    assert _make(rerank_candidates=rerank_candidates)._rerank_candidates == rerank_candidates


@pytest.mark.parametrize("rerank_candidates", [True, False, 5.0, "5", None])
def test_rerank_candidates_wrong_type_raises_type_error(rerank_candidates: Any) -> None:
    with pytest.raises(TypeError, match="rerank_candidates must be an int"):
        _make(rerank_candidates=rerank_candidates)


@pytest.mark.parametrize("rerank_candidates", [0, -1])
def test_rerank_candidates_below_one_raises_value_error(rerank_candidates: int) -> None:
    with pytest.raises(ValueError, match="rerank_candidates must be >= 1"):
        _make(rerank_candidates=rerank_candidates)


@pytest.mark.parametrize("rerank_timeout_s", [0.001, 3, 3.0, 60.0])
def test_valid_rerank_timeout_s(rerank_timeout_s: float) -> None:
    assert _make(rerank_timeout_s=rerank_timeout_s)._rerank_timeout_s == rerank_timeout_s


@pytest.mark.parametrize("rerank_timeout_s", [True, False, "3", None])
def test_rerank_timeout_s_wrong_type_raises_type_error(rerank_timeout_s: Any) -> None:
    with pytest.raises(TypeError, match="rerank_timeout_s must be an int or float"):
        _make(rerank_timeout_s=rerank_timeout_s)


@pytest.mark.parametrize("rerank_timeout_s", [0, 0.0, -1.0, math.nan, math.inf, -math.inf])
def test_rerank_timeout_s_not_finite_or_not_positive_raises_value_error(
    rerank_timeout_s: float,
) -> None:
    with pytest.raises(ValueError, match="rerank_timeout_s must be finite and > 0"):
        _make(rerank_timeout_s=rerank_timeout_s)


def test_reranker_is_keyword_only() -> None:
    untyped: Any = HybridRetriever  # mypy already rejects this call; check it at runtime
    with pytest.raises(TypeError):
        untyped(StubEmbedder(), StubIndex(), StubReranker())


# --- no reranker: behaviour unchanged ----------------------------------------------------


def test_without_a_reranker_rerank_settings_change_nothing() -> None:
    vector, text = [_rn("a"), _r("b"), _r("c")], [_r("c"), _rn("d")]
    plain = _search(_no_reranker(vector, text, release_notes_weight=0.5), "q", k=10)
    tuned = _search(
        _no_reranker(
            vector, text, release_notes_weight=0.5, rerank_candidates=1, rerank_timeout_s=0.001
        ),
        "q",
        k=10,
    )
    assert tuned == plain


# --- the candidate pool --------------------------------------------------------------------


def test_reranker_gets_the_top_rerank_candidates_with_k_equal_to_the_pool() -> None:
    reranker = StubReranker()
    _search(_with_reranker(reranker, FIVE, rerank_candidates=3), "q", k=2)

    ((query, pool, k),) = reranker.calls
    assert query == "q"
    assert pool == reciprocal_rank_fusion(FIVE, [])[:3]  # same order, same RRF scores
    assert k == len(pool) == 3  # all of them, not the final k


def test_pool_is_at_least_k() -> None:
    reranker = StubReranker()
    results = _search(_with_reranker(reranker, FIVE, rerank_candidates=2), "q", k=4)
    assert _ids(reranker.calls[0][1]) == ["a", "b", "c", "d"]
    assert len(results) == 4


def test_pool_is_capped_by_the_number_of_results() -> None:
    reranker = StubReranker()
    _search(_with_reranker(reranker, FIVE, rerank_candidates=20), "q", k=2)
    ((_, pool, k),) = reranker.calls
    assert _ids(pool) == ["a", "b", "c", "d", "e"]
    assert k == 5


def test_release_notes_weight_decides_pool_membership() -> None:
    # rn 1/61, x 1/62: weight 0.9 puts x first (0.9/61 < 1/62), so only x enters a pool of 1.
    vector = [_rn("rn"), _r("x"), _r("y")]
    full, reduced = StubReranker(), StubReranker()
    _search(_with_reranker(full, vector, rerank_candidates=1), "q", k=1)
    _search(
        _with_reranker(reduced, vector, rerank_candidates=1, release_notes_weight=0.9), "q", k=1
    )
    assert _ids(full.calls[0][1]) == ["rn"]
    assert _ids(reduced.calls[0][1]) == ["x"]


@pytest.mark.parametrize("query", ["", "   ", "?!"])
def test_query_without_tokens_never_calls_the_reranker(query: str) -> None:
    reranker = StubReranker()
    assert _search(_with_reranker(reranker, FIVE), query) == []
    assert reranker.calls == []


def test_no_results_never_calls_the_reranker() -> None:
    reranker = StubReranker()
    assert _search(_with_reranker(reranker, []), "q") == []
    assert reranker.calls == []


# --- order, k and the two score spaces -------------------------------------------------------


def test_output_follows_the_reranker_scores() -> None:
    reranker = StubReranker({"c": 3.0, "e": 2.0, "a": 1.0})
    results = _search(_with_reranker(reranker, FIVE), "q", k=10)
    assert [(r.chunk_id, r.score) for r in results] == [
        ("c", 3.0),
        ("e", 2.0),
        ("a", 1.0),
        ("b", 0.0),  # missing from the stub's mapping: 0.0, kept (reorder, not filter)
        ("d", 0.0),
    ]


def test_equal_reranker_scores_are_ordered_by_chunk_id() -> None:
    reranker = StubReranker({"e": 1.0, "c": 1.0, "a": 1.0, "b": 2.0})
    assert _ids(_search(_with_reranker(reranker, FIVE), "q", k=4)) == ["b", "a", "c", "e"]


def test_release_note_score_is_the_reranker_score_times_the_weight_once() -> None:
    vector = [_rn("rn"), _r("howto")]
    reranker = StubReranker({"rn": 0.8, "howto": 0.5})
    results = _search(_with_reranker(reranker, vector, release_notes_weight=0.5), "q", k=10)
    # 0.8 * 0.5 = 0.4: not the RRF score, not 0.8 * 0.5**2 (weighted twice)
    assert _scores(results) == {"howto": 0.5, "rn": 0.8 * 0.5}
    assert _ids(results) == ["howto", "rn"]


def test_weight_one_returns_the_reranker_output_exactly() -> None:
    vector = [_rn("rn"), _r("x"), _r("y")]
    reranker = StubReranker({"y": 0.9, "rn": 0.7, "x": 0.1})
    results = _search(_with_reranker(reranker, vector), "q", k=10)
    assert [(r.chunk_id, r.score) for r in results] == [("y", 0.9), ("rn", 0.7), ("x", 0.1)]


def test_int_reranker_score_becomes_a_float() -> None:
    def int_scores(pool: list[SearchResult]) -> list[SearchResult]:
        # model_copy skips validation, so the int is stored as is (a valid score >= 0).
        return [r.model_copy(update={"score": 3 if r.chunk_id == "a" else 1}) for r in pool]

    reranker = StubReranker(output=int_scores)
    results = _search(_with_reranker(reranker, FIVE, rerank_candidates=3), "q", k=3)

    assert [(r.chunk_id, r.score) for r in results] == [("a", 3.0), ("b", 1.0), ("c", 1.0)]
    assert all(type(r.score) is float for r in results)
    by_id = {r.chunk_id: r for r in FIVE}
    for r in results:  # every other field is kept
        assert r.model_dump(exclude={"score"}) == by_id[r.chunk_id].model_dump(exclude={"score"})


def test_cut_to_k_happens_after_the_second_weighting() -> None:
    # Reranker order: rn1 3.0, rn2 2.9, howto 1.0. Weight 0.3: rn1 0.9, rn2 0.87, so the
    # how-to (third for the reranker) becomes the only result for k=1.
    vector = [_rn("rn1"), _rn("rn2"), _r("howto")]
    scores = {"rn1": 3.0, "rn2": 2.9, "howto": 1.0}
    assert _ids(_search(_with_reranker(StubReranker(scores), vector), "q", k=1)) == ["rn1"]
    weighted = _with_reranker(StubReranker(scores), vector, release_notes_weight=0.3)
    assert _ids(_search(weighted, "q", k=1)) == ["howto"]


def test_fields_other_than_score_match_the_indexed_results() -> None:
    reranker = StubReranker({"b": 1.0})
    results = _search(_with_reranker(reranker, FIVE), "q", k=10)
    by_id = {r.chunk_id: r for r in FIVE}
    for r in results:
        assert r.model_dump(exclude={"score"}) == by_id[r.chunk_id].model_dump(exclude={"score"})


# --- fallback: RerankError / TimeoutError -> the RRF ranking -------------------------------


@pytest.mark.parametrize(
    "error",
    [
        # Messages that echo the request, as a provider error might: must not be logged.
        RerankError("provider down for 'secret customer question saml' (detail-x)"),
        TimeoutError("provider timeout, request body: secret customer question (detail-x)"),
    ],
)
def test_infra_failure_falls_back_to_the_weighted_rrf_ranking(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    vector, text = [_rn("a"), _r("b"), _r("c")], [_r("c"), _rn("d")]
    reranker = StubReranker({"d": 9.0}, behavior=Behavior(error=error))
    query = "secret customer question saml"

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = _search(
            _with_reranker(reranker, vector, text, release_notes_weight=0.5), query, k=3
        )

    # Identical results AND scores (weighted once, in the RRF score space) to no reranker.
    assert results == _search(_no_reranker(vector, text, release_notes_weight=0.5), query, k=3)
    assert len(reranker.calls) == 1
    (record,) = [r for r in caplog.records if r.name == LOGGER]
    assert record.levelno == logging.WARNING
    assert "falling back" in record.getMessage()
    assert type(error).__name__ in record.getMessage()
    assert record.exc_info is None  # no traceback attached
    # Nothing from the query or the exception, anywhere in the formatted log output.
    assert "secret" not in caplog.text
    assert "detail-x" not in caplog.text
    assert "Traceback" not in caplog.text


def test_reranker_timeout_falls_back_and_cancels_the_reranker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reranker = StubReranker({"e": 9.0}, behavior=Behavior(delay=LONG))

    async def scenario() -> list[SearchResult]:
        retriever = _with_reranker(reranker, FIVE, timeout_s=LONG, rerank_timeout_s=0.05)
        results = await retriever.search("q", k=2)
        assert _other_tasks() == set()
        return results

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = asyncio.run(scenario())

    assert _ids(results) == ["a", "b"]  # the RRF ranking
    assert reranker.behavior.cancelled  # its work was cancelled by the budget
    assert any("TimeoutError" in r.getMessage() for r in caplog.records if r.name == LOGGER)


# --- bugs are raised, never a fallback ------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bug"),
        TypeError("bug"),
        ExceptionGroup("group", [RerankError("inside a group")]),
    ],
)
def test_other_reranker_errors_are_raised_unchanged(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    reranker = StubReranker(behavior=Behavior(error=error))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        exc = _run_and_capture(_with_reranker(reranker, FIVE))
    assert exc is error
    assert not [r for r in caplog.records if r.name == LOGGER]  # no fallback


def _replace_score(value: Any) -> Callable[[list[SearchResult]], list[SearchResult]]:
    # model_copy(update=...) skips validation, so a broken reranker can return anything.
    return lambda pool: [pool[0].model_copy(update={"score": value}), *pool[1:]]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda pool: pool[:-1], "returned 2 results for 3 candidates"),
        (lambda pool: [*pool, _r("zzz")], "returned 4 results for 3 candidates"),
        (lambda pool: [*pool[:-1], _r("zzz")], r"missing \['c'\], extra \['zzz'\]"),
        (lambda pool: [pool[0], pool[0], pool[1]], r"duplicate chunk_ids: \['a'\]"),
        (_replace_score(-1.0), "must be >= 0, got -1.0"),
        (_replace_score(-1), "must be >= 0, got -1"),
        (_replace_score(math.nan), "must be finite"),
        (_replace_score(math.inf), "must be finite"),
        (_replace_score(-math.inf), "must be finite"),
        (_replace_score(10**400), "must be finite, got an int too large for a float"),
        (_replace_score(None), "must be a number, got NoneType"),
        (_replace_score(True), "must be a number, got bool"),
        (_replace_score("1.0"), "must be a number, got str"),
    ],
)
def test_invalid_reranker_output_raises_value_error(
    output: Callable[[list[SearchResult]], list[SearchResult]], message: str
) -> None:
    reranker = StubReranker(output=output)
    retriever = _with_reranker(reranker, FIVE, rerank_candidates=3)
    with pytest.raises(ValueError, match=message):
        _search(retriever, "q", k=3)


def test_score_overflow_from_model_copy_raises_a_clear_value_error() -> None:
    huge = _r("a").model_copy(update={"score": 10**400})
    with pytest.raises(OverflowError):  # what math.isfinite alone would raise
        math.isfinite(huge.score)
    reranker = StubReranker(output=lambda pool: [huge, *pool[1:]])
    with pytest.raises(ValueError, match="reranker score for 'a' must be finite"):
        _search(_with_reranker(reranker, FIVE, rerank_candidates=3), "q", k=3)


def test_zero_scores_are_valid_output() -> None:
    reranker = StubReranker({})  # every score 0.0
    results = _search(_with_reranker(reranker, FIVE), "q", k=10)
    assert [(r.chunk_id, r.score) for r in results] == [(i, 0.0) for i in "abcde"]


# --- cancellation during reranking -------------------------------------------------------


def test_external_cancel_during_reranking_is_not_a_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reranker = StubReranker(behavior=Behavior(delay=LONG))

    async def scenario() -> None:
        retriever = _with_reranker(reranker, FIVE, rerank_timeout_s=LONG)
        task = asyncio.create_task(retriever.search("q"))
        await asyncio.wait_for(reranker.behavior.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert _other_tasks() == set()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(scenario())

    assert reranker.behavior.cancelled
    assert not [r for r in caplog.records if r.name == LOGGER]  # no fallback, no warning
