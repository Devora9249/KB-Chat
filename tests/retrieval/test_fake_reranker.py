"""Tests for FakeReranker and the Reranker protocol (Search & Ranking)."""

import asyncio
import itertools
import math
from collections.abc import Sequence
from typing import Any

import pytest

from contracts.retrieval import SearchResult
from kc.retrieval.fake_reranker import FakeReranker
from kc.retrieval.reranker import Reranker, RerankError


def _item(chunk_id: str, text: str, **overrides: Any) -> SearchResult:
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "url": f"https://knowledgecenter.gcora.genpact.com/help/{chunk_id}",
        "title": f"Title {chunk_id}",
        "text": text,
        "score": 0.5,
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
        "heading_path": ["Section", chunk_id],
    }
    base.update(overrides)
    return SearchResult(**base)


def _rerank(
    reranker: FakeReranker, query: str, results: Sequence[SearchResult], k: int = 10
) -> list[SearchResult]:
    return asyncio.run(reranker.rerank(query, results, k=k))


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.chunk_id for r in results]


# --- Protocol ------------------------------------------------------------------------------


def test_satisfies_reranker_protocol() -> None:
    reranker: Reranker = FakeReranker()  # mypy checks the structural match
    assert isinstance(reranker, Reranker)  # runtime_checkable: the method exists
    results = asyncio.run(reranker.rerank("saml", [_item("a", "saml")], k=1))
    assert _ids(results) == ["a"]


# --- token-overlap scoring -------------------------------------------------------------------


def test_orders_by_token_overlap_with_the_query() -> None:
    results = _rerank(
        FakeReranker(),
        "configure SAML authentication",
        [
            _item("one", "SAML only"),
            _item("none", "unrelated words"),
            _item("three", "Configure SAML authentication here"),
            _item("two", "configure saml"),
        ],
    )
    assert [(r.chunk_id, r.score) for r in results] == [
        ("three", 3.0),
        ("two", 2.0),
        ("one", 1.0),
        ("none", 0.0),  # kept: a reranker reorders, it does not filter
    ]


def test_scoring_matches_the_fake_retriever_convention() -> None:
    # Lowercase \w+ tokens; a repeated query word counts once; every occurrence in the
    # text counts; title and heading_path are ignored.
    results = _rerank(
        FakeReranker(),
        "SAML saml",
        [_item("a", "saml SAML config", title="saml", heading_path=["saml"])],
    )
    assert results[0].score == 2.0


def test_hebrew_tokens_are_scored_deterministically() -> None:
    items = [
        _item("one", "איך להגדיר"),
        _item("none", "טקסט אחר לגמרי"),
        _item("two", "מגדירים SAML איך"),
    ]
    first = _rerank(FakeReranker(), "איך מגדירים SAML?", items)
    second = _rerank(FakeReranker(), "איך מגדירים SAML?", items)

    assert [(r.chunk_id, r.score) for r in first] == [("two", 3.0), ("one", 1.0), ("none", 0.0)]
    assert first == second


def test_ties_are_ordered_by_chunk_id() -> None:
    results = _rerank(
        FakeReranker(), "saml", [_item("c", "saml"), _item("a", "saml"), _item("b", "saml")]
    )
    assert _ids(results) == ["a", "b", "c"]


@pytest.mark.parametrize("query", ["", "   ", "?! ..."])
def test_query_without_tokens_scores_zero_and_orders_by_chunk_id(query: str) -> None:
    # FakeReranker decision: nothing is discarded; every result gets 0.0.
    items = [_item("c", "saml"), _item("a", "saml saml"), _item("b", "x")]
    results = _rerank(FakeReranker(), query, items)
    assert [(r.chunk_id, r.score) for r in results] == [("a", 0.0), ("b", 0.0), ("c", 0.0)]


def test_incoming_scores_are_ignored() -> None:
    results = _rerank(
        FakeReranker(),
        "saml",
        [_item("high", "nothing", score=0.99), _item("low", "saml", score=0.01)],
    )
    assert _ids(results) == ["low", "high"]


# --- explicit score mapping ------------------------------------------------------------------


def test_explicit_scores_override_token_overlap() -> None:
    items = [_item("match", "saml saml saml"), _item("other", "unrelated")]
    results = _rerank(FakeReranker({"other": 5.0, "match": 1.0}), "saml", items)
    assert [(r.chunk_id, r.score) for r in results] == [("other", 5.0), ("match", 1.0)]


def test_chunk_missing_from_the_mapping_gets_zero() -> None:
    # "a" would score 1.0 by token overlap; in mapping mode a missing id gets 0.0.
    results = _rerank(FakeReranker({"b": 1.0}), "saml", [_item("a", "saml"), _item("b", "x")])
    assert [(r.chunk_id, r.score) for r in results] == [("b", 1.0), ("a", 0.0)]


def test_int_scores_are_accepted_as_float() -> None:
    (result,) = _rerank(FakeReranker({"a": 3}), "q", [_item("a", "x")])
    assert result.score == 3.0
    assert isinstance(result.score, float)


def test_mapping_is_copied_and_never_modified() -> None:
    scores = {"a": 1.0, "b": 2.0}
    reranker = FakeReranker(scores)
    scores["a"] = 100.0  # a later change by the caller has no effect
    scores["c"] = 50.0

    results = _rerank(reranker, "q", [_item("a", "x"), _item("b", "x"), _item("c", "x")])

    assert [(r.chunk_id, r.score) for r in results] == [("b", 2.0), ("a", 1.0), ("c", 0.0)]
    assert scores == {"a": 100.0, "b": 2.0, "c": 50.0}


@pytest.mark.parametrize("score", [True, False, "1.0", None])
def test_mapping_with_a_non_number_raises_type_error(score: Any) -> None:
    with pytest.raises(TypeError, match="score for 'a'"):
        FakeReranker({"a": score})


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_mapping_with_a_non_finite_score_raises_value_error(score: float) -> None:
    with pytest.raises(ValueError, match="score for 'a' must be finite"):
        FakeReranker({"a": score})


@pytest.mark.parametrize("score", [10**400, -(10**400)])
def test_mapping_with_an_int_too_large_for_a_float_raises_value_error(score: int) -> None:
    with pytest.raises(ValueError, match="score for 'a' must be finite"):
        FakeReranker({"a": score})


@pytest.mark.parametrize("score", [-1.0, -0.5, -1e-300, -1, -100])
def test_mapping_with_a_negative_score_raises_value_error(score: float) -> None:
    with pytest.raises(ValueError, match="score for 'a' must be >= 0"):
        FakeReranker({"a": score})


def test_mapping_accepts_zero_and_negative_zero_as_positive_zero() -> None:
    results = _rerank(FakeReranker({"a": -0.0, "b": 0}), "q", [_item("a", "x"), _item("b", "x")])
    assert [(r.chunk_id, r.score) for r in results] == [("a", 0.0), ("b", 0.0)]
    assert all(math.copysign(1.0, r.score) == 1.0 for r in results)  # no -0.0 in the output


def test_empty_mapping_scores_zero_instead_of_token_overlap() -> None:
    items = [_item("b", "saml saml"), _item("a", "unrelated")]

    overlap = _rerank(FakeReranker(), "saml", items)
    empty_mapping = _rerank(FakeReranker({}), "saml", items)

    assert [(r.chunk_id, r.score) for r in overlap] == [("b", 2.0), ("a", 0.0)]
    assert [(r.chunk_id, r.score) for r in empty_mapping] == [("a", 0.0), ("b", 0.0)]


def test_duplicate_chunk_id_in_mapping_mode_keeps_the_first_occurrence() -> None:
    first = _item("a", "first text", title="First")
    later = _item("a", "later text", title="Later", product="orchestration")

    results = _rerank(FakeReranker({"a": 5.0}), "q", [first, later], k=1)

    assert len(results) == 1
    (result,) = results
    assert result.score == 5.0
    assert result.model_dump(exclude={"score"}) == first.model_dump(exclude={"score"})

    # k larger than the input: only de-duplication can keep this to one result.
    wide = _rerank(FakeReranker({"a": 5.0}), "q", [first, later], k=10)
    assert len(wide) == 1
    assert wide[0].model_dump(exclude={"score"}) == first.model_dump(exclude={"score"})


def test_equal_mapping_scores_are_ordered_by_chunk_id() -> None:
    items = [_item("c", "x"), _item("a", "x"), _item("b", "x")]
    results = _rerank(FakeReranker({"a": 2.0, "b": 2.0, "c": 2.0}), "q", items)
    assert [(r.chunk_id, r.score) for r in results] == [("a", 2.0), ("b", 2.0), ("c", 2.0)]


# --- k, sizes, uniqueness --------------------------------------------------------------------


def test_results_are_cut_to_k() -> None:
    items = [_item(f"c{i}", "saml " * i) for i in range(5)]
    assert _ids(_rerank(FakeReranker(), "saml", items, k=2)) == ["c4", "c3"]


@pytest.mark.parametrize(("k", "expected_len"), [(1, 1), (2, 2), (3, 3), (4, 3), (10, 3)])
def test_output_length_is_exactly_min_of_k_and_unique_ids(k: int, expected_len: int) -> None:
    # 4 inputs, 3 unique chunk_ids; one gets score 0 and is still returned (no filtering).
    items = [_item("a", "saml"), _item("b", "x"), _item("a", "saml saml"), _item("c", "saml")]
    results = _rerank(FakeReranker(), "saml", items, k=k)
    assert len(results) == expected_len
    if k >= 3:
        assert sorted(_ids(results)) == ["a", "b", "c"]  # a full permutation of unique ids


def test_rerank_error_is_a_runtime_error() -> None:
    assert issubclass(RerankError, RuntimeError)
    cause = OSError("provider down")
    try:
        raise RerankError("rerank failed") from cause
    except RuntimeError as error:  # callers can catch it as RuntimeError
        assert isinstance(error, RerankError)
        assert error.__cause__ is cause


def test_fewer_results_than_k_returns_all() -> None:
    assert len(_rerank(FakeReranker(), "saml", [_item("a", "saml"), _item("b", "x")], k=10)) == 2


@pytest.mark.parametrize("results", [[], ()])
def test_empty_input_returns_empty_list(results: Sequence[SearchResult]) -> None:
    assert _rerank(FakeReranker(), "saml", results) == []


def test_output_is_a_subset_of_the_input_with_unique_ids() -> None:
    # Duplicate chunk_id in the input: FakeReranker decision (not a protocol requirement),
    # only the first occurrence is used.
    items = [_item("a", "saml"), _item("b", "saml saml"), _item("a", "saml saml saml")]
    results = _rerank(FakeReranker(), "saml", items)
    assert sorted(_ids(results)) == ["a", "b"]
    assert next(r for r in results if r.chunk_id == "a").text == "saml"  # first occurrence


@pytest.mark.parametrize("k", [True, False, 5.0, "5", None])
def test_k_wrong_type_raises_type_error(k: Any) -> None:
    with pytest.raises(TypeError, match="k must be an int"):
        _rerank(FakeReranker(), "saml", [_item("a", "saml")], k=k)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises_value_error(k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        _rerank(FakeReranker(), "saml", [_item("a", "saml")], k=k)


@pytest.mark.parametrize(("k", "error"), [(True, TypeError), (5.0, TypeError), (0, ValueError)])
def test_k_is_validated_before_the_input(k: Any, error: type[Exception]) -> None:
    # Even an empty input (which would return []) must not hide an invalid k.
    with pytest.raises(error):
        _rerank(FakeReranker(), "saml", [], k=k)


# --- metadata, isolation, determinism ---------------------------------------------------------


def test_every_field_except_score_is_preserved() -> None:
    item = _item(
        "rn",
        "saml release",
        title="Release Notes",
        product="orchestration",
        version=None,
        doc_type="release-notes",
        heading_path=["Release Notes", "11"],
    )
    (result,) = _rerank(FakeReranker(), "saml", [item])
    assert result.model_dump(exclude={"score"}) == item.model_dump(exclude={"score"})
    assert result.score == 1.0


def test_input_sequence_and_objects_are_not_modified() -> None:
    items = [_item("b", "saml"), _item("a", "saml saml")]
    before = [item.model_copy(deep=True) for item in items]

    _rerank(FakeReranker(), "saml", items)

    assert items == before  # same order, same scores, same fields


def test_output_objects_are_independent_copies() -> None:
    item = _item("a", "saml")
    (result,) = _rerank(FakeReranker(), "saml", [item])
    assert result is not item
    assert result.heading_path is not item.heading_path

    result.heading_path.append("Injected")
    result.text = "changed"
    assert item.heading_path == ["Section", "a"]
    assert item.text == "saml"


def test_mutating_the_input_after_rerank_does_not_change_the_output() -> None:
    item = _item("a", "saml")
    (result,) = _rerank(FakeReranker(), "saml", [item])
    expected = result.model_copy(deep=True)

    item.heading_path.append("Injected")
    item.text = "changed"
    item.title = "Changed"

    assert result == expected
    assert result.heading_path == ["Section", "a"]


def test_input_order_does_not_change_the_output() -> None:
    # Unique chunk_ids (with duplicates, the first occurrence depends on order by design).
    items = [
        _item("c", "saml saml"),
        _item("a", "saml"),
        _item("b", "saml"),
        _item("d", "x"),
    ]
    expected = _rerank(FakeReranker(), "saml", items, k=3)
    assert _ids(expected) == ["c", "a", "b"]
    for permutation in itertools.permutations(items):
        assert _rerank(FakeReranker(), "saml", list(permutation), k=3) == expected


def test_is_deterministic() -> None:
    items = [_item(f"c{i}", f"saml {'x ' * i}") for i in range(6)]
    first = _rerank(FakeReranker(), "saml x", items, k=4)
    second = _rerank(FakeReranker(), "saml x", items, k=4)
    assert first == second
