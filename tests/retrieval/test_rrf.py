"""Tests for Reciprocal Rank Fusion."""

from fractions import Fraction
from typing import Any

import pytest

from contracts.retrieval import SearchResult
from kc.retrieval.rrf import reciprocal_rank_fusion


def _r(chunk_id: str, score: float = 0.5, **overrides: Any) -> SearchResult:
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "url": f"https://knowledgecenter.gcora.genpact.com/help/{chunk_id}",
        "title": f"Title {chunk_id}",
        "text": f"Text {chunk_id}",
        "score": score,
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
        "heading_path": ["Section"],
    }
    base.update(overrides)
    return SearchResult(**base)


def _rrf(*ranks: int, rrf_k: int = 60) -> float:
    """Expected RRF score for a chunk at the given ranks (exact sum, then float)."""
    return float(sum((Fraction(1, rrf_k + rank) for rank in ranks), Fraction(0)))


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.chunk_id for r in results]


# 1
def test_both_empty_returns_empty_list() -> None:
    assert reciprocal_rank_fusion([], []) == []


# 2
def test_text_empty_keeps_vector_order_and_rrf_scores() -> None:
    fused = reciprocal_rank_fusion([_r("a"), _r("b"), _r("c")], [])
    assert _ids(fused) == ["a", "b", "c"]
    assert [r.score for r in fused] == [_rrf(1), _rrf(2), _rrf(3)]


# 3
def test_vector_empty_keeps_text_order_and_rrf_scores() -> None:
    fused = reciprocal_rank_fusion([], [_r("x"), _r("y"), _r("z")])
    assert _ids(fused) == ["x", "y", "z"]
    assert [r.score for r in fused] == [_rrf(1), _rrf(2), _rrf(3)]


# 4
def test_rank_starts_at_one() -> None:
    (only,) = reciprocal_rank_fusion([_r("a")], [])
    assert only.score == 1 / 61


# 5
def test_item_in_both_lists_gets_summed_score() -> None:
    fused = reciprocal_rank_fusion([_r("a"), _r("b")], [_r("c"), _r("d"), _r("a")])
    by_id = {r.chunk_id: r.score for r in fused}
    assert by_id["a"] == _rrf(1, 3)


# 6
def test_item_in_both_lists_beats_top_of_single_list() -> None:
    fused = reciprocal_rank_fusion([_r("b"), _r("a")], [_r("x"), _r("a")])
    assert _ids(fused)[0] == "a"
    assert fused[0].score == _rrf(2, 2)
    assert _rrf(2, 2) > _rrf(1)


# 7
def test_output_has_unique_chunk_ids() -> None:
    fused = reciprocal_rank_fusion(
        [_r("a"), _r("b"), _r("c")],
        [_r("b"), _r("c"), _r("d")],
    )
    assert sorted(_ids(fused)) == ["a", "b", "c", "d"]
    assert len(fused) == 4


# 8
def test_equal_scores_sorted_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([_r("zeta")], [_r("alpha")])
    assert _ids(fused) == ["alpha", "zeta"]
    assert fused[0].score == fused[1].score


# 9
def test_input_scores_are_ignored() -> None:
    fused = reciprocal_rank_fusion([_r("first", score=0.1), _r("second", score=999.0)], [])
    assert _ids(fused) == ["first", "second"]
    assert fused[1].score == _rrf(2)


# 10
def test_conflicting_metadata_keeps_first_vector_result() -> None:
    from_vector = _r("a", title="Vector title", text="Vector text", heading_path=["V"])
    from_text = _r("a", title="Text title", text="Text text", heading_path=["T"])
    (fused,) = reciprocal_rank_fusion([from_vector], [from_text])
    assert fused.title == "Vector title"
    assert fused.text == "Vector text"
    assert fused.heading_path == ["V"]
    assert fused.score == _rrf(1, 1)


# 11
def test_conflicting_metadata_only_in_text_keeps_first_text_result() -> None:
    first = _r("a", title="First", heading_path=["One"])
    second = _r("a", title="Second", heading_path=["Two"])
    (fused,) = reciprocal_rank_fusion([], [first, second])
    assert fused.title == "First"
    assert fused.heading_path == ["One"]
    assert fused.score == _rrf(1)


# 12
def test_duplicate_within_one_list_counts_first_occurrence_and_keeps_positions() -> None:
    fused = reciprocal_rank_fusion([_r("a"), _r("a"), _r("b")], [])
    assert _ids(fused) == ["a", "b"]
    by_id = {r.chunk_id: r.score for r in fused}
    assert by_id["a"] == _rrf(1)  # not 1/61 + 1/62
    assert by_id["b"] == _rrf(3)  # original position kept, not promoted to 2


# 13
def test_inputs_are_not_mutated() -> None:
    vector = [_r("a", score=0.9, heading_path=["V"]), _r("b", score=0.8)]
    text = [_r("b", score=3.0), _r("c", score=2.0)]
    vector_before = [r.model_dump() for r in vector]
    text_before = [r.model_dump() for r in text]

    fused = reciprocal_rank_fusion(vector, text)
    for r in fused:
        r.text = "changed"
        r.heading_path.append("Injected")

    assert len(vector) == 2
    assert len(text) == 2
    assert [r.model_dump() for r in vector] == vector_before
    assert [r.model_dump() for r in text] == text_before


# 14
def test_returns_new_objects() -> None:
    vector = [_r("a"), _r("b")]
    text = [_r("b"), _r("c")]
    fused = reciprocal_rank_fusion(vector, text)
    for out in fused:
        for src in [*vector, *text]:
            assert out is not src
            assert out.heading_path is not src.heading_path


# 15
def test_custom_rrf_k() -> None:
    (only,) = reciprocal_rank_fusion([_r("a")], [], rrf_k=1)
    assert only.score == 1 / 2


# 16
@pytest.mark.parametrize(
    ("rrf_k", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (60.5, TypeError),
        (60.0, TypeError),
        ("60", TypeError),
        (True, TypeError),
    ],
)
def test_invalid_rrf_k(rrf_k: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        reciprocal_rank_fusion([_r("a")], [], rrf_k=rrf_k)


# 17
def test_hand_computed_example_and_determinism() -> None:
    def build() -> tuple[list[SearchResult], list[SearchResult]]:
        return [_r("a"), _r("b"), _r("c")], [_r("c"), _r("d"), _r("a")]

    # a: vector 1 + text 3; c: vector 3 + text 1  -> exact tie, broken by chunk_id
    # b: vector 2;          d: text 2             -> exact tie, broken by chunk_id
    fused = reciprocal_rank_fusion(*build())
    assert _ids(fused) == ["a", "c", "b", "d"]
    assert [r.score for r in fused] == [_rrf(1, 3), _rrf(3, 1), _rrf(2), _rrf(2)]

    vector, text = build()
    assert reciprocal_rank_fusion(vector, text) == reciprocal_rank_fusion(vector, text)
    assert reciprocal_rank_fusion(*build()) == fused
