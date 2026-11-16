"""Tests for the retrieval metrics (eval/metrics.py)."""

from collections.abc import Callable, Sequence

import pytest
from eval_factories import KC, make_result

from eval.metrics import hit, ranked_unique_urls, recall, reciprocal_rank

A, B, C, D = (KC + s for s in "abcd")


def test_ranked_unique_urls_keeps_first_occurrence_order() -> None:
    results = [make_result(B), make_result(A), make_result(B), make_result(C)]
    assert ranked_unique_urls(results) == [B, A, C]


@pytest.mark.parametrize(
    ("ranked", "expected", "want"),
    [
        ([A, B, C], [B], True),
        ([A, B, C], [D], False),
        ([], [A], False),
        ([A], [D, A], True),  # any one expected article is enough
    ],
)
def test_hit(ranked: list[str], expected: list[str], want: bool) -> None:
    assert hit(ranked, expected) is want


@pytest.mark.parametrize(
    ("ranked", "expected", "want"),
    [
        ([A, B, C], [A], 1.0),
        ([A, B, C], [A, D], 0.5),
        ([A, B, C], [D], 0.0),
    ],
)
def test_recall(ranked: list[str], expected: list[str], want: float) -> None:
    assert recall(ranked, expected) == pytest.approx(want)


@pytest.mark.parametrize(
    ("ranked", "expected", "want"),
    [
        ([A, B, C], [A], 1.0),
        ([A, B, C], [C, B], 0.5),  # the first expected article found counts
        ([A, B, C], [D], 0.0),
    ],
)
def test_reciprocal_rank(ranked: list[str], expected: list[str], want: float) -> None:
    assert reciprocal_rank(ranked, expected) == pytest.approx(want)


@pytest.mark.parametrize("metric", [hit, recall, reciprocal_rank])
def test_metrics_reject_negative_case(
    metric: Callable[[Sequence[str], Sequence[str]], object],
) -> None:
    with pytest.raises(ValueError, match="negative case"):
        metric([A], [])
