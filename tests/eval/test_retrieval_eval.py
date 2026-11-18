"""Tests for eval/retrieval_eval.py.

Scoring is pure (`score_case`), so it is tested without any retriever. Only the
interaction with the Retriever (arguments, error handling) uses a stdlib AsyncMock.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from eval_factories import KC, make_case, make_result

from contracts.retrieval import RetrievalError
from eval.retrieval_eval import RetrievalReport, evaluate_retrieval, score_case


def test_score_case_ranks_articles_not_chunks() -> None:
    results = [make_result(KC + "a"), make_result(KC + "a"), make_result(KC + "b")]
    scored = score_case(make_case(expected_urls=[KC + "b"]), results)
    assert scored.retrieved_urls == (KC + "a", KC + "b")
    assert scored.scores is not None and scored.scores.reciprocal_rank == pytest.approx(0.5)


def test_score_case_does_not_score_negative_case() -> None:
    scored = score_case(make_case(expected_urls=[]), [make_result(KC + "a")])
    assert scored.scores is None
    assert scored.retrieved_urls == (KC + "a",)


def test_report_averages_scored_cases_only() -> None:
    report = RetrievalReport(
        k=5,
        results=(
            score_case(make_case(), [make_result(KC + "a")]),  # hit at rank 1
            score_case(make_case(), [make_result(KC + "x"), make_result(KC + "a")]),  # rank 2
            score_case(make_case(), []),  # miss
            score_case(make_case(expected_urls=[]), []),  # negative: not averaged
        ),
    )
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx(0.5)
    assert report.num_negative == 1


def test_report_without_scored_cases_has_no_metrics() -> None:
    report = RetrievalReport(k=5, results=())
    assert report.hit_rate is report.mean_recall is report.mrr is None


def test_evaluate_passes_k_and_no_filters() -> None:
    retriever = AsyncMock()
    retriever.search.return_value = [make_result(KC + "a")]
    report = asyncio.run(evaluate_retrieval(retriever, [make_case("q")], k=3))
    retriever.search.assert_awaited_once_with("q", k=3)
    assert report.hit_rate == 1.0


def test_evaluate_records_retrieval_error_and_continues() -> None:
    retriever = AsyncMock()
    retriever.search.side_effect = [RetrievalError("index down"), [make_result(KC + "a")]]
    report = asyncio.run(evaluate_retrieval(retriever, [make_case(), make_case()], k=5))
    assert report.results[0].error == "RetrievalError: index down"
    assert report.num_errors == 1
    assert len(report.scores) == 1


def test_evaluate_propagates_other_exceptions() -> None:
    retriever = AsyncMock()
    retriever.search.side_effect = KeyError("bug")
    with pytest.raises(KeyError):
        asyncio.run(evaluate_retrieval(retriever, [make_case()], k=5))


def test_evaluate_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        asyncio.run(evaluate_retrieval(AsyncMock(), [], k=0))
