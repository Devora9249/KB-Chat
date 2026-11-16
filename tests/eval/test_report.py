"""Tests for the text report (eval/report.py)."""

from eval_factories import KC, make_case, make_result

from eval.report import format_report
from eval.retrieval_eval import CaseResult, RetrievalReport, score_case


def test_format_report_shows_each_status_and_summary() -> None:
    report = RetrievalReport(
        k=5,
        results=(
            score_case(make_case("hit"), [make_result(KC + "a")]),
            score_case(make_case("miss"), []),
            score_case(make_case("neg", expected_urls=[]), []),
            CaseResult(case=make_case("err"), error="RetrievalError: down"),
        ),
    )
    text = format_report(report)
    for marker in ("HIT ", "MISS ", "NEG ", "ERROR", "Hit Rate:   0.500", "1 errors"):
        assert marker in text


def test_format_report_shows_na_when_nothing_scored() -> None:
    assert "Hit Rate:   n/a" in format_report(RetrievalReport(k=5, results=()))
