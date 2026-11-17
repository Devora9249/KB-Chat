"""Tests for the generation metrics (eval/generation_metrics.py)."""

import pytest

from contracts.evaluation import EvalCase
from contracts.generation import Answer, AnswerStatus, Citation
from eval.generation_metrics import (
    cites_expected_article,
    dangling_markers,
    expected_status,
    marker_ids,
    markers_valid,
    status_correct,
)

KC = "https://knowledgecenter.gcora.genpact.com/help/"
A, B, C = (KC + s for s in "abc")


def make_case(expected_urls: list[str]) -> EvalCase:
    return EvalCase(question="q", expected_urls=expected_urls, product=None, reference_answer=None)


def make_citation(id: int, url: str) -> Citation:
    return Citation(id=id, url=url, title="t", chunk_id=f"chunk-{id}")


def make_answer(
    text: str = "x",
    status: AnswerStatus = "answered",
    citations: list[Citation] | None = None,
) -> Answer:
    # An "answered" Answer must have a citation (contract rule), so default to one.
    if citations is None:
        citations = [make_citation(1, A)] if status == "answered" else []
    return Answer(text=text, citations=citations, status=status)


def test_expected_status() -> None:
    assert expected_status(make_case([A])) == "answered"
    assert expected_status(make_case([])) == "not_found"


@pytest.mark.parametrize(
    ("status", "expected_urls", "want"),
    [
        ("answered", [A], True),
        ("not_found", [], True),  # negative case: saying "not found" is correct
        ("answered", [], False),  # negative case: made up an answer
        ("not_found", [A], False),  # positive case: gave up although an answer exists
        ("needs_clarification", [A], False),
        ("needs_clarification", [], False),
    ],
)
def test_status_correct(status: AnswerStatus, expected_urls: list[str], want: bool) -> None:
    assert status_correct(make_answer(status=status), make_case(expected_urls)) is want


def test_marker_ids_in_order_without_duplicates() -> None:
    assert marker_ids("see [2] and [1], and again [2]") == [2, 1]


def test_marker_ids_ignores_non_numeric_brackets() -> None:
    assert marker_ids("[Note] this is [1] and [a1] and [] and [10]") == [1, 10]


def test_marker_ids_empty_text() -> None:
    assert marker_ids("no sources here") == []


def test_dangling_markers_found() -> None:
    answer = make_answer("claim [1] and claim [3]", citations=[make_citation(1, A)])
    assert dangling_markers(answer) == [3]


def test_dangling_markers_none() -> None:
    answer = make_answer("claim [1]", citations=[make_citation(1, A)])
    assert dangling_markers(answer) == []


def test_markers_valid_for_proper_answer() -> None:
    answer = make_answer("claim [1] and [2]", citations=[make_citation(1, A), make_citation(2, B)])
    assert markers_valid(answer) is True


def test_markers_invalid_when_marker_dangles() -> None:
    answer = make_answer("claim [2]", citations=[make_citation(1, A)])
    assert markers_valid(answer) is False


def test_markers_invalid_when_answered_text_has_no_marker() -> None:
    # Has a citation object but the text never refers to it: the claims are not attributed.
    assert markers_valid(make_answer("a claim with no marker")) is False


@pytest.mark.parametrize("status", ["not_found", "needs_clarification"])
def test_markers_valid_trivially_for_answers_without_claims(status: AnswerStatus) -> None:
    assert markers_valid(make_answer("I didn't find it", status=status)) is True


def test_cites_expected_article_true_when_one_citation_matches() -> None:
    answer = make_answer("[1] [2]", citations=[make_citation(1, C), make_citation(2, B)])
    assert cites_expected_article(answer, make_case([A, B])) is True


def test_cites_expected_article_false_when_only_wrong_articles_cited() -> None:
    answer = make_answer("[1]", citations=[make_citation(1, C)])
    assert cites_expected_article(answer, make_case([A, B])) is False


def test_cites_expected_article_false_when_no_citations() -> None:
    answer = make_answer("nothing", status="not_found")
    assert cites_expected_article(answer, make_case([A])) is False


def test_cites_expected_article_raises_for_negative_case() -> None:
    with pytest.raises(ValueError, match="negative case"):
        cites_expected_article(make_answer(), make_case([]))
