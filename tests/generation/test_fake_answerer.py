"""Tests for FakeAnswerer: one test per status rule."""

import asyncio

from contracts.generation import Answer, AskRequest
from kc.generation.fake_answerer import FakeAnswerer


def ask(question: str) -> Answer:
    """Run the async answerer from a plain (sync) test."""
    return asyncio.run(FakeAnswerer().answer(AskRequest(question=question)))


def test_normal_question_is_answered_with_citations() -> None:
    answer = ask("How do I configure SAML in SeQuence 9?")
    assert answer.status == "answered"
    assert [c.id for c in answer.citations] == [1, 2]
    assert answer.text.startswith("[FAKE]")


def test_question_with_unknown_is_not_found() -> None:
    answer = ask("How do I configure the Unknown module?")
    assert answer.status == "not_found"
    assert answer.citations == []


def test_short_question_needs_clarification() -> None:
    answer = ask("saml")
    assert answer.status == "needs_clarification"
    assert answer.citations == []


def test_unknown_wins_over_short() -> None:
    assert ask("unknown").status == "not_found"
