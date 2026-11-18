"""Tests for LlmAnswer and FakeLlmClient."""

import asyncio

import pytest
from pydantic import ValidationError

from kc.generation.fake_llm import DEFAULT_FAKE_ANSWER, FakeLlmClient
from kc.generation.llm import GenerationError, LlmAnswer


def generate(client: FakeLlmClient) -> LlmAnswer:
    return asyncio.run(client.generate(system="rules", user_message="question"))


def test_default_answer_is_answered_with_a_citation() -> None:
    answer = generate(FakeLlmClient())
    assert answer.status == "answered"
    assert answer.cited_passage_ids == [1]


def test_returns_the_configured_answer() -> None:
    configured = LlmAnswer(status="not_found", text="Not found.", cited_passage_ids=[])
    assert generate(FakeLlmClient(answer=configured)) == configured


def test_records_every_call() -> None:
    client = FakeLlmClient()
    generate(client)
    generate(client)
    assert len(client.calls) == 2
    assert client.calls[0].system == "rules"
    assert client.calls[0].user_message == "question"


def test_raises_the_configured_error_and_still_records_the_call() -> None:
    client = FakeLlmClient(error=GenerationError("overloaded"))
    with pytest.raises(GenerationError, match="overloaded"):
        generate(client)
    assert len(client.calls) == 1


def test_returned_answer_is_a_copy() -> None:
    client = FakeLlmClient()
    generate(client).cited_passage_ids.append(99)
    assert DEFAULT_FAKE_ANSWER.cited_passage_ids == [1]
    assert generate(client).cited_passage_ids == [1]


def test_llm_answer_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        LlmAnswer.model_validate({"status": "maybe", "text": "x", "cited_passage_ids": []})
