"""Tests for ClaudeLlmClient, with a fake SDK client: no network, no API key, no cost."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage

from kc.generation.claude_llm import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeLlmClient,
)
from kc.generation.llm import GenerationError, LlmAnswer

GOOD_REPLY = {"status": "answered", "text": "Open the page [1].", "cited_passage_ids": [1]}
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def make_message(
    *,
    text: str | None = json.dumps(GOOD_REPLY),
    stop_reason: str = "end_turn",
    refusal_category: str | None = None,
) -> BetaMessage:
    """A real SDK response object, like the API would return."""
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    data: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }
    if stop_reason == "refusal":
        data["stop_details"] = {
            "type": "refusal",
            "category": refusal_category,
            "explanation": None,
        }
    return BetaMessage.model_validate(data)


class FakeMessages:
    """Stands in for `client.beta.messages`: records calls, returns or raises."""

    def __init__(self, response: BetaMessage | None = None, error: Exception | None = None):
        self._response = response or make_message()
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def make_client(messages: FakeMessages, **kwargs: Any) -> ClaudeLlmClient:
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeLlmClient(cast(AsyncAnthropic, sdk), **kwargs)


def generate(client: ClaudeLlmClient) -> LlmAnswer:
    return asyncio.run(client.generate(system="RULES", user_message="<question>q</question>"))


def status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    return cls("error", response=httpx2.Response(code, request=REQUEST), body=None)


# --- happy path ---------------------------------------------------------------------------


def test_returns_the_parsed_answer() -> None:
    answer = generate(make_client(FakeMessages()))
    assert answer == LlmAnswer.model_validate(GOOD_REPLY)


def test_sends_the_expected_request() -> None:
    messages = FakeMessages()
    generate(make_client(messages))

    call = messages.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["system"] == "RULES"
    assert call["messages"] == [{"role": "user", "content": "<question>q</question>"}]
    assert call["output_config"]["effort"] == DEFAULT_EFFORT == "medium"
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert "temperature" not in call  # rejected by the API version SDK 2.x uses


def test_requests_structured_output_matching_llm_answer() -> None:
    messages = FakeMessages()
    generate(make_client(messages))

    output_format = messages.calls[0]["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    schema = output_format["schema"]
    assert set(schema["required"]) == {"status", "text", "cited_passage_ids"}
    assert schema["additionalProperties"] is False
    # The model sees a description written for it, not the developer docstring.
    assert "cited_passage_ids" in schema["description"]
    assert "never trust" not in schema["description"]


def test_model_effort_and_max_tokens_are_configurable() -> None:
    messages = FakeMessages()
    generate(make_client(messages, model="claude-sonnet-5-5", effort="high", max_tokens=2000))
    call = messages.calls[0]
    assert (call["model"], call["output_config"]["effort"], call["max_tokens"]) == (
        "claude-sonnet-5-5",
        "high",
        2000,
    )


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_client(FakeMessages(), max_tokens=0)


# --- replies that cannot be used -> GenerationError ---------------------------------------


def test_refusal_raises_generation_error_with_category() -> None:
    messages = FakeMessages(
        make_message(text=None, stop_reason="refusal", refusal_category="cyber")
    )
    with pytest.raises(GenerationError, match="cyber"):
        generate(make_client(messages))


def test_cut_off_reply_raises_generation_error() -> None:
    messages = FakeMessages(make_message(text='{"status": "answ', stop_reason="max_tokens"))
    with pytest.raises(GenerationError, match="max_tokens"):
        generate(make_client(messages))


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"status": "maybe", "text": "x", "cited_passage_ids": []}),
        None,  # no text block
    ],
)
def test_malformed_reply_raises_generation_error(text: str | None) -> None:
    with pytest.raises(GenerationError, match="schema"):
        generate(make_client(FakeMessages(make_message(text=text))))


# --- SDK errors ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.OverloadedError, 529),
        status_error(anthropic.RequestTimeoutError, 408),
        anthropic.APIConnectionError(request=REQUEST),
        anthropic.APITimeoutError(request=REQUEST),
    ],
    ids=lambda e: type(e).__name__,
)
def test_temporary_errors_become_generation_error(error: Exception) -> None:
    with pytest.raises(GenerationError) as info:
        generate(make_client(FakeMessages(error=error)))
    assert info.value.__cause__ is error  # the original stays attached for the log


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.AuthenticationError, 401),
        status_error(anthropic.BadRequestError, 400),
        status_error(anthropic.NotFoundError, 404),
    ],
    ids=lambda e: type(e).__name__,
)
def test_configuration_errors_are_not_wrapped(error: Exception) -> None:
    # A bad key or a wrong model is a bug to fix (500), not "try again later" (503).
    with pytest.raises(type(error)):
        generate(make_client(FakeMessages(error=error)))
