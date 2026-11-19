"""The real LlmClient: calls Claude through the Anthropic SDK (anthropic 2.x).

- Structured output: Claude must reply with JSON matching `LlmAnswer`
  (`output_config.format`); we validate it ourselves after checking `stop_reason`.
- Effort is set explicitly (claude-opus-5-5 always thinks; effort controls how much).
- No `temperature`: the API version used by SDK 2.x rejects it.
- Refusals: server-side fallbacks (`fallbacks="default"`) let the API retry a safety
  decline on another model inside the same call. If the whole chain still refuses,
  we raise GenerationError.

Error mapping (the API turns GenerationError into 503):
- temporary: rate limit, 5xx / overloaded, connection, timeout -> GenerationError
- refusal, cut-off or malformed reply                          -> GenerationError
- configuration bugs (bad key, bad request, unknown model)     -> NOT wrapped (500)

The API key is read by the SDK from ANTHROPIC_API_KEY; it never appears in this code.
"""

from typing import Literal

import anthropic
from anthropic import AsyncAnthropic, transform_schema
from anthropic.types.beta import (
    BetaJSONOutputFormatParam,
    BetaOutputConfigParam,
    BetaUserMessageParam,
)
from pydantic import ValidationError

from kc.generation.llm import GenerationError, LlmAnswer

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT: Effort = "medium"
DEFAULT_MAX_TOKENS = 4096  # thinking counts toward this limit too
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Failures worth "try again" (503). Everything else from the SDK is a bug or a config error.
_TEMPORARY_ERRORS: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,  # 429
    anthropic.InternalServerError,  # 5xx, including OverloadedError (529)
    anthropic.RequestTimeoutError,  # 408
    anthropic.APIConnectionError,  # network, including APITimeoutError
)


def _output_format() -> BetaJSONOutputFormatParam:
    """JSON schema for `LlmAnswer`, with a description written for the model.

    The SDK would otherwise send the class docstring, which is a note for developers.
    """
    schema = transform_schema(LlmAnswer)
    schema["description"] = (
        "Your reply to the question, based only on the passages. "
        "status: answered | not_found | needs_clarification. "
        "text: the answer in Markdown, citing passages as [n]. "
        "cited_passage_ids: every passage id cited in text."
    )
    return {"type": "json_schema", "schema": schema}


_OUTPUT_FORMAT = _output_format()


class ClaudeLlmClient:
    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str = DEFAULT_MODEL,
        effort: Effort = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    async def generate(self, *, system: str, user_message: str) -> LlmAnswer:
        message: BetaUserMessageParam = {"role": "user", "content": user_message}
        output_config: BetaOutputConfigParam = {"effort": self._effort, "format": _OUTPUT_FORMAT}
        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[message],
                output_config=output_config,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except _TEMPORARY_ERRORS as e:
            raise GenerationError(f"Claude request failed: {type(e).__name__}") from e

        # Check why the model stopped BEFORE reading the content.
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise GenerationError(f"Claude declined the request (category={category})")
        if response.stop_reason != "end_turn":
            raise GenerationError(f"Claude stopped early (stop_reason={response.stop_reason})")

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            return LlmAnswer.model_validate_json(text)
        except ValidationError as e:
            raise GenerationError("Claude reply did not match the LlmAnswer schema") from e
