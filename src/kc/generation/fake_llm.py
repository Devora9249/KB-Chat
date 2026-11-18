"""A scriptable LlmClient for tests and local development. Never calls a real model."""

from dataclasses import dataclass

from kc.generation.llm import LlmAnswer

DEFAULT_FAKE_ANSWER = LlmAnswer(
    status="answered",
    text="[FAKE] This answer is generated without a model, from passage [1].",
    cited_passage_ids=[1],
)


@dataclass(frozen=True)
class LlmCall:
    """One recorded call, so tests can check what was sent."""

    system: str
    user_message: str


class FakeLlmClient:
    """Returns a fixed answer (or raises a fixed error) and records every call.

    - `FakeLlmClient()` returns DEFAULT_FAKE_ANSWER.
    - `FakeLlmClient(answer=...)` returns that answer.
    - `FakeLlmClient(error=...)` raises that error (e.g. GenerationError, to test 503).
    """

    def __init__(
        self, answer: LlmAnswer = DEFAULT_FAKE_ANSWER, error: Exception | None = None
    ) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[LlmCall] = []

    async def generate(self, *, system: str, user_message: str) -> LlmAnswer:
        self.calls.append(LlmCall(system=system, user_message=user_message))
        if self._error is not None:
            raise self._error
        return self._answer.model_copy(deep=True)
