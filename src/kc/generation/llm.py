"""The language-model boundary of the Generation domain.

`LlmClient` is the only thing the answerer knows about the model: it sends the prompt and
gets back an `LlmAnswer`. The real implementation calls Claude; tests use `FakeLlmClient`.
Internal to this domain (not in contracts/): only the answerer depends on it.
"""

from typing import Protocol

from pydantic import BaseModel

from contracts.generation import AnswerStatus


class LlmAnswer(BaseModel):
    """The exact shape the model must reply in (enforced with structured output).

    Validation of the content (do the cited ids exist? is "answered" really cited?) is the
    answerer's job, not the model's: never trust model output blindly.
    """

    status: AnswerStatus
    text: str
    cited_passage_ids: list[int]


class GenerationError(RuntimeError):
    """The model could not be reached or did not produce an answer (temporary failure).

    Raised for infrastructure problems only (overloaded, rate limited, connection, timeout).
    The API maps it to 503. Configuration bugs (bad API key, bad request) are NOT wrapped,
    so they surface as 500. Implementations raise it with `raise GenerationError(...) from e`.
    """


class LlmClient(Protocol):
    async def generate(self, *, system: str, user_message: str) -> LlmAnswer: ...
