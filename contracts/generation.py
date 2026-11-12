"""Contracts for the Generation / API domain (owner: Leah).

Shared shapes for what `/v1/ask` returns, used by the UI (Chani), evaluation (Chaya)
and the API itself. Changes follow the contract rules in CLAUDE.md: add optional
fields only, approval from all six, and update the Fake.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

AnswerStatus = Literal["answered", "not_found", "needs_clarification"]


class Citation(BaseModel):
    """One source the answer relies on, referenced in the answer text as [id]."""

    id: int = Field(ge=1, description="Marker number used in the answer text, e.g. [1]")
    url: str = Field(description="Link to the Knowledge Center article")
    title: str = Field(description="Article title, shown to the user")
    chunk_id: str = Field(description="ID of the retrieved chunk the claim is based on")


class Answer(BaseModel):
    """What `/v1/ask` returns."""

    text: str = Field(description="Answer text in Markdown, with [n] citation markers")
    citations: list[Citation] = Field(default_factory=list)
    status: AnswerStatus = Field(
        description=(
            "answered: grounded answer with citations; "
            "not_found: the knowledge base has no answer; "
            "needs_clarification: text is a question back to the user"
        )
    )

    @model_validator(mode="after")
    def answered_requires_citations(self) -> Self:
        # Grounding rule: an answer must be backed by at least one source.
        if self.status == "answered" and not self.citations:
            raise ValueError("status 'answered' requires at least one citation")
        return self
