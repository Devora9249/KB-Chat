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


class AskRequest(BaseModel):
    """What a client (UI, eval) sends to `/v1/ask`.

    Open question for the team, product/version selection:
    Users will often know their product and version (e.g. "SeQuence 9"), and passing
    them lets search filter out same-named articles from other products (sensitive
    point #1 in CLAUDE.md). We deliberately did NOT add `product` / `version` fields
    yet, because they overlap with `Filters`, which is still to be agreed in Sprint 0.
    Once `Filters` is defined, the plan is to add an optional `filters: Filters | None`
    field here (optional, so this is a backward-compatible change). Until then, product
    and version are detected from the question text by query understanding.
    """

    question: str = Field(min_length=1, max_length=2000, description="The user's question")


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
