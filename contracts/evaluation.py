"""Contracts for the Evaluation & Quality domain (owner: Chaya).

`EvalCase` is one question in the Golden Set: a question with a known correct answer,
used to measure retrieval (did search find the right article?) and generation (is the
answer correct?). Changes follow the contract rules in CLAUDE.md: add optional fields
only, approval from all six.
"""

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One Golden Set question with its expected result."""

    question: str = Field(description="The question as a user asks it")
    expected_urls: list[str] = Field(
        description="Knowledge Center articles that answer the question"
    )
    product: str | None = Field(
        description=(
            "Product the question is about: sequence | orchestration | case-manager | "
            "opsmanager; None if the question is not product-specific"
        )
    )
    reference_answer: str | None = Field(
        description="Correct answer written by a human, used to grade generation; None if absent"
    )
