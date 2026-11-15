"""Local UI view models for an answer and its citations.

Temporary: these mirror what the UI needs to render. Once Leah's `Answer`/`Citation`
are agreed in `contracts/`, the UI should switch to those and this file goes away.
"""

from pydantic import BaseModel


class Citation(BaseModel):
    title: str
    url: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]
