"""Contracts for the Generation / API domain (owner: Leah).

Shared shapes for what `/v1/ask` returns, used by the UI (Chani), evaluation (Chaya)
and the API itself. Changes follow the contract rules in CLAUDE.md: add optional
fields only, approval from all six, and update the Fake.
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One source the answer relies on, referenced in the answer text as [id]."""

    id: int = Field(ge=1, description="Marker number used in the answer text, e.g. [1]")
    url: str = Field(description="Link to the Knowledge Center article")
    title: str = Field(description="Article title, shown to the user")
    chunk_id: str = Field(description="ID of the retrieved chunk the claim is based on")
