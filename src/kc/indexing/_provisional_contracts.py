"""TEMPORARY stand-ins for shared contracts that do not exist yet.

TODO(D11): replace every definition here with imports from `contracts/` once the
module layout (D11) and Filters (D10) are agreed. Then delete this file.

Rules:
- Import ONLY from inside `src/kc/indexing/` and from tests/indexing (Raizy's code).
  Other domains must never import this module.
- All Storage & Indexing code imports contract types from HERE, so the switch to
  `contracts/` is a one-line change per name.
"""

from typing import Literal

from pydantic import BaseModel


# =====================================================================================
# Copied verbatim from CLAUDE.md
# =====================================================================================
# fmt: off
class Document(BaseModel):
    url: str
    title: str
    breadcrumb: list[str]
    product: str            # sequence | orchestration | case-manager | opsmanager
    version: str | None
    doc_type: str           # how-to | release-notes | troubleshooting ...
    content_md: str
    content_hash: str

class Chunk(BaseModel):
    id: str
    document_url: str
    text: str
    heading_path: list[str]
    product: str
    version: str | None
    doc_type: str

class SearchResult(BaseModel):
    chunk_id: str
    url: str
    title: str
    text: str
    score: float
# fmt: on

# =====================================================================================
# From my storage contract draft (D2), not yet approved by the team
# =====================================================================================

# Required, keyword-only argument of Embedder.embed.
InputType = Literal["query", "document"]

# =====================================================================================
# Placeholder, NOT defined yet (D10): no fields on purpose
# =====================================================================================


class Filters(BaseModel):
    """Placeholder only, so signatures can say `Filters | None`.

    Fields are not agreed yet (D10). Stores accept only `None` and raise
    NotImplementedError for any Filters instance.
    """
