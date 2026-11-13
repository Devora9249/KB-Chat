"""Ingestion contract: the data the Ingestion pipeline hands to the rest of the system.

Owner: Devora. Consumers: Storage & Indexing (Raizy), Search & Ranking (Racheli).

DRAFT: the value lists below are a proposal and still need agreement with the team
(Raizy, Racheli) before this file is merged to `main`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Closed value lists. Search filters match these strings exactly, so the Enricher
# may only produce values from these lists ("Sequence" or "SeQuence" are rejected).
# ---------------------------------------------------------------------------

Product = Literal[
    "orchestration",  # Cora Orchestration 10.x, Orchestration AI 11.x
    "case-manager",  # Cora Case Manager 4.x
    "sequence",  # Cora SeQuence 7.x-9.x (legacy name of Orchestration)
    "opsmanager",  # Cora OpsManager (legacy name of Case Manager)
    "general",  # pages not tied to one product (glossary, support matrix)
]

DocType = Literal[
    "how-to",  # "Configure...", "Create...", "Add..."
    "reference",  # PowerShell functions, expressions, API
    "installation",  # installation, deployment, pre-deployment requirements
    "troubleshooting",  # troubleshooting, known issues
    "release-notes",  # down-weighted in search (they flood results)
    "concept",  # overviews and explanations
    "other",  # anything that fits none of the above
]

# Legacy product -> its current name. We keep both values in the data (their content
# differs between versions), but search can use this map to widen a query if needed.
PRODUCT_SUCCESSOR: dict[Product, Product] = {
    "sequence": "orchestration",
    "opsmanager": "case-manager",
}

# A product line as shown in the site navigation: "9.x", "10.x", "11.x", "4.x".
VERSION_PATTERN = r"^\d+\.x$"
Version = Annotated[str, Field(pattern=VERSION_PATTERN)]

HttpsUrl = Annotated[str, Field(pattern=r"^https://")]


class Document(BaseModel):
    """One cleaned page from the Knowledge Center."""

    model_config = ConfigDict(extra="forbid")  # a typo in a field name is an error

    url: HttpsUrl  # normalized: no #fragment, no trailing "/"
    title: str = Field(min_length=1)
    breadcrumb: list[str]  # e.g. ["Cora SeQuence 9.x", "Administration", "SAML"]
    product: Product
    # Product lines this page applies to. Empty list = not version-specific.
    versions: list[Version] = Field(default_factory=list)
    # Exact release, only for release notes, e.g. "9.9.2".
    release: str | None = None
    doc_type: DocType
    content_md: str = Field(min_length=1)  # page body as Markdown
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")  # sha256 hex of content_md


class Chunk(BaseModel):
    """One piece of a Document (~300-600 tokens), cut at heading boundaries."""

    model_config = ConfigDict(extra="forbid")

    # Deterministic and unique across the whole index:
    # sha256(f"{document_url}|{' > '.join(heading_path)}|{part}")[:16]
    id: str = Field(pattern=r"^[0-9a-f]{16}$")
    document_url: HttpsUrl
    text: str = Field(min_length=1)
    heading_path: list[str]  # headings above this chunk, e.g. ["Configure SAML", "Step 2"]
    # Copied from the parent Document so search can filter chunks directly.
    product: Product
    versions: list[Version] = Field(default_factory=list)
    doc_type: DocType
