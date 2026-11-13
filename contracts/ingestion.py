"""Ingestion contract: the data the Ingestion pipeline hands to the rest of the system.

Owner: Devora. Consumers: Storage & Indexing (Raizy), Search & Ranking (Racheli).

DRAFT: the value lists below are a proposal and still need agreement with the team
(Raizy, Racheli) before this file is merged to `main`.

Open questions (to settle in the PR):
- content_hash: today it covers content_md only. If Enricher labels or chunking change
  while the text does not, change detection would skip the page. Proposal: hash content
  + labels + an ingestion pipeline version.
- Chunk.text does not repeat its headings. Proposal: the indexer embeds
  " > ".join(heading_path) + text, so a chunk keeps its context.
- Pages removed from the site: IngestionSource has no way to report them yet (Sprint 3).
"""

from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

# A normalized article URL, exactly as listed in the site's sitemap.xml:
#   https://knowledgecenter.gcora.genpact.com/help/<slug>
# The slug is lowercase letters, digits and hyphens (runs like "versions-77---82" exist).
# No trailing "/", no "?query", no "#fragment". Chunk ids are built from this URL, so
# one page must always have exactly one spelling.
KC_URL_PATTERN = r"^https://knowledgecenter\.gcora\.genpact\.com/help/[a-z0-9][a-z0-9-]*$"
KcUrl = Annotated[str, Field(pattern=KC_URL_PATTERN)]


class Document(BaseModel):
    """One cleaned page from the Knowledge Center."""

    model_config = ConfigDict(extra="forbid")  # a typo in a field name is an error

    url: KcUrl
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
    document_url: KcUrl
    text: str = Field(min_length=1)
    heading_path: list[str]  # headings above this chunk, e.g. ["Configure SAML", "Step 2"]
    # Copied from the parent Document so search can filter chunks directly.
    product: Product
    versions: list[Version] = Field(default_factory=list)
    doc_type: DocType


class IngestedDocument(BaseModel):
    """One page together with all of its chunks (indexed as one unit).

    Guarantees for consumers: there is at least one chunk, every chunk belongs to this
    document and carries the same labels (product, versions, doc_type), and chunk ids
    are unique within the document.
    """

    model_config = ConfigDict(extra="forbid")

    document: Document
    chunks: list[Chunk] = Field(min_length=1)

    @model_validator(mode="after")
    def _chunks_match_document(self) -> "IngestedDocument":
        doc = self.document
        labels = (doc.product, doc.versions, doc.doc_type)
        for chunk in self.chunks:
            if chunk.document_url != doc.url:
                raise ValueError(f"chunk {chunk.id} belongs to {chunk.document_url}, not {doc.url}")
            if (chunk.product, chunk.versions, chunk.doc_type) != labels:
                raise ValueError(f"chunk {chunk.id} labels differ from its document {doc.url}")
        if len({c.id for c in self.chunks}) != len(self.chunks):
            raise ValueError(f"duplicate chunk ids in {doc.url}")
        return self


class IngestionSource(Protocol):
    """Where indexing gets its documents from.

    Implementations: `SampleIngestionSource` (fixtures, for now) and later the real
    crawler-based source. Consumers depend only on this Protocol, so swapping one for
    the other is a one-line change.

    Usage:
        async for item in source.iter_documents():
            ...  # item.document, item.chunks
    """

    def iter_documents(self) -> AsyncIterator[IngestedDocument]: ...
