"""Ingestion contract: the data the Ingestion pipeline hands to the rest of the system.

Owner: Devora. Consumers: Storage & Indexing (Raizy), and through the index, Search.

Value lists (Product, DocType, Version) are shared and live in `contracts.common`.

Two levels of checks:
- `Document` and `Chunk` check the closed value lists only, so other domains can build
  simple test data (e.g. url="u1").
- `IngestedDocument`, what Ingestion actually produces, also checks URL, hash and chunk
  id formats and that every chunk belongs to its document.

Open questions (to settle in the PR):
- content_hash: today it covers content_md only. If Enricher labels or chunking change
  while the text does not, change detection would skip the page. Proposal: hash content
  + labels + an ingestion pipeline version.
- Chunk.text does not repeat its headings. Proposal: the indexer embeds
  " > ".join(heading_path) + text, so a chunk keeps its context.
- Pages that are not tied to one product (e.g. Glossary) have no valid `product`.
  Proposal: add "general" to contracts.common.Product (backward compatible).
- Pages that cover several versions (e.g. "Versions 7.7 - 8.2") get one `version`.
  Revisit after the full crawl.
- Pages removed from the site: IngestionSource has no way to report them yet (Sprint 3).
"""

import re
from collections.abc import AsyncIterator
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.common import DocType, Product, Version

# A normalized article URL, exactly as listed in the site's sitemap.xml:
#   https://knowledgecenter.gcora.genpact.com/help/<slug>
# The slug is lowercase letters, digits and hyphens (runs like "versions-77---82" exist).
# No trailing "/", no "?query", no "#fragment". Chunk ids are built from this URL, so
# one page must always have exactly one spelling.
KC_URL_PATTERN = r"^https://knowledgecenter\.gcora\.genpact\.com/help/[a-z0-9][a-z0-9-]*$"

# sha256 hex of content_md.
CONTENT_HASH_PATTERN = r"^[0-9a-f]{64}$"

# Deterministic and unique across the whole index:
#   sha256(f"{document_url}|{' > '.join(heading_path)}|{part}")[:16]
CHUNK_ID_PATTERN = r"^[0-9a-f]{16}$"


class Document(BaseModel):
    """One cleaned page from the Knowledge Center."""

    model_config = ConfigDict(extra="forbid")  # a typo in a field name is an error

    url: str  # normalized article URL, see KC_URL_PATTERN
    title: str = Field(min_length=1)
    breadcrumb: list[str]  # e.g. ["Cora SeQuence 9.x", "Administration", "SAML"]
    product: Product
    version: Version | None  # major version, e.g. "9"; None = not version-specific
    doc_type: DocType
    content_md: str = Field(min_length=1)  # page body as Markdown
    content_hash: str  # see CONTENT_HASH_PATTERN


class Chunk(BaseModel):
    """One piece of a Document (~300-600 tokens), cut at heading boundaries."""

    model_config = ConfigDict(extra="forbid")

    id: str  # see CHUNK_ID_PATTERN
    document_url: str
    text: str = Field(min_length=1)
    heading_path: list[str]  # headings above this chunk, e.g. ["Configure SAML", "Step 2"]
    # Copied from the parent Document so search can filter chunks directly.
    product: Product
    version: Version | None
    doc_type: DocType


class IngestedDocument(BaseModel):
    """One page together with all of its chunks (indexed as one unit).

    Guarantees for consumers:
    - the URL is normalized, content_hash is a sha256 hex, chunk ids have the id format;
    - there is at least one chunk;
    - every chunk belongs to this document and carries the same product, version and
      doc_type;
    - chunk ids are unique within the document (and, since the URL is part of the id,
      across the whole index).
    """

    model_config = ConfigDict(extra="forbid")

    document: Document
    chunks: list[Chunk] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        doc = self.document
        if not re.fullmatch(KC_URL_PATTERN, doc.url):
            raise ValueError(f"url is not a normalized article URL: {doc.url!r}")
        if not re.fullmatch(CONTENT_HASH_PATTERN, doc.content_hash):
            raise ValueError(f"content_hash is not a sha256 hex: {doc.url}")
        labels = (doc.product, doc.version, doc.doc_type)
        for chunk in self.chunks:
            if not re.fullmatch(CHUNK_ID_PATTERN, chunk.id):
                raise ValueError(f"bad chunk id format: {chunk.id!r}")
            if chunk.document_url != doc.url:
                raise ValueError(f"chunk {chunk.id} belongs to {chunk.document_url}, not {doc.url}")
            if (chunk.product, chunk.version, chunk.doc_type) != labels:
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
