"""TEMPORARY stand-ins for shared contracts that do not exist yet.

TODO(D11): replace every definition here with imports from `contracts/` once the
module layout (D11) is agreed. Then delete this file.

Rules:
- Import ONLY from inside `src/kc/indexing/` and from tests/indexing (Raizy's code).
  Other domains must never import this module.
- All Storage & Indexing code imports contract types from HERE, so the switch to
  `contracts/` is a one-line change per name.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator


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
# Filters: Agreed with Racheli (D10), 2026-11-12. Owner: Racheli.
# Will be replaced by contracts import (D11).
# =====================================================================================

# TODO(contracts): Product, Version and DocType will be defined ONCE in the shared
#     contracts (Racheli + Devora) and used by Document, Chunk and Filters alike.
#     Plain `str` until then.
Product = str  # sequence | orchestration | case-manager | opsmanager (values TBD)
Version = str  # exact match; allowed values: "11", "10", "9", "8", "7", "4" (major only)
DocType = str  # how-to | release-notes | troubleshooting ... (values TBD)


class Filters(BaseModel):
    """Search filters. Agreed with Racheli (D10), 2026-11-12. Owner: Racheli.

    Semantics:
    - a field set to None does not filter;
    - several values in one field = OR; different fields = AND;
    - exclude_doc_types removes chunks of those types;
    - filtering by `versions` never returns chunks with version=None (deliberate).

    Validation lives HERE (stores do not re-check): an empty list, or the same value in
    both doc_types and exclude_doc_types, raises a ValidationError (a ValueError).
    """

    products: list[Product] | None = None
    versions: list[Version] | None = None
    doc_types: list[DocType] | None = None
    exclude_doc_types: list[DocType] | None = None

    @field_validator("products", "versions", "doc_types", "exclude_doc_types")
    @classmethod
    def _reject_empty_list(cls, value: list[str] | None, info: ValidationInfo) -> list[str] | None:
        if value is not None and not value:
            raise ValueError(f"{info.field_name} must be None (no filter) or a non-empty list")
        return value

    @model_validator(mode="after")
    def _reject_include_exclude_overlap(self) -> "Filters":
        if self.doc_types is not None and self.exclude_doc_types is not None:
            overlap = sorted(set(self.doc_types) & set(self.exclude_doc_types))
            if overlap:
                raise ValueError(f"doc_types and exclude_doc_types overlap: {overlap}")
        return self


# =====================================================================================
# From my storage contract draft (D1-D9), not yet approved by the team
# (defined after Filters because the protocols refer to it)
# =====================================================================================


class EmbeddingError(RuntimeError):
    """The embedding provider failed after retries (network, quota, invalid input)."""


@runtime_checkable
class Embedder(Protocol):
    """See storage contract draft D1-D5: one unit-length vector per text, same order."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]: ...


@runtime_checkable
class IndexStore(Protocol):
    """See storage contract draft D6-D9."""

    async def upsert_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None: ...

    async def delete_document(self, url: str) -> None: ...

    async def get_document_hashes(self) -> dict[str, str]: ...

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]: ...

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]: ...
