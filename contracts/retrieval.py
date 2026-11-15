"""Search & Ranking contracts: Filters, SearchResult and the Retriever protocol."""

from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.common import DocType, Product, Version


class Filters(BaseModel):
    """Metadata filters for search.

    Semantics:
    - A field set to None means "no filter on this field".
    - Within a field: OR (match any of the listed values).
    - Across fields: AND (all set fields must match).
    - When `versions` is set, items with version=None are NOT included.
    - `exclude_doc_types` removes items whose doc_type is listed.

    Invalid input raises (pydantic.ValidationError is a subclass of ValueError):
    - an empty list in any field (would silently yield zero results);
    - the same value in both `doc_types` and `exclude_doc_types`.

    Immutability: `frozen=True` freezes the fields themselves (no reassignment),
    not the contents of the lists. Do not mutate the lists after construction;
    build a new Filters(...) instead (note: `model_copy(update=...)` is not validated).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    products: list[Product] | None = None
    versions: list[Version] | None = None
    doc_types: list[DocType] | None = None
    exclude_doc_types: list[DocType] | None = None

    @field_validator("products", "versions", "doc_types", "exclude_doc_types")
    @classmethod
    def _reject_empty_list[T](cls, value: list[T] | None) -> list[T] | None:
        if value is not None and len(value) == 0:
            raise ValueError("empty list is not allowed; use None for 'no filter'")
        return value

    @model_validator(mode="after")
    def _reject_doc_type_overlap(self) -> Self:
        if self.doc_types is not None and self.exclude_doc_types is not None:
            overlap = set(self.doc_types) & set(self.exclude_doc_types)
            if overlap:
                raise ValueError(f"doc_types and exclude_doc_types overlap: {sorted(overlap)}")
        return self


class SearchResult(BaseModel):
    """A single retrieved chunk.

    `score`: higher is better; comparable only within the same result list/method.
    NaN and +/-Infinity are rejected (they break sorting).
    All fields are required; `version` is None for pages not tied to a version.
    """

    chunk_id: str
    url: str
    title: str
    text: str
    score: float = Field(allow_inf_nan=False)
    product: Product
    version: Version | None
    doc_type: DocType
    heading_path: list[str]


class Retriever(Protocol):
    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        """Return up to `k` results, best first.

        - `k` must be >= 1; otherwise raises ValueError.
        - Sorted by `score` descending; results with an equal score are sorted
          by `chunk_id` ascending, so the order is deterministic.
        - Each `chunk_id` appears at most once.
        - May return fewer than `k` results (e.g. when filters are restrictive).
        """
        ...
