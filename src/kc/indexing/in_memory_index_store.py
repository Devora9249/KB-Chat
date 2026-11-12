"""In-memory IndexStore for tests and local development (draft storage contract D6-D9).

Storage: dict url -> (Document, [(Chunk, vector), ...]).

- vector_search: cosine similarity against every stored vector.
- text_search: a simple stand-in for Postgres full-text search: score = how many times
  query words appear in the chunk text (case-insensitive); chunks with score 0 are left out.
- Results: score descending, ties broken by chunk_id, at most k.

Atomic replace: upsert_document validates ALL input first, builds the new entry aside,
then swaps it into the dict in one assignment. If validation fails nothing changes.
Inputs are copied, so later changes by the caller do not leak into the store.

Embedding-model guard: the real PgIndexStore compares model_name/dimension with the
embedding_config table; here there is no table, so model_name is only kept.
"""

import re
from dataclasses import dataclass

from kc.indexing._provisional_contracts import Chunk, Document, Filters, SearchResult
from kc.indexing.vector_math import cosine_similarity

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class _StoredDocument:
    document: Document
    entries: tuple[tuple[Chunk, tuple[float, ...]], ...]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def _check_filters(filters: Filters | None) -> None:
    if filters is not None:
        raise NotImplementedError("Filters not agreed yet (D10)")


class InMemoryIndexStore:
    def __init__(self, *, dimension: int, model_name: str) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not model_name:
            raise ValueError("model_name must not be empty")
        self._dimension = dimension
        self._model_name = model_name
        self._documents: dict[str, _StoredDocument] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def upsert_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        # 1. Validate everything before touching the store.
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if len(vector) != self._dimension:
                raise ValueError(
                    f"vector {i} has dimension {len(vector)}, index dimension is {self._dimension}"
                )
            if chunk.document_url != document.url:
                raise ValueError(
                    f"chunk {chunk.id!r} has document_url {chunk.document_url!r}, "
                    f"expected {document.url!r}"
                )

        # 2. Build the new entry aside, from copies.
        stored = _StoredDocument(
            document=document.model_copy(deep=True),
            entries=tuple(
                (chunk.model_copy(deep=True), tuple(vector))
                for chunk, vector in zip(chunks, vectors, strict=True)
            ),
        )

        # 3. Swap in one assignment: old chunks of this document disappear entirely.
        self._documents[document.url] = stored

    async def delete_document(self, url: str) -> None:
        self._documents.pop(url, None)

    async def get_document_hashes(self) -> dict[str, str]:
        return {url: stored.document.content_hash for url, stored in self._documents.items()}

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        _check_k(k)
        _check_filters(filters)
        if len(vector) != self._dimension:
            raise ValueError(
                f"query vector has dimension {len(vector)}, index dimension is {self._dimension}"
            )
        scored = [
            (cosine_similarity(vector, stored_vector), chunk, stored.document)
            for stored in self._documents.values()
            for chunk, stored_vector in stored.entries
        ]
        return self._top_k(scored, k)

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        _check_k(k)
        _check_filters(filters)
        query_terms = set(_tokens(query))
        if not query_terms:
            return []
        scored = []
        for stored in self._documents.values():
            for chunk, _vector in stored.entries:
                score = sum(1 for token in _tokens(chunk.text) if token in query_terms)
                if score > 0:
                    scored.append((float(score), chunk, stored.document))
        return self._top_k(scored, k)

    @staticmethod
    def _top_k(scored: list[tuple[float, Chunk, Document]], k: int) -> list[SearchResult]:
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            SearchResult(
                chunk_id=chunk.id,
                url=chunk.document_url,
                title=document.title,
                text=chunk.text,
                score=score,
            )
            for score, chunk, document in scored[:k]
        ]
