"""InMemoryIndexStore-specific tests.

Contract behaviour (shared by every IndexStore) lives in tests/indexing/contract/.
Only implementation details stay here.
"""

import asyncio

import pytest

from kc.indexing._provisional_contracts import Chunk, Document, Filters
from kc.indexing.in_memory_index_store import InMemoryIndexStore


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        InMemoryIndexStore(dimension=0, model_name="m")
    with pytest.raises(ValueError):
        InMemoryIndexStore(dimension=3, model_name="")
    store = InMemoryIndexStore(dimension=3, model_name="fake-embedder-v1")
    assert (store.dimension, store.model_name) == (3, "fake-embedder-v1")


def test_filters_placeholder_has_no_fields() -> None:
    assert Filters.model_fields == {}


def test_text_search_score_is_number_of_matching_words() -> None:
    # The contract only fixes the ORDER; this exact scoring is the in-memory stand-in for
    # Postgres full-text rank.
    store = InMemoryIndexStore(dimension=3, model_name="m")
    doc = Document(
        url="u1",
        title="T",
        breadcrumb=[],
        product="sequence",
        version=None,
        doc_type="how-to",
        content_md="",
        content_hash="h",
    )

    def chunk(cid: str, text: str) -> Chunk:
        return Chunk(
            id=cid,
            document_url="u1",
            text=text,
            heading_path=[],
            product="sequence",
            version=None,
            doc_type="how-to",
        )

    chunks = [chunk("one", "SAML is configured here"), chunk("two", "configure saml: SAML")]
    asyncio.run(store.upsert_document(doc, chunks, [[1.0, 0, 0], [1.0, 0, 0]]))

    results = asyncio.run(store.text_search("saml"))

    assert [(r.chunk_id, r.score) for r in results] == [("two", 2.0), ("one", 1.0)]
