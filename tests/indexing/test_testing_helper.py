"""Tests for kc.indexing.testing (the public test helper other teams use to load an index).

Written the way another team uses it: only kc.indexing.testing, contracts and the fakes.
"""

import asyncio
import hashlib
import re
from typing import Any

import pytest

from contracts.retrieval import Filters
from contracts.storage import EmbeddingError, InputType
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.testing import SampleDocument, sample_document, seed_index

BASE = "https://knowledgecenter.gcora.genpact.com/help/"
SAML_URL = BASE + "configure-saml-sequence"


def _saml(**overrides: Any) -> SampleDocument:
    arguments: dict[str, Any] = {
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
    }
    arguments.update(overrides)
    return sample_document(
        SAML_URL,
        "Configure SAML Authentication",
        [
            "Use SAML to sign in with your identity provider.",
            "Run Set-CoraSeQuenceSiteAuthentication -AuthenticationType SAML.",
        ],
        **arguments,
    )


def _store_and_embedder() -> tuple[InMemoryIndexStore, FakeEmbedder]:
    embedder = FakeEmbedder(dimension=64)
    return InMemoryIndexStore(dimension=64, model_name=embedder.model_name), embedder


# --- sample_document ---------------------------------------------------------------------------


def test_chunk_ids_follow_d12() -> None:
    document = _saml()
    expected = tuple(
        hashlib.sha256(f"{SAML_URL}|Configure SAML Authentication|{part}".encode()).hexdigest()[:16]
        for part in (0, 1)
    )
    assert document.chunk_ids == expected
    assert all(re.fullmatch(r"[0-9a-f]{16}", chunk_id) for chunk_id in document.chunk_ids)


def test_chunk_ids_are_deterministic_and_depend_on_url_and_heading_path() -> None:
    assert _saml().chunk_ids == _saml().chunk_ids
    other_heading = _saml(heading_path=["Configure SAML Authentication", "Enable SAML"])
    assert set(other_heading.chunk_ids).isdisjoint(_saml().chunk_ids)
    other_url = sample_document(
        BASE + "saml-copy",
        "Configure SAML Authentication",
        ["x"],
        product="sequence",
        doc_type="how-to",
    )
    assert other_url.chunk_ids[0] not in _saml().chunk_ids


def test_fields_and_defaults() -> None:
    document = sample_document(
        BASE + "glossary", "Glossary", ["Term"], product="orchestration", doc_type="reference"
    )
    assert (document.url, document.title) == (BASE + "glossary", "Glossary")
    assert (document.product, document.version, document.doc_type) == (
        "orchestration",
        None,
        "reference",
    )
    assert document.chunk_texts == ("Term",)


@pytest.mark.parametrize(
    "url",
    [
        "https://kc.example/help/saml",  # another site
        "http://knowledgecenter.gcora.genpact.com/help/saml",  # not https
        BASE + "saml/",  # trailing slash
        BASE + "Configure-SAML",  # uppercase
        BASE + "saml?lang=en",  # query
        BASE + "saml#step-2",  # fragment
        BASE,  # no slug
    ],
)
def test_url_must_be_a_knowledge_center_article(url: str) -> None:
    with pytest.raises(ValueError, match=r"knowledgecenter\.gcora\.genpact\.com/help/<slug>"):
        sample_document(url, "Title", ["text"], product="sequence", doc_type="how-to")


def test_empty_chunk_texts_and_single_string_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sample_document(SAML_URL, "T", [], product="sequence", doc_type="how-to")
    with pytest.raises(ValueError, match="not a single string"):
        sample_document(SAML_URL, "T", "one text", product="sequence", doc_type="how-to")


def test_invalid_label_is_rejected() -> None:
    bad: Any = "SeQuence"
    with pytest.raises(ValueError):  # pydantic.ValidationError is a ValueError
        sample_document(SAML_URL, "T", ["x"], product=bad, doc_type="how-to")


# --- seed_index ----------------------------------------------------------------------------------


def test_seeded_documents_are_found_by_both_searches_with_their_chunk_ids() -> None:
    store, embedder = _store_and_embedder()
    saml = _saml()
    notes = sample_document(
        BASE + "release-notes-orchestration-11",
        "Release Notes 11",
        ["New features and fixed issues."],
        product="orchestration",
        version="11",
        doc_type="release-notes",
    )

    asyncio.run(seed_index(store, embedder, [saml, notes]))

    found = asyncio.run(store.text_search("Set-CoraSeQuenceSiteAuthentication", k=5))
    assert [(r.chunk_id, r.url, r.title) for r in found] == [
        (saml.chunk_ids[1], SAML_URL, "Configure SAML Authentication")
    ]
    (query,) = asyncio.run(embedder.embed(["fixed issues"], input_type="query"))
    filtered = asyncio.run(store.vector_search(query, Filters(products=["orchestration"]), k=5))
    assert [r.chunk_id for r in filtered] == list(notes.chunk_ids)
    assert asyncio.run(store.get_document_hashes()).keys() == {SAML_URL, notes.url}


def test_seed_index_embeds_as_documents() -> None:
    seen: list[InputType] = []

    class Recording(FakeEmbedder):
        async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
            seen.append(input_type)
            return await super().embed(texts, input_type=input_type)

    embedder = Recording(dimension=64)
    store = InMemoryIndexStore(dimension=64, model_name=embedder.model_name)
    asyncio.run(seed_index(store, embedder, [_saml()]))
    assert seen == ["document"]


def test_duplicate_url_is_rejected_before_anything_is_indexed() -> None:
    store, embedder = _store_and_embedder()
    with pytest.raises(ValueError, match="more than once"):
        asyncio.run(seed_index(store, embedder, [_saml(), _saml(version="8")]))
    assert asyncio.run(store.get_document_hashes()) == {}


def test_embedding_errors_are_not_swallowed() -> None:
    class Broken(FakeEmbedder):
        async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
            raise EmbeddingError("provider down")

    store = InMemoryIndexStore(dimension=64, model_name="m")
    with pytest.raises(EmbeddingError, match="provider down"):
        asyncio.run(seed_index(store, Broken(dimension=64), [_saml()]))


def test_accepts_any_iterable_including_a_generator() -> None:
    store, embedder = _store_and_embedder()
    asyncio.run(seed_index(store, embedder, (d for d in [_saml()])))
    assert list(asyncio.run(store.get_document_hashes())) == [SAML_URL]
