"""Tests for the indexing pipeline (B3), using the fakes via Dependency Injection."""

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from kc.indexing._provisional_contracts import (
    Chunk,
    Document,
    Embedder,
    EmbeddingError,
    IndexStore,
    InputType,
)
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.pipeline import index_document, index_documents

DIM = 32


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def make_doc(url: str, content_hash: str = "h1") -> Document:
    return Document(
        url=url,
        title=f"Title of {url}",
        breadcrumb=["Home"],
        product="sequence",
        version="9.x",
        doc_type="how-to",
        content_md="...",
        content_hash=content_hash,
    )


def make_chunk(chunk_id: str, url: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_url=url,
        text=text,
        heading_path=["H1"],
        product="sequence",
        version="9.x",
        doc_type="how-to",
    )


class SpyEmbedder:
    """Wraps FakeEmbedder; records every call and can be told to fail."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self._inner = FakeEmbedder(dimension=DIM)
        self.fail_for = fail_for or set()
        self.calls: list[tuple[list[str], InputType]] = []

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        self.calls.append((texts, input_type))
        if any(text in self.fail_for for text in texts):
            raise EmbeddingError("provider unavailable")
        return await self._inner.embed(texts, input_type=input_type)


def new_store() -> InMemoryIndexStore:
    return InMemoryIndexStore(dimension=DIM, model_name="fake-embedder-v1")


def chunk_ids(store: InMemoryIndexStore) -> list[str]:
    results = run(store.text_search("old new saml notes page", k=1000))
    return sorted(r.chunk_id for r in results)


# --- the fakes satisfy the draft protocols (checked by mypy too) --------------------------


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(new_store(), IndexStore)
    assert isinstance(SpyEmbedder(), Embedder)


# --- happy path -------------------------------------------------------------------------


def test_index_document_embeds_as_document_and_makes_it_searchable() -> None:
    embedder, store = SpyEmbedder(), new_store()
    chunks = [make_chunk("c1", "u1", "configure SAML"), make_chunk("c2", "u1", "release notes")]

    run(index_document(make_doc("u1"), chunks, embedder=embedder, store=store))

    assert embedder.calls == [(["configure SAML", "release notes"], "document")]
    assert run(store.get_document_hashes()) == {"u1": "h1"}
    (query,) = run(embedder.embed(["configure SAML"], input_type="query"))
    assert run(store.vector_search(query, k=1))[0].chunk_id == "c1"


def test_document_without_chunks_is_recorded() -> None:
    store = new_store()
    run(index_document(make_doc("u1"), [], embedder=SpyEmbedder(), store=store))
    assert run(store.get_document_hashes()) == {"u1": "h1"}


# --- EmbeddingError: the old version stays ------------------------------------------------


def test_embedding_error_keeps_previous_version_in_index() -> None:
    store = new_store()
    run(
        index_document(
            make_doc("u1", "v1"),
            [make_chunk("old", "u1", "old page")],
            embedder=SpyEmbedder(),
            store=store,
        )
    )

    failing = SpyEmbedder(fail_for={"new page"})
    with pytest.raises(EmbeddingError):
        run(
            index_document(
                make_doc("u1", "v2"),
                [make_chunk("new", "u1", "new page")],
                embedder=failing,
                store=store,
            )
        )

    assert run(store.get_document_hashes()) == {"u1": "v1"}  # old hash
    assert chunk_ids(store) == ["old"]  # old chunks still searchable


def test_embedding_error_on_new_document_leaves_it_absent() -> None:
    store = new_store()
    with pytest.raises(EmbeddingError):
        run(
            index_document(
                make_doc("u1"),
                [make_chunk("c1", "u1", "boom")],
                embedder=SpyEmbedder(fail_for={"boom"}),
                store=store,
            )
        )
    assert run(store.get_document_hashes()) == {}


# --- pre-check before paying for embeddings -----------------------------------------------


def test_chunk_of_another_document_fails_before_embedding() -> None:
    embedder, store = SpyEmbedder(), new_store()
    run(
        index_document(
            make_doc("u1", "v1"),
            [make_chunk("old", "u1", "old page")],
            embedder=embedder,
            store=store,
        )
    )
    embedder.calls.clear()

    with pytest.raises(ValueError, match="document_url"):
        run(
            index_document(
                make_doc("u1", "v2"),
                [make_chunk("x", "OTHER", "new page")],
                embedder=embedder,
                store=store,
            )
        )

    assert embedder.calls == []  # never paid for embeddings
    assert run(store.get_document_hashes()) == {"u1": "v1"}


# --- many documents: one failure does not stop the run --------------------------------------


def test_index_documents_continues_after_failure_and_reports() -> None:
    store = new_store()
    run(
        index_document(
            make_doc("u2", "v1"),
            [make_chunk("old2", "u2", "old page")],
            embedder=SpyEmbedder(),
            store=store,
        )
    )
    items = [
        (make_doc("u1"), [make_chunk("a", "u1", "saml")]),
        (make_doc("u2", "v2"), [make_chunk("b", "u2", "bad")]),  # embedding fails
        (make_doc("u3"), [make_chunk("c", "OTHER", "notes")]),  # wrong document_url
        (make_doc("u4"), [make_chunk("d", "u4", "notes")]),
    ]

    report = run(index_documents(items, embedder=SpyEmbedder(fail_for={"bad"}), store=store))

    assert report.indexed == ["u1", "u4"]
    assert set(report.failed) == {"u2", "u3"}
    assert report.failed["u2"].startswith("EmbeddingError")
    assert report.failed["u3"].startswith("ValueError")
    hashes = run(store.get_document_hashes())
    assert hashes == {"u1": "h1", "u2": "v1", "u4": "h1"}  # u2 keeps its old version


def test_unexpected_errors_are_not_swallowed() -> None:
    class BrokenEmbedder(SpyEmbedder):
        async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
            raise RuntimeError("bug, not a provider failure")

    with pytest.raises(RuntimeError, match="bug"):
        run(
            index_documents(
                [(make_doc("u1"), [make_chunk("a", "u1", "x")])],
                embedder=BrokenEmbedder(),
                store=new_store(),
            )
        )
