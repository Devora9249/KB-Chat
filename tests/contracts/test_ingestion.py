"""Tests for the Ingestion contract (Document, Chunk, IngestedDocument)."""

from typing import Any

import pytest
from pydantic import ValidationError

from contracts.ingestion import Chunk, Document, IngestedDocument

KC = "https://knowledgecenter.gcora.genpact.com/help/"
URL = KC + "configure-saml"


def valid_document(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "url": URL,
        "title": "Configure SAML",
        "breadcrumb": ["Cora SeQuence 9.x", "Administration"],
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
        "content_md": "# Configure SAML\n\nStep 1...",
        "content_hash": "a" * 64,
    }
    data.update(overrides)
    return data


def valid_chunk(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "0123456789abcdef",
        "document_url": URL,
        "text": "Step 1...",
        "heading_path": ["Configure SAML"],
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
    }
    data.update(overrides)
    return data


def ingested(
    doc: dict[str, Any] | None = None, chunk: dict[str, Any] | None = None
) -> IngestedDocument:
    return IngestedDocument(
        document=Document(**valid_document(**(doc or {}))),
        chunks=[Chunk(**valid_chunk(**(chunk or {})))],
    )


# --- Document and Chunk: closed value lists ------------------------------------


def test_valid_document_and_chunk() -> None:
    assert Document(**valid_document()).version == "9"
    assert Chunk(**valid_chunk()).heading_path == ["Configure SAML"]


def test_version_can_be_none() -> None:
    assert Document(**valid_document(version=None)).version is None


@pytest.mark.parametrize("product", ["Sequence", "SeQuence", "cora-sequence", ""])
def test_rejects_unknown_product(product: str) -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(product=product))


@pytest.mark.parametrize("doc_type", ["howto", "Release Notes", "faq"])
def test_rejects_unknown_doc_type(doc_type: str) -> None:
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk(doc_type=doc_type))


@pytest.mark.parametrize("version", ["9.x", "9.0", "v9", "9.9.2"])
def test_rejects_bad_version(version: str) -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(version=version))


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(prodcut="sequence"))  # typo on purpose


def test_rejects_empty_chunk_text() -> None:
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk(text=""))


def test_simple_test_data_is_allowed_outside_ingested_document() -> None:
    # Other domains build simple test data; only IngestedDocument checks formats.
    Document(**valid_document(url="u1", content_hash="h"))
    Chunk(**valid_chunk(id="seq-howto-9", document_url="u1"))


# --- IngestedDocument: what Ingestion produces ---------------------------------


def test_ingested_document_valid() -> None:
    assert len(ingested().chunks) == 1


@pytest.mark.parametrize(
    "slug",
    [
        "glossary",
        "corasequence-992-release-notes",
        "versions-77---82",  # real slug with a run of hyphens
        "release-notes-00f433e",  # real slug with a disambiguation suffix
        "10x",
    ],
)
def test_accepts_real_site_urls(slug: str) -> None:
    ingested(doc={"url": KC + slug}, chunk={"document_url": KC + slug})


@pytest.mark.parametrize(
    "url",
    [
        "/help/glossary",  # relative
        "http://knowledgecenter.gcora.genpact.com/help/glossary",  # not https
        "https://example.com/help/glossary",  # other host
        KC + "glossary/",  # trailing slash
        KC + "glossary#intro",  # fragment
        KC + "glossary?x=1",  # query
        KC + "Glossary",  # uppercase
        KC,  # no slug
        KC + "a/b",  # nested path
    ],
)
def test_rejects_non_normalized_url(url: str) -> None:
    with pytest.raises(ValidationError):
        ingested(doc={"url": url}, chunk={"document_url": url})


def test_rejects_bad_hash() -> None:
    with pytest.raises(ValidationError):
        ingested(doc={"content_hash": "not-a-hash"})


@pytest.mark.parametrize("chunk_id", ["123", "0123456789ABCDEF", "uuid-like-id-here"])
def test_rejects_bad_chunk_id(chunk_id: str) -> None:
    with pytest.raises(ValidationError):
        ingested(chunk={"id": chunk_id})


def test_needs_chunks() -> None:
    with pytest.raises(ValidationError):
        IngestedDocument(document=Document(**valid_document()), chunks=[])


@pytest.mark.parametrize(
    "chunk_overrides",
    [
        {"document_url": KC + "glossary"},  # chunk of another page
        {"product": "orchestration"},  # label differs from the document
        {"version": "8"},
        {"version": None},
        {"doc_type": "reference"},
    ],
)
def test_rejects_mismatched_chunk(chunk_overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ingested(chunk=chunk_overrides)


def test_rejects_duplicate_chunk_ids() -> None:
    chunk = Chunk(**valid_chunk())
    with pytest.raises(ValidationError):
        IngestedDocument(document=Document(**valid_document()), chunks=[chunk, chunk])
