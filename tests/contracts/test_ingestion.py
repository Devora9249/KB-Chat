"""Tests for the Ingestion contract (Document, Chunk)."""

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from contracts import PRODUCT_SUCCESSOR, Chunk, Document, IngestedDocument, Product

URL = "https://knowledgecenter.gcora.genpact.com/help/configure-saml"


def valid_document(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "url": URL,
        "title": "Configure SAML",
        "breadcrumb": ["Cora SeQuence 9.x", "Administration"],
        "product": "sequence",
        "versions": ["9.x"],
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
        "versions": ["9.x"],
        "doc_type": "how-to",
    }
    data.update(overrides)
    return data


# --- Document ---------------------------------------------------------------


def test_valid_document() -> None:
    doc = Document(**valid_document())
    assert doc.product == "sequence"
    assert doc.release is None


def test_versions_default_to_empty_list() -> None:
    data = valid_document()
    del data["versions"]
    assert Document(**data).versions == []


@pytest.mark.parametrize("product", ["Sequence", "SeQuence", "cora-sequence", ""])
def test_rejects_unknown_product(product: str) -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(product=product))


@pytest.mark.parametrize("doc_type", ["howto", "Release Notes", "faq"])
def test_rejects_unknown_doc_type(doc_type: str) -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(doc_type=doc_type))


@pytest.mark.parametrize("version", ["9", "9.0", "9.X", "v9.x", "9.9.2"])
def test_rejects_bad_version_format(version: str) -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(versions=[version]))


def test_accepts_several_versions() -> None:
    doc = Document(**valid_document(versions=["7.x", "8.x"]))
    assert doc.versions == ["7.x", "8.x"]


def test_rejects_bad_hash() -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(content_hash="not-a-hash"))


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(prodcut="sequence"))  # typo on purpose


KC = "https://knowledgecenter.gcora.genpact.com/help/"


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
    Document(**valid_document(url=KC + slug))


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
        Document(**valid_document(url=url))


# --- Chunk ------------------------------------------------------------------


def test_valid_chunk() -> None:
    chunk = Chunk(**valid_chunk())
    assert chunk.heading_path == ["Configure SAML"]


@pytest.mark.parametrize("chunk_id", ["123", "0123456789ABCDEF", "uuid-like-id-here"])
def test_rejects_bad_chunk_id(chunk_id: str) -> None:
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk(id=chunk_id))


def test_rejects_empty_chunk_text() -> None:
    with pytest.raises(ValidationError):
        Chunk(**valid_chunk(text=""))


# --- Value lists --------------------------------------------------------------


def test_product_successor_uses_known_products() -> None:
    known = set(get_args(Product))
    for legacy, current in PRODUCT_SUCCESSOR.items():
        assert legacy in known
        assert current in known


# --- IngestedDocument -----------------------------------------------------------


def test_ingested_document_valid() -> None:
    item = IngestedDocument(document=Document(**valid_document()), chunks=[Chunk(**valid_chunk())])
    assert len(item.chunks) == 1


def test_ingested_document_needs_chunks() -> None:
    with pytest.raises(ValidationError):
        IngestedDocument(document=Document(**valid_document()), chunks=[])


@pytest.mark.parametrize(
    "chunk_overrides",
    [
        {"document_url": KC + "glossary"},  # chunk of another page
        {"product": "orchestration"},  # label differs from the document
        {"versions": ["8.x"]},
        {"doc_type": "reference"},
    ],
)
def test_ingested_document_rejects_mismatched_chunk(chunk_overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        IngestedDocument(
            document=Document(**valid_document()),
            chunks=[Chunk(**valid_chunk(**chunk_overrides))],
        )


def test_ingested_document_rejects_duplicate_chunk_ids() -> None:
    chunk = Chunk(**valid_chunk())
    with pytest.raises(ValidationError):
        IngestedDocument(document=Document(**valid_document()), chunks=[chunk, chunk])
