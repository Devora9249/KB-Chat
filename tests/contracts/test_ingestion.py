"""Tests for the Ingestion contract (Document, Chunk)."""

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from contracts import PRODUCT_SUCCESSOR, Chunk, Document, Product

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


def test_rejects_non_https_url() -> None:
    with pytest.raises(ValidationError):
        Document(**valid_document(url="/help/configure-saml"))


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
