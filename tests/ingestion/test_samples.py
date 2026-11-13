"""Tests for the sample fixtures teammates build against."""

from fixtures.build_samples_v0 import build
from kc.ingestion.samples import load_sample_chunks, load_sample_documents


def test_files_load_and_match_contract() -> None:
    # model_validate_json inside the loaders raises if any line breaks the contract
    assert len(load_sample_documents()) == 10
    assert len(load_sample_chunks()) > 10


def test_files_are_up_to_date_with_builder() -> None:
    # If this fails: run `uv run python fixtures/build_samples_v0.py` and commit the files.
    docs, chunks = build()
    assert load_sample_documents() == docs
    assert load_sample_chunks() == chunks


def test_chunk_ids_are_unique() -> None:
    ids = [c.id for c in load_sample_chunks()]
    assert len(ids) == len(set(ids))


def test_every_chunk_belongs_to_a_document_with_same_labels() -> None:
    docs = {d.url: d for d in load_sample_documents()}
    for chunk in load_sample_chunks():
        doc = docs[chunk.document_url]
        assert (chunk.product, chunk.versions, chunk.doc_type) == (
            doc.product,
            doc.versions,
            doc.doc_type,
        )


def test_every_document_has_chunks() -> None:
    chunk_urls = {c.document_url for c in load_sample_chunks()}
    assert {d.url for d in load_sample_documents()} == chunk_urls


def test_hard_cases_are_covered() -> None:
    docs = load_sample_documents()
    # same title in two products
    saml = {d.product for d in docs if d.title == "Configure SAML Authentication"}
    assert saml == {"sequence", "orchestration"}
    # release notes with an exact release
    assert any(d.doc_type == "release-notes" and d.release for d in docs)
    # a page that applies to several versions, and one to none
    assert any(len(d.versions) > 1 for d in docs)
    assert any(d.product == "general" and d.versions == [] for d in docs)
    # an exact technical name, for text search
    assert any("Set-CoraSeQuenceSiteAuthentication" in c.text for c in load_sample_chunks())
