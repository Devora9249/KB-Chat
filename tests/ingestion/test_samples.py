"""Tests for SampleIngestionSource and the fixture files behind it.

The tests go through the public interface (`iter_documents`), the same way
teammates use it.
"""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import Chunk, Document, IngestedDocument, IngestionSource
from fixtures.build_samples_v0 import build, write_jsonl
from kc.ingestion.samples import SampleIngestionSource


def collect(source: IngestionSource) -> list[IngestedDocument]:
    async def run() -> list[IngestedDocument]:
        return [item async for item in source.iter_documents()]

    return asyncio.run(run())


@pytest.fixture(scope="module")
def items() -> list[IngestedDocument]:
    return collect(SampleIngestionSource())


def all_docs(items: list[IngestedDocument]) -> list[Document]:
    return [i.document for i in items]


def all_chunks(items: list[IngestedDocument]) -> list[Chunk]:
    return [c for i in items for c in i.chunks]


# --- The fixture files ---------------------------------------------------------


def test_files_are_up_to_date_with_builder(items: list[IngestedDocument]) -> None:
    # If this fails: run `uv run python fixtures/build_samples_v0.py` and commit the files.
    docs, chunks = build()
    assert all_docs(items) == docs
    assert sorted(all_chunks(items), key=lambda c: c.id) == sorted(chunks, key=lambda c: c.id)


def test_chunk_ids_are_unique_across_all_documents(items: list[IngestedDocument]) -> None:
    ids = [c.id for c in all_chunks(items)]
    assert len(ids) == len(set(ids))


def test_hard_cases_are_covered(items: list[IngestedDocument]) -> None:
    docs = all_docs(items)
    # same title in two products
    saml = {d.product for d in docs if d.title == "Configure SAML Authentication"}
    assert saml == {"sequence", "orchestration"}
    # release notes with an exact release
    assert any(d.doc_type == "release-notes" and d.release for d in docs)
    # a page that applies to several versions, and one to none
    assert any(len(d.versions) > 1 for d in docs)
    assert any(d.product == "general" and d.versions == [] for d in docs)
    # an exact technical name, for text search
    assert any("Set-CoraSeQuenceSiteAuthentication" in c.text for c in all_chunks(items))


# --- SampleIngestionSource -----------------------------------------------------


def test_yields_every_document_once(items: list[IngestedDocument]) -> None:
    urls = [d.url for d in all_docs(items)]
    assert len(urls) == 10
    assert len(set(urls)) == 10


def test_custom_dir_with_orphan_chunk_fails(tmp_path: Path) -> None:
    docs, chunks = build()
    write_jsonl(tmp_path / "sample_docs.jsonl", docs[:1])  # drop all docs but one
    write_jsonl(tmp_path / "sample_chunks.jsonl", chunks)
    with pytest.raises(ValueError, match="chunks without a document"):
        collect(SampleIngestionSource(tmp_path))


def test_custom_dir_with_document_without_chunks_fails(tmp_path: Path) -> None:
    docs, _ = build()
    write_jsonl(tmp_path / "sample_docs.jsonl", docs[:1])
    write_jsonl(tmp_path / "sample_chunks.jsonl", [])
    with pytest.raises(ValidationError):
        collect(SampleIngestionSource(tmp_path))
