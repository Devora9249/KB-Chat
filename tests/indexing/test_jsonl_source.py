"""Tests for kc.indexing.jsonl_source (reading the CLI's input files)."""

from pathlib import Path

import pytest

from kc.indexing.jsonl_source import InputFileError, read_jsonl_items

FIXTURES = Path(__file__).parent / "cli_fixtures"
DOCS = FIXTURES / "docs.jsonl"
CHUNKS = FIXTURES / "chunks.jsonl"
SAML = "https://kc.example/help/configure-saml-sequence"


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_groups_every_chunk_with_its_document_in_file_order() -> None:
    loaded = read_jsonl_items(DOCS, CHUNKS)

    assert [document.url for document, _ in loaded.items] == [
        SAML,
        "https://kc.example/help/release-notes-orchestration-11",
        "https://kc.example/help/install-case-manager",
    ]
    by_url = {document.url: [chunk.id for chunk in chunks] for document, chunks in loaded.items}
    # chunks of one document stay in the order of the chunks file, even when interleaved
    assert by_url[SAML] == ["a000000000000001", "a000000000000003"]
    assert loaded.chunk_count == 6
    assert loaded.documents_without_chunks == []


def test_document_without_chunks_is_reported_not_returned(tmp_path: Path) -> None:
    chunks = _write(tmp_path, "chunks.jsonl", [line for line in _lines(CHUNKS) if SAML in line])
    loaded = read_jsonl_items(DOCS, chunks)
    assert [document.url for document, _ in loaded.items] == [SAML]
    assert len(loaded.documents_without_chunks) == 2


def test_chunk_without_document_is_an_error_with_file_and_line(tmp_path: Path) -> None:
    docs = _write(tmp_path, "docs.jsonl", _lines(DOCS)[:1])  # only the SAML document
    with pytest.raises(InputFileError, match=r"chunks\.jsonl:2: chunk a000000000000002"):
        read_jsonl_items(docs, CHUNKS)


def test_invalid_json_line_names_file_and_line(tmp_path: Path) -> None:
    docs = _write(tmp_path, "docs.jsonl", [_lines(DOCS)[0], "{not json"])
    with pytest.raises(InputFileError, match=r"docs\.jsonl:2: not a valid Document"):
        read_jsonl_items(docs, CHUNKS)


def test_line_that_breaks_the_model_names_the_field(tmp_path: Path) -> None:
    bad = _lines(DOCS)[0].replace('"product": "sequence"', '"product": "SeQuence"')
    docs = _write(tmp_path, "docs.jsonl", [bad])
    with pytest.raises(InputFileError, match=r"docs\.jsonl:1: not a valid Document \(product"):
        read_jsonl_items(docs, CHUNKS)


def test_duplicate_document_url_is_an_error(tmp_path: Path) -> None:
    docs = _write(tmp_path, "docs.jsonl", [_lines(DOCS)[0], _lines(DOCS)[0]])
    with pytest.raises(InputFileError, match=r"docs\.jsonl:2: duplicate document url"):
        read_jsonl_items(docs, CHUNKS)


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InputFileError, match="cannot read file"):
        read_jsonl_items(tmp_path / "nope.jsonl", CHUNKS)
