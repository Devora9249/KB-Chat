"""Tests for the indexing CLI (kc.indexing.cli), with FakeEmbedder and InMemoryIndexStore."""

import asyncio
import importlib.util
import io
from pathlib import Path
from typing import Any

import pytest

from contracts.storage import EmbeddingError, IndexStoreError, InputType
from kc.db.embedding_config import EmbeddingConfigMismatchError
from kc.indexing.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_SOME_FAILED,
    FAKE_DIMENSION,
    IndexOptions,
    build_parser,
    main,
    run_index,
)
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.jsonl_source import read_jsonl_items

FIXTURES = Path(__file__).parent / "cli_fixtures"
DOCS = FIXTURES / "docs.jsonl"
CHUNKS = FIXTURES / "chunks.jsonl"
SAML = "https://kc.example/help/configure-saml-sequence"
RELEASE_NOTES = "https://kc.example/help/release-notes-orchestration-11"
INSTALL = "https://kc.example/help/install-case-manager"


def _cli(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["index", "--docs", str(DOCS), "--chunks", str(CHUNKS), *args], out=out)
    return code, out.getvalue()


def _run(embedder: Any, index: Any, **options: Any) -> tuple[int, str]:
    """run_index with injected components; `options` are IndexOptions fields."""
    out = io.StringIO()
    opts = IndexOptions(docs=DOCS, chunks=CHUNKS, **options)
    code = asyncio.run(run_index(opts, embedder=embedder, store=index, out=out))
    return code, out.getvalue()


def _new_store(embedder: FakeEmbedder) -> InMemoryIndexStore:
    return InMemoryIndexStore(dimension=embedder.dimension, model_name=embedder.model_name)


# --- command line ---------------------------------------------------------------------------


def test_defaults_are_fake_and_memory() -> None:
    args = build_parser().parse_args(["index", "--docs", "d", "--chunks", "c"])
    assert (args.embedder, args.store, args.skip_unchanged, args.batch_size) == (
        "fake",
        "memory",
        False,
        16,
    )


def test_end_to_end_with_fake_and_memory() -> None:
    code, output = _cli()
    assert code == EXIT_OK
    assert "read 3 documents and 6 chunks" in output
    assert f"(dim {FAKE_DIMENSION}); store: memory" in output
    assert "indexed: 3 documents" in output
    assert "failed: 0" in output


@pytest.mark.parametrize(
    "args",
    [["--embedder", "openai"], ["--store", "sqlite"], ["--batch-size", "0"]],
    ids=["unknown-embedder", "unknown-store", "batch-size-0"],
)
def test_invalid_arguments_exit_with_usage_error(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _cli(*args)
    assert exc_info.value.code == 2  # argparse usage error


def test_bad_input_file_indexes_nothing(tmp_path: Path) -> None:
    docs = tmp_path / "docs.jsonl"
    docs.write_text("{not json\n", encoding="utf-8")
    out = io.StringIO()
    code = main(["index", "--docs", str(docs), "--chunks", str(CHUNKS)], out=out)
    assert code == EXIT_CONFIG_ERROR
    assert "docs.jsonl:1: not a valid Document" in out.getvalue()


def test_document_without_chunks_is_reported_as_failed(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    lines = CHUNKS.read_text(encoding="utf-8").splitlines()
    chunks.write_text("\n".join(line for line in lines if INSTALL not in line), encoding="utf-8")
    out = io.StringIO()
    code = main(["index", "--docs", str(DOCS), "--chunks", str(chunks)], out=out)
    assert code == EXIT_SOME_FAILED
    assert "indexed: 2 documents" in out.getvalue()
    assert f"{INSTALL}: no chunks in the chunks file" in out.getvalue()


@pytest.mark.parametrize(
    ("args", "module", "pr"),
    [
        (["--embedder", "bge-m3"], "kc.indexing.bge_m3_embedder", "PR #19"),
        (["--embedder", "e5"], "kc.indexing.e5_embedder", "PR #19"),
        (["--store", "pg"], "kc.db.pg_index_store", "PR #14"),
    ],
    ids=["bge-m3", "e5", "pg"],
)
def test_implementation_not_on_this_branch_gives_a_clear_error(
    args: list[str], module: str, pr: str
) -> None:
    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"{module} is merged; covered by its own tests")
    code, output = _cli(*args)
    assert code == EXIT_CONFIG_ERROR
    assert f"is not available on this branch yet ({pr})" in output


# --- run_index with injected components ----------------------------------------------------


def test_report_lists_failed_documents_with_reason() -> None:
    class FailsOnRelease(FakeEmbedder):
        async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
            if any("Orchestration 11" in text for text in texts):
                raise EmbeddingError("provider timed out")
            return await super().embed(texts, input_type=input_type)

    embedder = FailsOnRelease(dimension=8)
    store = _new_store(embedder)

    code, output = _run(embedder, store)

    assert code == EXIT_SOME_FAILED
    assert "indexed: 2 documents" in output
    assert f"{RELEASE_NOTES}: EmbeddingError: provider timed out" in output
    assert set(asyncio.run(store.get_document_hashes())) == {SAML, INSTALL}


def test_skip_unchanged_skips_only_documents_with_the_same_hash() -> None:
    embedder = FakeEmbedder(dimension=8)
    store = _new_store(embedder)
    loaded = read_jsonl_items(DOCS, CHUNKS)
    for document, chunks in loaded.items:  # index everything once
        vectors = asyncio.run(embedder.embed([c.text for c in chunks], input_type="document"))
        if document.url == INSTALL:  # ...but INSTALL with an OLD hash: it has changed since
            document = document.model_copy(update={"content_hash": "old"})
        asyncio.run(store.upsert_document(document, chunks, vectors))

    code, output = _run(embedder, store, skip_unchanged=True, store="pg-like")

    assert code == EXIT_OK
    assert "skipped (unchanged content_hash): 2" in output
    assert "indexed: 1 documents" in output
    assert asyncio.run(store.get_document_hashes())[INSTALL] == "3" * 64


def test_skip_unchanged_with_memory_store_warns() -> None:
    embedder = FakeEmbedder(dimension=8)
    code, output = _run(embedder, _new_store(embedder), skip_unchanged=True, store="memory")
    assert code == EXIT_OK
    assert "--skip-unchanged has no effect with --store memory" in output


class _GuardedStore(InMemoryIndexStore):
    """InMemoryIndexStore + the pg-only embedding-model guard, to test the CLI's use of it."""

    def __init__(self, *, error: BaseException | None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.error = error
        self.verified: list[tuple[str, int]] = []

    async def verify_embedding_config(self, model_name: str, dimension: int) -> None:
        self.verified.append((model_name, dimension))
        if self.error is not None:
            raise self.error


def test_embedding_model_is_verified_before_indexing() -> None:
    embedder = FakeEmbedder(dimension=8)
    store = _GuardedStore(error=None, dimension=8, model_name=embedder.model_name)
    code, _ = _run(embedder, store)
    assert code == EXIT_OK
    assert store.verified == [(embedder.model_name, 8)]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (EmbeddingConfigMismatchError("index was built with 'BAAI/bge-m3@5617a9f'"), "mismatch"),
        (ValueError("embedder has dimension 8, but the index column is vector(1024)"), "mismatch"),
        (IndexStoreError("cannot connect"), "error (database)"),
    ],
    ids=["other-model", "other-dimension", "database-down"],
)
def test_failed_verification_stops_before_anything_is_indexed(
    error: BaseException, message: str
) -> None:
    embedder = FakeEmbedder(dimension=8)
    store = _GuardedStore(error=error, dimension=8, model_name=embedder.model_name)

    code, output = _run(embedder, store)

    assert code == EXIT_CONFIG_ERROR
    assert message in output
    assert "nothing was indexed" in output
    assert asyncio.run(store.get_document_hashes()) == {}
