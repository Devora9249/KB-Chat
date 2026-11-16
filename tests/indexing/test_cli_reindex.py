"""Tests for `cli reindex` (replace the whole index with a new embedding model).

PgIndexStore.reset_index does not exist yet (follow-up after PR #14), so the store-side
behaviour is played by ResettableStore: InMemoryIndexStore + reset_index +
current_embedding_config, recording every call.
"""

import asyncio
import importlib.util
import io
from pathlib import Path
from typing import Any

import pytest

from contracts.storage import EmbeddingError, IndexStoreError, InputType
from kc.indexing import cli
from kc.indexing.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    ReindexOptions,
    ResetsIndex,
    build_parser,
    main,
    run_reindex,
)
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.jsonl_source import read_jsonl_items

FIXTURES = Path(__file__).parent / "cli_fixtures"
DOCS = FIXTURES / "docs.jsonl"
CHUNKS = FIXTURES / "chunks.jsonl"
DIM = 8
OLD_MODEL = "old-model@1234567"
OLD_PAGES = ["https://kc.example/help/old-page-1", "https://kc.example/help/old-page-2"]
NEW_PAGES = {
    "https://kc.example/help/configure-saml-sequence",
    "https://kc.example/help/release-notes-orchestration-11",
    "https://kc.example/help/install-case-manager",
}


class ResettableStore(InMemoryIndexStore):
    """What PgIndexStore will offer after the follow-up PR, in memory."""

    def __init__(
        self,
        *,
        dimension: int = DIM,
        config: tuple[str, int] | None = (OLD_MODEL, DIM),
        fail_reset: bool = False,
    ) -> None:
        super().__init__(dimension=dimension, model_name="unused")
        self.config = config
        self.fail_reset = fail_reset
        self.resets: list[tuple[str, int]] = []

    async def current_embedding_config(self) -> tuple[str, int] | None:
        return self.config

    async def reset_index(self, model_name: str, dimension: int) -> None:
        if self.fail_reset:
            raise IndexStoreError("connection lost") from OSError("reset by peer")
        self.resets.append((model_name, dimension))
        for url in await self.get_document_hashes():
            await self.delete_document(url)
        self.config = (model_name, dimension)


def _store_with_old_pages(**kwargs: Any) -> ResettableStore:
    store = ResettableStore(**kwargs)
    (document, chunks), *_ = read_jsonl_items(DOCS, CHUNKS).items
    vector = [1.0] + [0.0] * (store.dimension - 1)
    for i, url in enumerate(OLD_PAGES):
        old_document = document.model_copy(update={"url": url})
        old_chunk = chunks[0].model_copy(update={"id": f"old{i}", "document_url": url})
        asyncio.run(store.upsert_document(old_document, [old_chunk], [vector]))
    return store


def _run(embedder: Any, store: Any, *, yes: bool = False) -> tuple[int, str]:
    out = io.StringIO()
    options = ReindexOptions(docs=DOCS, chunks=CHUNKS, store="pg", yes_delete_everything=yes)
    code = asyncio.run(run_reindex(options, embedder=embedder, store=store, out=out))
    return code, out.getvalue()


def _urls(store: InMemoryIndexStore) -> set[str]:
    return set(asyncio.run(store.get_document_hashes()))


def test_resettable_store_matches_the_protocol() -> None:
    assert isinstance(ResettableStore(), ResetsIndex)
    assert not isinstance(InMemoryIndexStore(dimension=DIM, model_name="m"), ResetsIndex)


# --- dry run and the real run ---------------------------------------------------------------


def test_without_the_flag_it_is_a_dry_run_that_changes_nothing() -> None:
    embedder = FakeEmbedder(dimension=DIM, model_name="new-model@abcdef0")
    store = _store_with_old_pages()

    code, output = _run(embedder, store)

    assert code == EXIT_OK
    assert "will delete 2 documents (and all their chunks)" in output
    assert f"index model: {OLD_MODEL} (dim {DIM}) -> new-model@abcdef0 (dim {DIM})" in output
    assert "dry run: nothing was changed; add --yes-delete-everything" in output
    assert store.resets == []
    assert _urls(store) == set(OLD_PAGES)


def test_with_the_flag_deletes_everything_switches_model_and_reindexes() -> None:
    embedder = FakeEmbedder(dimension=DIM, model_name="new-model@abcdef0")
    store = _store_with_old_pages()

    code, output = _run(embedder, store, yes=True)

    assert code == EXIT_OK
    assert store.resets == [("new-model@abcdef0", DIM)]  # exactly once
    assert store.config == ("new-model@abcdef0", DIM)
    assert _urls(store) == NEW_PAGES  # old pages gone, files indexed
    assert "deleted 2 documents; index model is now new-model@abcdef0" in output
    assert "indexed: 3 documents" in output


def test_empty_index_shows_no_previous_model() -> None:
    embedder = FakeEmbedder(dimension=DIM)
    code, output = _run(embedder, ResettableStore(config=None))
    assert code == EXIT_OK
    assert "will delete 0 documents" in output
    assert "index model: none (empty index) ->" in output


# --- checks that run BEFORE anything is deleted ---------------------------------------------


def test_other_dimension_needs_a_migration_and_deletes_nothing() -> None:
    embedder = FakeEmbedder(dimension=DIM)
    store = _store_with_old_pages(dimension=16)

    code, output = _run(embedder, store, yes=True)

    assert code == EXIT_CONFIG_ERROR
    assert f"has dimension {DIM}, but the vector column is vector(16)" in output
    assert "A new migration is needed first" in output
    assert "Nothing was deleted" in output
    assert store.resets == []
    assert _urls(store) == set(OLD_PAGES)


def test_bad_input_file_deletes_nothing(tmp_path: Path) -> None:
    docs = tmp_path / "docs.jsonl"
    docs.write_text("{broken\n", encoding="utf-8")
    store = _store_with_old_pages()
    options = ReindexOptions(docs=docs, chunks=CHUNKS, store="pg", yes_delete_everything=True)
    out = io.StringIO()

    code = asyncio.run(
        run_reindex(options, embedder=FakeEmbedder(dimension=DIM), store=store, out=out)
    )

    assert code == EXIT_CONFIG_ERROR
    assert "docs.jsonl:1" in out.getvalue()
    assert store.resets == []
    assert _urls(store) == set(OLD_PAGES)


def test_embedder_that_cannot_start_deletes_nothing() -> None:
    class BrokenModel(FakeEmbedder):
        async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
            raise EmbeddingError("model download failed")

    store = _store_with_old_pages()

    code, output = _run(BrokenModel(dimension=DIM), store, yes=True)

    assert code == EXIT_CONFIG_ERROR
    assert "the embedder failed to start: model download failed" in output
    assert "nothing was deleted" in output
    assert store.resets == []
    assert _urls(store) == set(OLD_PAGES)


def test_failed_reset_reports_rollback() -> None:
    store = _store_with_old_pages(fail_reset=True)

    code, output = _run(FakeEmbedder(dimension=DIM), store, yes=True)

    assert code == EXIT_CONFIG_ERROR
    assert "error (database): connection lost" in output
    assert "rolled back" in output
    assert _urls(store) == set(OLD_PAGES)


# --- command line -----------------------------------------------------------------------------


def _cli(*args: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(["reindex", "--docs", str(DOCS), "--chunks", str(CHUNKS), *args], out=out)
    return code, out.getvalue()


def test_flag_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(["reindex", "--docs", "d", "--chunks", "c"])
    assert args.yes_delete_everything is False
    assert (args.embedder, args.store) == ("fake", "memory")


def test_memory_store_warns_that_there_is_nothing_to_delete_and_indexes() -> None:
    code, output = _cli()  # no flag needed: nothing is deleted
    assert code == EXIT_OK
    assert "store 'memory' keeps nothing between runs, so there is nothing to delete" in output
    assert "indexed: 3 documents" in output


def test_pg_with_another_dimension_explains_the_migration_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "FAKE_DIMENSION", 768)  # an embedder that does not fit the column
    code, output = _cli("--store", "pg", "--yes-delete-everything")
    assert code == EXIT_CONFIG_ERROR
    assert "has dimension 768, but the vector column is vector(1024)" in output


def test_pg_is_not_available_before_pr_14() -> None:
    if importlib.util.find_spec("kc.db.pg_index_store") is not None:
        pytest.skip("PgIndexStore is merged; covered by its own tests")
    code, output = _cli("--store", "pg")
    assert code == EXIT_CONFIG_ERROR
    assert "'pg' is not available on this branch yet (PR #14)" in output
