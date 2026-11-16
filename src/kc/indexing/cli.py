"""Command-line indexing tool (integration day): JSONL files -> Embedder -> index.

    uv run python -m kc.indexing.cli index --docs <path> --chunks <path> \\
        --embedder fake|bge-m3|e5 --store memory|pg [--skip-unchanged] [--batch-size 16]

Defaults are `--embedder fake --store memory`: no model download, no database. With
`--store memory` the index lives only in this process, so the run checks the files and the
embedding, and the report shows what WOULD be indexed.

Steps:
1. Read both files and group every chunk with its document (kc.indexing.jsonl_source).
2. Create the embedder and the store. `pg` reads DATABASE_URL from the settings (.env).
3. If the store supports it (PgIndexStore), verify the embedding model first: a different
   model or dimension stops the run before anything is written.
4. Optional --skip-unchanged: skip documents whose content_hash equals the indexed one.
   NOTE: change detection belongs to Ingestion (Devora). content_hash covers content_md
   only, so a change in labels or chunking alone is not detected; removed pages are not
   handled here.
5. index_documents, then print a report: indexed, failed (with reasons), skipped, times.

Exit codes: 0 = everything indexed; 1 = some documents failed; 2 = configuration or input
error (nothing was indexed).

    uv run python -m kc.indexing.cli reindex --docs <path> --chunks <path> \\
        --embedder ... --store ... [--yes-delete-everything]

`reindex` changes the embedding model: delete EVERYTHING, record the new model in
embedding_config, and index again from the files. Destructive, so:
- every check that can fail runs BEFORE the deletion: the files, the model's dimension
  against the vector column (a different dimension needs a migration first), and a
  warm-up `embed` call (a model that fails to download or load fails here);
- it prints how many documents would be deleted and the old -> new model;
- without --yes-delete-everything it stops there (dry run, exit code 0, nothing changed);
- the deletion and the new embedding_config happen in ONE transaction (`reset_index`).
Only stores with `reset_index` (PgIndexStore, after PR #14 + a follow-up) support it; with
`--store memory` there is nothing to delete, so it warns and simply indexes.
Search sees a partial index while re-indexing runs: run it when the system is not in use.

Implementations are looked up by NAME and imported only when chosen, so this tool already
works with fake/memory while bge-m3, e5 (PR #19) and pg (PR #14) are not on main yet.
TODO(after #14 and #19): replace the string registry with direct imports (type-checked).
"""

import argparse
import asyncio
import importlib
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from contracts.storage import Embedder, EmbeddingError, IndexStoreError
from kc.db.embedding_config import EmbeddingConfigMismatchError
from kc.indexing._provisional_contracts import Chunk, Document, IndexStore
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.jsonl_source import InputFileError, LoadedItems, read_jsonl_items
from kc.indexing.pipeline import index_documents

EXIT_OK = 0
EXIT_SOME_FAILED = 1
EXIT_CONFIG_ERROR = 2

# chunks.embedding is vector(1024) (migration 0004, PR #14). Another dimension needs a new
# migration before any re-index.
PG_EMBEDDING_DIMENSION = 1024
# The fake embedder uses the same dimension so that `--embedder fake --store pg` also
# works for a smoke test against a real database.
FAKE_DIMENSION = PG_EMBEDDING_DIMENSION
WARM_UP_TEXT = "warm-up"


class UnavailableError(RuntimeError):
    """The chosen implementation is not available (not merged yet, or missing dependency)."""


@dataclass(frozen=True)
class Implementation:
    target: str  # "module.path:ClassName"
    available_from: str  # shown when the module is not on this branch yet


EMBEDDERS: dict[str, Implementation] = {
    "fake": Implementation("kc.indexing.fake_embedder:FakeEmbedder", "main"),
    "bge-m3": Implementation("kc.indexing.bge_m3_embedder:BgeM3Embedder", "PR #19"),
    "e5": Implementation("kc.indexing.e5_embedder:E5Embedder", "PR #19"),
}
STORES: dict[str, Implementation] = {
    "memory": Implementation("kc.indexing.in_memory_index_store:InMemoryIndexStore", "main"),
    "pg": Implementation("kc.db.pg_index_store:PgIndexStore", "PR #14"),
}


@runtime_checkable
class VerifiesEmbeddingConfig(Protocol):
    """Stores that guard against mixing embedding models (PgIndexStore)."""

    async def verify_embedding_config(self, model_name: str, dimension: int) -> None: ...


@runtime_checkable
class ResetsIndex(Protocol):
    """Stores whose whole index can be replaced (PgIndexStore, follow-up after PR #14).

    Deliberately NOT part of the shared contract: changing the model is a maintenance
    operation of Storage & Indexing, not something searching or evaluating code needs.
    """

    @property
    def dimension(self) -> int: ...

    async def current_embedding_config(self) -> tuple[str, int] | None:
        """(model_name, dimension) the index was built with, or None if it is empty."""
        ...

    async def reset_index(self, model_name: str, dimension: int) -> None:
        """In ONE transaction: delete all chunks and documents, record the new model."""
        ...


@dataclass
class IndexOptions:
    docs: Path
    chunks: Path
    embedder: str = "fake"
    store: str = "memory"
    skip_unchanged: bool = False
    batch_size: int = 16


@dataclass
class ReindexOptions:
    docs: Path
    chunks: Path
    embedder: str = "fake"
    store: str = "memory"
    batch_size: int = 16
    yes_delete_everything: bool = False


# --- implementations (imported lazily) ---------------------------------------------------


def load_class(name: str, implementation: Implementation) -> Any:
    module_name, _, attribute = implementation.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name is not None and module_name.startswith(e.name):
            raise UnavailableError(
                f"{name!r} is not available on this branch yet ({implementation.available_from})"
            ) from e
        raise UnavailableError(f"{name!r} needs the missing package {e.name!r}") from e
    return getattr(module, attribute)


def create_embedder(name: str, batch_size: int) -> Embedder:
    cls = load_class(name, EMBEDDERS[name])
    embedder: Embedder
    if name == "fake":
        embedder = cls(dimension=FAKE_DIMENSION)
    else:
        embedder = cls(batch_size=batch_size)
    return embedder


@asynccontextmanager
async def open_store(name: str, embedder: Embedder) -> AsyncIterator[IndexStore]:
    """The store for `name`; for pg also the engine, which is disposed at the end."""
    if name == "memory":
        yield InMemoryIndexStore(dimension=embedder.dimension, model_name=embedder.model_name)
        return
    store_cls = load_class(name, STORES[name])
    from kc.db.engine import create_engine, create_session_factory
    from kc.db.settings import load_db_settings

    engine = create_engine(load_db_settings())
    try:
        yield store_cls(create_session_factory(engine), dimension=embedder.dimension)
    finally:
        await engine.dispose()


# --- the index command -------------------------------------------------------------------


async def run_index(
    options: IndexOptions, *, embedder: Embedder, store: IndexStore, out: TextIO
) -> int:
    """Index the files into `store` and print the report. Components are passed in (DI)."""
    started = time.perf_counter()
    loaded = _read_files(options.docs, options.chunks, out)
    if loaded is None:
        return EXIT_CONFIG_ERROR
    _print_components(out, embedder, options.store)

    if isinstance(store, VerifiesEmbeddingConfig):
        try:
            await store.verify_embedding_config(embedder.model_name, embedder.dimension)
        except (EmbeddingConfigMismatchError, ValueError) as e:
            print(f"error (embedding model mismatch): {e}", file=out)
            print("nothing was indexed", file=out)
            return EXIT_CONFIG_ERROR
        except IndexStoreError as e:
            print(f"error (database): {e}; cause: {e.__cause__!r}", file=out)
            print("nothing was indexed", file=out)
            return EXIT_CONFIG_ERROR

    items = loaded.items
    skipped: list[str] = []
    if options.skip_unchanged:
        if options.store == "memory":
            print(
                "warning: --skip-unchanged has no effect with --store memory "
                "(the index starts empty)",
                file=out,
            )
        indexed_hashes = await store.get_document_hashes()
        kept: list[tuple[Document, list[Chunk]]] = []
        for document, chunks in items:
            if indexed_hashes.get(document.url) == document.content_hash:
                skipped.append(document.url)
            else:
                kept.append((document, chunks))
        items = kept

    return await _index_and_report(
        items, loaded, embedder=embedder, store=store, out=out, started=started, skipped=skipped
    )


# --- the reindex command -----------------------------------------------------------------


async def run_reindex(
    options: ReindexOptions, *, embedder: Embedder, store: IndexStore, out: TextIO
) -> int:
    """Replace the whole index with the files, embedded with `embedder` (see module doc)."""
    started = time.perf_counter()
    # 1. Everything that can fail runs BEFORE anything is deleted.
    loaded = _read_files(options.docs, options.chunks, out)
    if loaded is None:
        return EXIT_CONFIG_ERROR
    _print_components(out, embedder, options.store)

    resettable: ResetsIndex | None = store if isinstance(store, ResetsIndex) else None
    if resettable is not None and embedder.dimension != resettable.dimension:
        print(_needs_migration(embedder, resettable.dimension), file=out)
        return EXIT_CONFIG_ERROR

    try:  # a model that cannot be downloaded or loaded fails here, not after the deletion
        await embedder.embed([WARM_UP_TEXT], input_type="query")
    except EmbeddingError as e:
        print(f"error: the embedder failed to start: {e}", file=out)
        print("nothing was deleted", file=out)
        return EXIT_CONFIG_ERROR

    if resettable is None:
        print(
            f"warning: store {options.store!r} keeps nothing between runs, so there is "
            "nothing to delete; indexing only",
            file=out,
        )
        return await _index_and_report(
            loaded.items, loaded, embedder=embedder, store=store, out=out, started=started
        )

    # 2. What would be deleted.
    try:
        document_count = len(await store.get_document_hashes())
        current = await resettable.current_embedding_config()
    except IndexStoreError as e:
        print(f"error (database): {e}; cause: {e.__cause__!r}", file=out)
        print("nothing was deleted", file=out)
        return EXIT_CONFIG_ERROR
    old_model = f"{current[0]} (dim {current[1]})" if current else "none (empty index)"
    print(f"will delete {document_count} documents (and all their chunks)", file=out)
    new_model = f"{embedder.model_name} (dim {embedder.dimension})"
    print(f"index model: {old_model} -> {new_model}", file=out)

    if not options.yes_delete_everything:
        print("dry run: nothing was changed; add --yes-delete-everything to run", file=out)
        return EXIT_OK

    # 3. Delete everything and record the new model, in one transaction.
    try:
        await resettable.reset_index(embedder.model_name, embedder.dimension)
    except IndexStoreError as e:
        print(f"error (database): {e}; cause: {e.__cause__!r}", file=out)
        print("nothing was deleted (the transaction was rolled back)", file=out)
        return EXIT_CONFIG_ERROR
    print(f"deleted {document_count} documents; index model is now {new_model}", file=out)

    # 4. Index again.
    return await _index_and_report(
        loaded.items, loaded, embedder=embedder, store=store, out=out, started=started
    )


def _needs_migration(embedder: Embedder, column_dimension: int) -> str:
    return (
        f"error: {embedder.model_name} has dimension {embedder.dimension}, but the vector "
        f"column is vector({column_dimension}). A new migration is needed first (column "
        "type and HNSW index; Storage & Indexing creates it). Nothing was deleted."
    )


# --- shared steps --------------------------------------------------------------------------


def _read_files(docs: Path, chunks: Path, out: TextIO) -> LoadedItems | None:
    started = time.perf_counter()
    try:
        loaded = read_jsonl_items(docs, chunks)
    except InputFileError as e:
        print(f"error: {e}", file=out)
        return None
    print(
        f"read {len(loaded.items) + len(loaded.documents_without_chunks)} documents and "
        f"{loaded.chunk_count} chunks in {time.perf_counter() - started:.2f} s",
        file=out,
    )
    return loaded


def _print_components(out: TextIO, embedder: Embedder, store_name: str) -> None:
    print(
        f"embedder: {embedder.model_name} (dim {embedder.dimension}); store: {store_name}",
        file=out,
    )


async def _index_and_report(
    items: list[tuple[Document, list[Chunk]]],
    loaded: LoadedItems,
    *,
    embedder: Embedder,
    store: IndexStore,
    out: TextIO,
    started: float,
    skipped: list[str] | None = None,
) -> int:
    index_started = time.perf_counter()
    report = await index_documents(items, embedder=embedder, store=store)
    index_seconds = time.perf_counter() - index_started

    failed = dict(report.failed)
    for url in loaded.documents_without_chunks:
        failed[url] = "no chunks in the chunks file"

    _print_report(out, report.indexed, failed, skipped or [], index_seconds)
    print(f"total time {time.perf_counter() - started:.2f} s", file=out)
    return EXIT_SOME_FAILED if failed else EXIT_OK


def _print_report(
    out: TextIO, indexed: list[str], failed: dict[str, str], skipped: list[str], seconds: float
) -> None:
    per_document = f" ({seconds / len(indexed) * 1000:,.0f} ms per document)" if indexed else ""
    print(f"indexed: {len(indexed)} documents in {seconds:.2f} s{per_document}", file=out)
    if skipped:
        print(f"skipped (unchanged content_hash): {len(skipped)}", file=out)
    print(f"failed: {len(failed)}", file=out)
    for url, reason in failed.items():
        print(f"  - {url}: {reason}", file=out)


async def _index_command(options: IndexOptions, out: TextIO) -> int:
    async with AsyncExitStack() as stack:
        try:
            embedder = create_embedder(options.embedder, options.batch_size)
            store = await stack.enter_async_context(open_store(options.store, embedder))
        except (UnavailableError, ValueError) as e:  # ValueError: bad DATABASE_URL, dimension
            print(f"error: {e}", file=out)
            return EXIT_CONFIG_ERROR
        return await run_index(options, embedder=embedder, store=store, out=out)


async def _reindex_command(options: ReindexOptions, out: TextIO) -> int:
    async with AsyncExitStack() as stack:
        try:
            embedder = create_embedder(options.embedder, options.batch_size)
            # Before creating the store: PgIndexStore itself refuses another dimension, but
            # with a less helpful message.
            if options.store == "pg" and embedder.dimension != PG_EMBEDDING_DIMENSION:
                print(_needs_migration(embedder, PG_EMBEDDING_DIMENSION), file=out)
                return EXIT_CONFIG_ERROR
            store = await stack.enter_async_context(open_store(options.store, embedder))
        except (UnavailableError, ValueError) as e:
            print(f"error: {e}", file=out)
            return EXIT_CONFIG_ERROR
        return await run_reindex(options, embedder=embedder, store=store, out=out)


# --- command line -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kc.indexing.cli", description="Storage & Indexing command-line tool."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="index documents and chunks from JSONL files")
    _add_common_arguments(index)
    index.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="skip documents whose content_hash equals the one already indexed",
    )
    reindex = commands.add_parser(
        "reindex",
        help="DELETE the whole index, switch to the given embedder's model, index the files",
    )
    _add_common_arguments(reindex)
    reindex.add_argument(
        "--yes-delete-everything",
        action="store_true",
        help="really delete; without it, only print what would happen (dry run)",
    )
    return parser


def _add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--docs", type=Path, required=True, help="documents JSONL file")
    command.add_argument("--chunks", type=Path, required=True, help="chunks JSONL file")
    command.add_argument("--embedder", choices=sorted(EMBEDDERS), default="fake")
    command.add_argument("--store", choices=sorted(STORES), default="memory")
    command.add_argument("--batch-size", type=_positive_int, default=16)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    stream = out if out is not None else sys.stdout
    common = {
        "docs": args.docs,
        "chunks": args.chunks,
        "embedder": args.embedder,
        "store": args.store,
        "batch_size": args.batch_size,
    }
    command: Coroutine[Any, Any, int]
    if args.command == "reindex":
        options = ReindexOptions(**common, yes_delete_everything=args.yes_delete_everything)
        command = _reindex_command(options, stream)
    else:
        command = _index_command(IndexOptions(**common, skip_unchanged=args.skip_unchanged), stream)
    # psycopg's async mode does not support Windows' default Proactor event loop.
    loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = (
        asyncio.SelectorEventLoop if sys.platform == "win32" else None
    )
    return asyncio.run(command, loop_factory=loop_factory)


if __name__ == "__main__":
    sys.exit(main())
