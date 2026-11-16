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

Implementations are looked up by NAME and imported only when chosen, so this tool already
works with fake/memory while bge-m3, e5 (PR #19) and pg (PR #14) are not on main yet.
TODO(after #14 and #19): replace the string registry with direct imports (type-checked).
"""

import argparse
import asyncio
import importlib
import sys
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from contracts.storage import Embedder, IndexStoreError
from kc.db.embedding_config import EmbeddingConfigMismatchError
from kc.indexing._provisional_contracts import Chunk, Document, IndexStore
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.jsonl_source import InputFileError, read_jsonl_items
from kc.indexing.pipeline import index_documents

EXIT_OK = 0
EXIT_SOME_FAILED = 1
EXIT_CONFIG_ERROR = 2

# The pg column is vector(1024); the fake embedder uses the same dimension so that
# `--embedder fake --store pg` also works for a smoke test against a real database.
FAKE_DIMENSION = 1024


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


@dataclass
class IndexOptions:
    docs: Path
    chunks: Path
    embedder: str = "fake"
    store: str = "memory"
    skip_unchanged: bool = False
    batch_size: int = 16


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
    try:
        loaded = read_jsonl_items(options.docs, options.chunks)
    except InputFileError as e:
        print(f"error: {e}", file=out)
        return EXIT_CONFIG_ERROR
    read_seconds = time.perf_counter() - started
    print(
        f"read {len(loaded.items) + len(loaded.documents_without_chunks)} documents and "
        f"{loaded.chunk_count} chunks in {read_seconds:.2f} s",
        file=out,
    )
    print(
        f"embedder: {embedder.model_name} (dim {embedder.dimension}); store: {options.store}",
        file=out,
    )

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

    index_started = time.perf_counter()
    report = await index_documents(items, embedder=embedder, store=store)
    index_seconds = time.perf_counter() - index_started

    failed = dict(report.failed)
    for url in loaded.documents_without_chunks:
        failed[url] = "no chunks in the chunks file"

    _print_report(out, report.indexed, failed, skipped, index_seconds)
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


# --- command line -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kc.indexing.cli", description="Storage & Indexing command-line tool."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="index documents and chunks from JSONL files")
    index.add_argument("--docs", type=Path, required=True, help="documents JSONL file")
    index.add_argument("--chunks", type=Path, required=True, help="chunks JSONL file")
    index.add_argument("--embedder", choices=sorted(EMBEDDERS), default="fake")
    index.add_argument("--store", choices=sorted(STORES), default="memory")
    index.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="skip documents whose content_hash equals the one already indexed",
    )
    index.add_argument("--batch-size", type=_positive_int, default=16)
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = IndexOptions(
        docs=args.docs,
        chunks=args.chunks,
        embedder=args.embedder,
        store=args.store,
        skip_unchanged=args.skip_unchanged,
        batch_size=args.batch_size,
    )
    stream = out if out is not None else sys.stdout
    # psycopg's async mode does not support Windows' default Proactor event loop.
    loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = (
        asyncio.SelectorEventLoop if sys.platform == "win32" else None
    )
    return asyncio.run(_index_command(options, stream), loop_factory=loop_factory)


if __name__ == "__main__":
    sys.exit(main())
