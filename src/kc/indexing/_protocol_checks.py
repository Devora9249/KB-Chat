"""Static (mypy-only) proof that the fakes satisfy the contracts.

Nothing here runs: the assignments are inside `if TYPE_CHECKING`. mypy checks each
assignment, so if an implementation drifts from its Protocol (missing method, wrong
signature, wrong return type), `uv run mypy` fails.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.storage import Embedder, SearchIndex
    from kc.indexing._provisional_contracts import IndexStore, IndexWriter
    from kc.indexing.fake_embedder import FakeEmbedder
    from kc.indexing.in_memory_index_store import InMemoryIndexStore

    _store = InMemoryIndexStore(dimension=1, model_name="m")
    _as_search_index: SearchIndex = _store
    _as_index_writer: IndexWriter = _store
    _as_index_store: IndexStore = _store
    _as_embedder: Embedder = FakeEmbedder()
