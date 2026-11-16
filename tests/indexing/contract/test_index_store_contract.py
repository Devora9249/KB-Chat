"""IndexStore write-side contract tests (provisional IndexWriter / IndexStore).

Search behaviour lives in test_search_index_contract.py. These tests cover writing:
replace-all, atomicity, copies, delete and get_document_hashes. They use searches only to
observe the effect of a write. Runs against every implementation in conftest.IMPLEMENTATIONS.
"""

import pytest
from index_store_helpers import make_chunk, make_doc, run, snapshot

from kc.indexing._provisional_contracts import Chunk, IndexStore

V = [1.0, 0.0, 0.0]

# --- get_document_hashes (D7) -----------------------------------------------------------------


def test_empty_store_has_no_hashes(store: IndexStore) -> None:
    assert run(store.get_document_hashes()) == {}


def test_get_document_hashes_returns_url_to_hash(store: IndexStore) -> None:
    run(store.upsert_document(make_doc("u1", content_hash="a"), [], []))
    run(store.upsert_document(make_doc("u2", content_hash="b"), [], []))
    assert run(store.get_document_hashes()) == {"u1": "a", "u2": "b"}


# --- D6: replace all chunks of a document --------------------------------------------------


def test_update_replaces_all_old_chunks_and_hash(store: IndexStore) -> None:
    old_chunks = [make_chunk(f"old{i}", "u1") for i in range(3)]
    run(store.upsert_document(make_doc("u1", content_hash="v1"), old_chunks, [V, V, V]))

    run(store.upsert_document(make_doc("u1", content_hash="v2"), [make_chunk("new", "u1")], [V]))

    assert [r.chunk_id for r in run(store.vector_search(V))] == ["new"]
    assert run(store.get_document_hashes()) == {"u1": "v2"}


def test_update_of_one_document_does_not_touch_others(store: IndexStore) -> None:
    run(store.upsert_document(make_doc("u1"), [make_chunk("a", "u1")], [V]))
    run(store.upsert_document(make_doc("u2"), [make_chunk("b", "u2")], [V]))

    run(store.upsert_document(make_doc("u1"), [make_chunk("a2", "u1")], [V]))

    assert sorted(r.chunk_id for r in run(store.vector_search(V))) == ["a2", "b"]


# --- D6: validation + atomicity -----------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        # length mismatch: 2 chunks, 1 vector
        ([make_chunk("n1", "u1"), make_chunk("n2", "u1")], [V], "chunks but"),
        # wrong dimension in the LAST vector: a non-atomic store would have written n1 already
        ([make_chunk("n1", "u1"), make_chunk("n2", "u1")], [V, [1.0, 0.0]], "dimension"),
        # chunk belongs to another document
        ([make_chunk("n1", "u1"), make_chunk("n2", "OTHER")], [V, V], "url"),
    ],
    ids=["length-mismatch", "wrong-dimension", "wrong-document-url"],
)
def test_invalid_upsert_raises_and_changes_nothing(
    store: IndexStore, chunks: list[Chunk], vectors: list[list[float]], message: str
) -> None:
    run(store.upsert_document(make_doc("u1", content_hash="old"), [make_chunk("old", "u1")], [V]))
    before = snapshot(store)

    with pytest.raises(ValueError, match=message):
        run(store.upsert_document(make_doc("u1", content_hash="new"), chunks, vectors))

    assert snapshot(store) == before


def test_store_keeps_copies_of_inputs(store: IndexStore) -> None:
    vector = [1.0, 0.0, 0.0]
    chunk = make_chunk("c1", "u1", "original")
    run(store.upsert_document(make_doc("u1"), [chunk], [vector]))

    vector[0] = -1.0  # caller mutates its own objects afterwards
    chunk.text = "mutated"

    (result,) = run(store.vector_search(V))
    assert result.score == pytest.approx(1.0)
    assert result.text == "original"


# --- delete_document ----------------------------------------------------------------------


def test_delete_document_removes_chunks_and_hash(store: IndexStore) -> None:
    run(store.upsert_document(make_doc("u1"), [make_chunk("a", "u1")], [V]))
    run(store.upsert_document(make_doc("u2"), [make_chunk("b", "u2")], [V]))

    run(store.delete_document("u1"))

    assert [r.chunk_id for r in run(store.vector_search(V))] == ["b"]
    assert set(run(store.get_document_hashes())) == {"u2"}


def test_delete_missing_document_is_a_no_op(store: IndexStore) -> None:
    run(store.delete_document("does-not-exist"))
    assert run(store.get_document_hashes()) == {}
