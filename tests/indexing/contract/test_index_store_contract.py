"""IndexWriter contract tests (write side): replace-all, atomicity, copies, delete, hashes.

Each test gets `writer`: a fresh, empty IndexWriter plus an observer (`writer.stored()`)
that reads back what was stored WITHOUT using search, so these tests also run on
implementations whose search side does not exist yet. Vectors are built with
`writer.unit()`, i.e. in each implementation's own dimension (3 in memory, 1024 in pg).

Runs against every implementation in conftest.WRITER_IMPLEMENTATIONS ("pg" in CI only).
Search behaviour lives in test_search_index_contract.py.
"""

import pytest
from index_store_helpers import WriterUnderTest, make_chunk, make_doc, run, snapshot

# --- get_document_hashes (D7) -----------------------------------------------------------------


def test_empty_writer_has_no_hashes_and_no_chunks(writer: WriterUnderTest) -> None:
    assert run(writer.writer.get_document_hashes()) == {}
    assert writer.stored() == []


def test_get_document_hashes_returns_url_to_hash(writer: WriterUnderTest) -> None:
    run(writer.writer.upsert_document(make_doc("u1", content_hash="a"), [], []))
    run(writer.writer.upsert_document(make_doc("u2", content_hash="b"), [], []))
    assert run(writer.writer.get_document_hashes()) == {"u1": "a", "u2": "b"}


# --- D6: replace all chunks of a document --------------------------------------------------


def test_first_upsert_stores_every_chunk_with_its_vector(writer: WriterUnderTest) -> None:
    chunks = [make_chunk("c1", "u1", "first"), make_chunk("c2", "u1", "second")]
    vectors = [writer.unit(0), writer.unit(1)]
    run(writer.writer.upsert_document(make_doc("u1"), chunks, vectors))

    assert writer.stored() == [
        ("c1", "u1", "first", tuple(writer.unit(0))),
        ("c2", "u1", "second", tuple(writer.unit(1))),
    ]


def test_update_replaces_all_old_chunks_and_hash(writer: WriterUnderTest) -> None:
    v = writer.unit()
    old_chunks = [make_chunk(f"old{i}", "u1") for i in range(3)]
    run(writer.writer.upsert_document(make_doc("u1", content_hash="v1"), old_chunks, [v, v, v]))

    run(
        writer.writer.upsert_document(
            make_doc("u1", content_hash="v2"), [make_chunk("new", "u1")], [v]
        )
    )

    assert [chunk_id for chunk_id, *_ in writer.stored()] == ["new"]
    assert run(writer.writer.get_document_hashes()) == {"u1": "v2"}


def test_update_of_one_document_does_not_touch_others(writer: WriterUnderTest) -> None:
    v = writer.unit()
    run(writer.writer.upsert_document(make_doc("u1"), [make_chunk("a", "u1")], [v]))
    run(writer.writer.upsert_document(make_doc("u2"), [make_chunk("b", "u2")], [v]))

    run(writer.writer.upsert_document(make_doc("u1"), [make_chunk("a2", "u1")], [v]))

    assert [(chunk_id, url) for chunk_id, url, *_ in writer.stored()] == [
        ("a2", "u1"),
        ("b", "u2"),
    ]


# --- D6: validation + atomicity -----------------------------------------------------------


@pytest.mark.parametrize("case", ["length-mismatch", "wrong-dimension", "wrong-document-url"])
def test_invalid_upsert_raises_and_changes_nothing(writer: WriterUnderTest, case: str) -> None:
    v = writer.unit()
    run(
        writer.writer.upsert_document(
            make_doc("u1", content_hash="old"), [make_chunk("old", "u1")], [v]
        )
    )
    before = snapshot(writer)

    two_chunks = [make_chunk("n1", "u1"), make_chunk("n2", "u1")]
    if case == "length-mismatch":  # 2 chunks, 1 vector
        chunks, vectors, message = two_chunks, [v], "chunks but"
    elif case == "wrong-dimension":  # the LAST vector is wrong: a non-atomic writer would
        chunks, vectors, message = two_chunks, [v, v[:-1]], "dimension"  # store n1 first
    else:  # a chunk of another document
        chunks = [make_chunk("n1", "u1"), make_chunk("n2", "OTHER")]
        vectors, message = [v, v], "url"

    with pytest.raises(ValueError, match=message):
        run(writer.writer.upsert_document(make_doc("u1", content_hash="new"), chunks, vectors))

    assert snapshot(writer) == before


def test_writer_keeps_copies_of_inputs(writer: WriterUnderTest) -> None:
    vector = writer.unit()
    chunk = make_chunk("c1", "u1", "original")
    run(writer.writer.upsert_document(make_doc("u1"), [chunk], [vector]))

    vector[0] = -1.0  # caller mutates its own objects afterwards
    chunk.text = "mutated"

    assert writer.stored() == [("c1", "u1", "original", tuple(writer.unit()))]


# --- delete_document ----------------------------------------------------------------------


def test_delete_document_removes_chunks_and_hash(writer: WriterUnderTest) -> None:
    v = writer.unit()
    run(writer.writer.upsert_document(make_doc("u1"), [make_chunk("a", "u1")], [v]))
    run(writer.writer.upsert_document(make_doc("u2"), [make_chunk("b", "u2")], [v]))

    run(writer.writer.delete_document("u1"))

    assert [chunk_id for chunk_id, *_ in writer.stored()] == ["b"]
    assert set(run(writer.writer.get_document_hashes())) == {"u2"}


def test_delete_missing_document_is_a_no_op(writer: WriterUnderTest) -> None:
    run(writer.writer.delete_document("does-not-exist"))
    assert run(writer.writer.get_document_hashes()) == {}
