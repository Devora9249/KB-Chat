"""IndexStore contract tests (draft storage contract D6-D9, D10 placeholder).

Runs against every implementation registered in conftest.IMPLEMENTATIONS.
Only behaviour promised by the contract belongs here; implementation details
(constructors, exact text-search score values) are tested next to each implementation.
"""

import pytest
from index_store_helpers import StoreFactory, make_chunk, make_doc, run, snapshot

from kc.indexing._provisional_contracts import Chunk, Filters, IndexStore
from kc.indexing.fake_embedder import FakeEmbedder

# --- empty store ------------------------------------------------------------------------


def test_empty_store_returns_nothing(store: IndexStore) -> None:
    assert run(store.get_document_hashes()) == {}
    assert run(store.vector_search([1.0, 0.0, 0.0])) == []
    assert run(store.text_search("anything")) == []


# --- D7: get_document_hashes --------------------------------------------------------------


def test_get_document_hashes_returns_url_to_hash(store: IndexStore) -> None:
    run(store.upsert_document(make_doc("u1", content_hash="a"), [], []))
    run(store.upsert_document(make_doc("u2", content_hash="b"), [], []))
    assert run(store.get_document_hashes()) == {"u1": "a", "u2": "b"}


# --- vector_search: results, ordering, k ------------------------------------------------


def test_vector_search_orders_by_score_and_fills_result_fields(store: IndexStore) -> None:
    doc = make_doc("u1", title="SAML setup")
    chunks = [make_chunk("c-far", "u1", "far"), make_chunk("c-near", "u1", "near")]
    run(store.upsert_document(doc, chunks, [[0.0, 1.0, 0.0], [1.0, 0.1, 0.0]]))

    results = run(store.vector_search([1.0, 0.0, 0.0]))

    assert [r.chunk_id for r in results] == ["c-near", "c-far"]
    top = results[0]
    assert (top.url, top.title, top.text) == ("u1", "SAML setup", "near")
    assert top.score > results[1].score


def test_ties_are_broken_by_chunk_id(store: IndexStore) -> None:
    same = [1.0, 0.0, 0.0]
    chunks = [make_chunk(cid, "u1") for cid in ["c3", "c1", "c2"]]
    run(store.upsert_document(make_doc("u1"), chunks, [same, same, same]))
    assert [r.chunk_id for r in run(store.vector_search(same))] == ["c1", "c2", "c3"]


def test_k_limits_number_of_results(store: IndexStore) -> None:
    chunks = [make_chunk(f"c{i}", "u1") for i in range(5)]
    run(store.upsert_document(make_doc("u1"), chunks, [[1.0, 0.0, float(i)] for i in range(5)]))
    assert len(run(store.vector_search([1.0, 0.0, 0.0], k=2))) == 2
    assert len(run(store.vector_search([1.0, 0.0, 0.0], k=100))) == 5


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises_in_both_searches(store: IndexStore, k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        run(store.vector_search([1.0, 0.0, 0.0], k=k))
    with pytest.raises(ValueError, match="k must be positive"):
        run(store.text_search("query", k=k))


def test_vector_search_rejects_wrong_query_dimension(store: IndexStore) -> None:
    with pytest.raises(ValueError, match="dimension"):
        run(store.vector_search([1.0, 0.0]))


# --- D10: Filters (agreed with Racheli, 2026-11-12) ----------------------------------------
# Each rule is checked in BOTH searches. Every chunk text contains "saml" and every vector
# equals the query vector, so without filters every chunk is returned by either search.

SearchKind = str  # "vector" | "text"
SAME = [1.0, 0.0, 0.0]


def _search_ids(
    store: IndexStore, kind: SearchKind, filters: Filters | None, k: int = 100
) -> list[str]:
    if kind == "vector":
        results = run(store.vector_search(SAME, filters, k=k))
    else:
        results = run(store.text_search("saml", filters, k=k))
    return sorted(r.chunk_id for r in results)


def _index_variety(store: IndexStore) -> None:
    """seq-howto-9, seq-notes-9, orch-howto-10, cm-howto-none (version=None)."""
    chunks = [
        make_chunk("seq-howto-9", "u1", "saml", product="sequence", version="9"),
        make_chunk(
            "seq-notes-9", "u1", "saml", product="sequence", version="9", doc_type="release-notes"
        ),
        make_chunk("orch-howto-10", "u1", "saml", product="orchestration", version="10"),
        make_chunk("cm-howto-none", "u1", "saml", product="case-manager", version=None),
    ]
    run(store.upsert_document(make_doc("u1"), chunks, [SAME] * len(chunks)))


ALL_IDS = ["cm-howto-none", "orch-howto-10", "seq-howto-9", "seq-notes-9"]
search_kinds = pytest.mark.parametrize("kind", ["vector", "text"])


@search_kinds
def test_no_filters_and_all_none_filters_return_everything(
    store: IndexStore, kind: SearchKind
) -> None:
    _index_variety(store)
    assert _search_ids(store, kind, None) == ALL_IDS
    assert _search_ids(store, kind, Filters()) == ALL_IDS


@search_kinds
def test_values_in_one_field_are_or(store: IndexStore, kind: SearchKind) -> None:
    _index_variety(store)
    filters = Filters(products=["sequence", "orchestration"])
    assert _search_ids(store, kind, filters) == ["orch-howto-10", "seq-howto-9", "seq-notes-9"]


@search_kinds
def test_different_fields_are_and(store: IndexStore, kind: SearchKind) -> None:
    _index_variety(store)
    filters = Filters(products=["sequence"], doc_types=["how-to"])
    assert _search_ids(store, kind, filters) == ["seq-howto-9"]


@search_kinds
def test_exclude_doc_types_removes_those_types(store: IndexStore, kind: SearchKind) -> None:
    _index_variety(store)
    filters = Filters(exclude_doc_types=["release-notes"])
    assert _search_ids(store, kind, filters) == ["cm-howto-none", "orch-howto-10", "seq-howto-9"]


@search_kinds
def test_versions_filter_never_returns_chunks_without_version(
    store: IndexStore, kind: SearchKind
) -> None:
    _index_variety(store)
    # Every versioned chunk is requested; the version=None chunk must still be left out.
    filters = Filters(versions=["9", "10", "11"])
    ids = _search_ids(store, kind, filters)
    assert "cm-howto-none" not in ids
    assert ids == ["orch-howto-10", "seq-howto-9", "seq-notes-9"]


def test_filters_apply_before_k_in_vector_search(store: IndexStore) -> None:
    # 5 non-matching chunks score HIGHER than all matching ones. A store that took the
    # top k first and filtered afterwards would return nothing.
    blockers = [make_chunk(f"x-{i}", "u1", product="orchestration", version="10") for i in range(5)]
    wanted = [make_chunk(f"s-{i}", "u1", product="sequence") for i in range(8)]
    blocker_vectors = [[1.0, 0.0, 0.0]] * 5  # cosine 1.0 with the query
    wanted_vectors = [[1.0, 0.5 + i * 0.5, 0.0] for i in range(8)]  # s-0 best ... s-7 worst
    run(store.upsert_document(make_doc("u1"), blockers + wanted, blocker_vectors + wanted_vectors))

    results = run(store.vector_search([1.0, 0.0, 0.0], Filters(products=["sequence"]), k=5))

    assert [r.chunk_id for r in results] == ["s-0", "s-1", "s-2", "s-3", "s-4"]


def test_filters_apply_before_k_in_text_search(store: IndexStore) -> None:
    v = [1.0, 0.0, 0.0]
    blockers = [
        make_chunk(f"x-{i}", "u1", "saml saml saml saml", product="orchestration") for i in range(5)
    ]
    wanted = [
        make_chunk(f"s-{i}", "u1", " ".join(["saml"] * (3 - min(i, 2))), product="sequence")
        for i in range(6)
    ]  # s-0: 3 matches, s-1: 2, s-2..s-5: 1
    run(store.upsert_document(make_doc("u1"), blockers + wanted, [v] * 11))

    results = run(store.text_search("saml", Filters(products=["sequence"]), k=5))

    assert [r.chunk_id for r in results] == ["s-0", "s-1", "s-2", "s-3", "s-4"]


# --- D6: replace all chunks of a document --------------------------------------------------


def test_update_replaces_all_old_chunks_and_hash(store: IndexStore) -> None:
    v = [1.0, 0.0, 0.0]
    old_chunks = [make_chunk(f"old{i}", "u1") for i in range(3)]
    run(store.upsert_document(make_doc("u1", content_hash="v1"), old_chunks, [v, v, v]))

    run(store.upsert_document(make_doc("u1", content_hash="v2"), [make_chunk("new", "u1")], [v]))

    assert [r.chunk_id for r in run(store.vector_search(v))] == ["new"]
    assert run(store.get_document_hashes()) == {"u1": "v2"}


def test_update_of_one_document_does_not_touch_others(store: IndexStore) -> None:
    v = [1.0, 0.0, 0.0]
    run(store.upsert_document(make_doc("u1"), [make_chunk("a", "u1")], [v]))
    run(store.upsert_document(make_doc("u2"), [make_chunk("b", "u2")], [v]))

    run(store.upsert_document(make_doc("u1"), [make_chunk("a2", "u1")], [v]))

    assert sorted(r.chunk_id for r in run(store.vector_search(v))) == ["a2", "b"]


# --- D6: validation + atomicity -----------------------------------------------------------


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        # length mismatch: 2 chunks, 1 vector
        ([make_chunk("n1", "u1"), make_chunk("n2", "u1")], [[1.0, 0.0, 0.0]], "chunks but"),
        # wrong dimension in the LAST vector: a non-atomic store would have written n1 already
        (
            [make_chunk("n1", "u1"), make_chunk("n2", "u1")],
            [[1.0, 0.0, 0.0], [1.0, 0.0]],
            "dimension",
        ),
        # chunk belongs to another document
        ([make_chunk("n1", "u1"), make_chunk("n2", "OTHER")], [[1.0, 0, 0], [1.0, 0, 0]], "url"),
    ],
    ids=["length-mismatch", "wrong-dimension", "wrong-document-url"],
)
def test_invalid_upsert_raises_and_changes_nothing(
    store: IndexStore, chunks: list[Chunk], vectors: list[list[float]], message: str
) -> None:
    run(
        store.upsert_document(
            make_doc("u1", content_hash="old"), [make_chunk("old", "u1")], [[1.0, 0.0, 0.0]]
        )
    )
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

    (result,) = run(store.vector_search([1.0, 0.0, 0.0]))
    assert result.score == pytest.approx(1.0)
    assert result.text == "original"


# --- delete_document ----------------------------------------------------------------------


def test_delete_document_removes_chunks_and_hash(store: IndexStore) -> None:
    v = [1.0, 0.0, 0.0]
    run(store.upsert_document(make_doc("u1"), [make_chunk("a", "u1")], [v]))
    run(store.upsert_document(make_doc("u2"), [make_chunk("b", "u2")], [v]))

    run(store.delete_document("u1"))

    assert [r.chunk_id for r in run(store.vector_search(v))] == ["b"]
    assert set(run(store.get_document_hashes())) == {"u2"}


def test_delete_missing_document_is_a_no_op(store: IndexStore) -> None:
    run(store.delete_document("does-not-exist"))
    assert run(store.get_document_hashes()) == {}


# --- text_search (order only; score values are implementation-specific) ---------------------


def test_text_search_ranks_more_matches_first_case_insensitive(store: IndexStore) -> None:
    v = [1.0, 0.0, 0.0]
    chunks = [
        make_chunk("one", "u1", "SAML is configured here"),
        make_chunk("two", "u1", "configure saml: SAML settings"),
        make_chunk("none", "u1", "release notes"),
    ]
    run(store.upsert_document(make_doc("u1"), chunks, [v, v, v]))

    results = run(store.text_search("saml"))

    assert [r.chunk_id for r in results] == ["two", "one"]  # more matches first; no match left out
    assert results[0].score > results[1].score


def test_text_search_with_no_words_or_no_match_returns_empty(store: IndexStore) -> None:
    run(store.upsert_document(make_doc("u1"), [make_chunk("a", "u1", "hello")], [[1.0, 0, 0]]))
    assert run(store.text_search("   !!! ")) == []
    assert run(store.text_search("absent")) == []


# --- end to end with FakeEmbedder ---------------------------------------------------------


def test_end_to_end_with_fake_embedder(make_store: StoreFactory) -> None:
    embedder = FakeEmbedder(dimension=64)
    store = make_store(embedder.dimension)
    texts = {
        "saml": "how to configure SAML authentication in SeQuence",
        "notes": "release notes for case manager version 11",
    }
    chunks = [make_chunk(cid, "u1", text) for cid, text in texts.items()]
    vectors = run(embedder.embed(list(texts.values()), input_type="document"))
    run(store.upsert_document(make_doc("u1"), chunks, vectors))

    (query,) = run(embedder.embed(["configure SAML"], input_type="query"))

    assert run(store.vector_search(query, k=1))[0].chunk_id == "saml"
