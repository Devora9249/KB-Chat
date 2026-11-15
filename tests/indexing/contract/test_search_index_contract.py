"""SearchIndex contract tests (contracts.storage.SearchIndex): read side only.

Each test gets a SearchIndex already loaded with its data (fixture `make_search_index`)
and calls ONLY vector_search / text_search. Runs against every implementation registered
in conftest.py. Implementation details (exact text-search score values) are tested next
to each implementation.
"""

import pytest
from index_store_helpers import IndexedDocument, SearchIndexFactory, make_chunk, make_doc, run

from contracts.retrieval import Filters
from contracts.storage import SearchIndex
from kc.indexing.fake_embedder import FakeEmbedder

SAME = [1.0, 0.0, 0.0]


# --- empty index ----------------------------------------------------------------------------


def test_empty_index_returns_nothing(make_search_index: SearchIndexFactory) -> None:
    index = make_search_index([])
    assert run(index.vector_search(SAME)) == []
    assert run(index.text_search("anything")) == []


# --- vector_search: results, ordering, k ----------------------------------------------------


def test_vector_search_orders_by_score_and_fills_result_fields(
    make_search_index: SearchIndexFactory,
) -> None:
    doc = make_doc("u1", title="SAML setup")
    chunks = [make_chunk("c-far", "u1", "far"), make_chunk("c-near", "u1", "near")]
    index = make_search_index([(doc, chunks, [[0.0, 1.0, 0.0], [1.0, 0.1, 0.0]])])

    results = run(index.vector_search(SAME))

    assert [r.chunk_id for r in results] == ["c-near", "c-far"]
    top = results[0]
    assert (top.url, top.title, top.text) == ("u1", "SAML setup", "near")
    assert top.score > results[1].score


@pytest.mark.parametrize("kind", ["vector", "text"])
def test_search_result_metadata_is_filled_from_the_chunk(
    make_search_index: SearchIndexFactory, kind: str
) -> None:
    # contracts.retrieval.SearchResult requires product, version, doc_type, heading_path.
    chunks = [
        make_chunk(
            "a",
            "u1",
            "saml",
            product="orchestration",
            version="11",
            doc_type="release-notes",
            heading_path=["Release Notes", "11.2"],
        ),
        make_chunk("b", "u1", "saml", product="case-manager", version=None),
    ]
    index = make_search_index([(make_doc("u1"), chunks, [SAME] * 2)])

    results = run(index.vector_search(SAME)) if kind == "vector" else run(index.text_search("saml"))
    by_id = {r.chunk_id: r for r in results}

    assert (
        by_id["a"].product,
        by_id["a"].version,
        by_id["a"].doc_type,
        by_id["a"].heading_path,
    ) == ("orchestration", "11", "release-notes", ["Release Notes", "11.2"])
    assert (by_id["b"].product, by_id["b"].version, by_id["b"].doc_type) == (
        "case-manager",
        None,
        "how-to",
    )


def test_ties_are_broken_by_chunk_id(make_search_index: SearchIndexFactory) -> None:
    chunks = [make_chunk(cid, "u1") for cid in ["c3", "c1", "c2"]]
    index = make_search_index([(make_doc("u1"), chunks, [SAME] * 3)])
    assert [r.chunk_id for r in run(index.vector_search(SAME))] == ["c1", "c2", "c3"]


def test_k_limits_number_of_results(make_search_index: SearchIndexFactory) -> None:
    chunks = [make_chunk(f"c{i}", "u1") for i in range(5)]
    index = make_search_index([(make_doc("u1"), chunks, [[1.0, 0.0, float(i)] for i in range(5)])])
    assert len(run(index.vector_search(SAME, k=2))) == 2
    assert len(run(index.vector_search(SAME, k=100))) == 5


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises_in_both_searches(
    make_search_index: SearchIndexFactory, k: int
) -> None:
    index = make_search_index([])
    with pytest.raises(ValueError, match="k must be positive"):
        run(index.vector_search(SAME, k=k))
    with pytest.raises(ValueError, match="k must be positive"):
        run(index.text_search("query", k=k))


def test_vector_search_rejects_wrong_query_dimension(make_search_index: SearchIndexFactory) -> None:
    with pytest.raises(ValueError, match="dimension"):
        run(make_search_index([]).vector_search([1.0, 0.0]))


# --- Filters (contracts.retrieval.Filters) --------------------------------------------------
# Each rule is checked in BOTH searches. Every chunk text contains "saml" and every vector
# equals the query vector, so without filters every chunk is returned by either search.

SearchKind = str  # "vector" | "text"
ALL_IDS = ["cm-howto-none", "orch-howto-10", "seq-howto-9", "seq-notes-9"]
search_kinds = pytest.mark.parametrize("kind", ["vector", "text"])


def _variety() -> list[IndexedDocument]:
    """seq-howto-9, seq-notes-9, orch-howto-10, cm-howto-none (version=None)."""
    chunks = [
        make_chunk("seq-howto-9", "u1", "saml", product="sequence", version="9"),
        make_chunk(
            "seq-notes-9", "u1", "saml", product="sequence", version="9", doc_type="release-notes"
        ),
        make_chunk("orch-howto-10", "u1", "saml", product="orchestration", version="10"),
        make_chunk("cm-howto-none", "u1", "saml", product="case-manager", version=None),
    ]
    return [(make_doc("u1"), chunks, [SAME] * len(chunks))]


def _search_ids(
    index: SearchIndex, kind: SearchKind, filters: Filters | None, k: int = 100
) -> list[str]:
    if kind == "vector":
        results = run(index.vector_search(SAME, filters, k=k))
    else:
        results = run(index.text_search("saml", filters, k=k))
    return sorted(r.chunk_id for r in results)


@search_kinds
def test_no_filters_and_all_none_filters_return_everything(
    make_search_index: SearchIndexFactory, kind: SearchKind
) -> None:
    index = make_search_index(_variety())
    assert _search_ids(index, kind, None) == ALL_IDS
    assert _search_ids(index, kind, Filters()) == ALL_IDS


@search_kinds
def test_values_in_one_field_are_or(
    make_search_index: SearchIndexFactory, kind: SearchKind
) -> None:
    index = make_search_index(_variety())
    filters = Filters(products=["sequence", "orchestration"])
    assert _search_ids(index, kind, filters) == ["orch-howto-10", "seq-howto-9", "seq-notes-9"]


@search_kinds
def test_different_fields_are_and(make_search_index: SearchIndexFactory, kind: SearchKind) -> None:
    index = make_search_index(_variety())
    filters = Filters(products=["sequence"], doc_types=["how-to"])
    assert _search_ids(index, kind, filters) == ["seq-howto-9"]


@search_kinds
def test_exclude_doc_types_removes_those_types(
    make_search_index: SearchIndexFactory, kind: SearchKind
) -> None:
    index = make_search_index(_variety())
    filters = Filters(exclude_doc_types=["release-notes"])
    assert _search_ids(index, kind, filters) == ["cm-howto-none", "orch-howto-10", "seq-howto-9"]


@search_kinds
def test_versions_filter_never_returns_chunks_without_version(
    make_search_index: SearchIndexFactory, kind: SearchKind
) -> None:
    index = make_search_index(_variety())
    # Every versioned chunk is requested; the version=None chunk must still be left out.
    ids = _search_ids(index, kind, Filters(versions=["9", "10", "11"]))
    assert "cm-howto-none" not in ids
    assert ids == ["orch-howto-10", "seq-howto-9", "seq-notes-9"]


def test_filters_apply_before_k_in_vector_search(make_search_index: SearchIndexFactory) -> None:
    # 5 non-matching chunks score HIGHER than all matching ones. An index that took the
    # top k first and filtered afterwards would return nothing.
    blockers = [make_chunk(f"x-{i}", "u1", product="orchestration", version="10") for i in range(5)]
    wanted = [make_chunk(f"s-{i}", "u1", product="sequence") for i in range(8)]
    vectors = [SAME] * 5 + [[1.0, 0.5 + i * 0.5, 0.0] for i in range(8)]  # s-0 best ... s-7 worst
    index = make_search_index([(make_doc("u1"), blockers + wanted, vectors)])

    results = run(index.vector_search(SAME, Filters(products=["sequence"]), k=5))

    assert [r.chunk_id for r in results] == ["s-0", "s-1", "s-2", "s-3", "s-4"]


def test_filters_apply_before_k_in_text_search(make_search_index: SearchIndexFactory) -> None:
    blockers = [
        make_chunk(f"x-{i}", "u1", "saml saml saml saml", product="orchestration") for i in range(5)
    ]
    wanted = [
        make_chunk(f"s-{i}", "u1", " ".join(["saml"] * (3 - min(i, 2))), product="sequence")
        for i in range(6)
    ]  # s-0: 3 matches, s-1: 2, s-2..s-5: 1
    index = make_search_index([(make_doc("u1"), blockers + wanted, [SAME] * 11)])

    results = run(index.text_search("saml", Filters(products=["sequence"]), k=5))

    assert [r.chunk_id for r in results] == ["s-0", "s-1", "s-2", "s-3", "s-4"]


# --- text_search (order only; score values are implementation-specific) ---------------------


def test_text_search_ranks_more_matches_first_case_insensitive(
    make_search_index: SearchIndexFactory,
) -> None:
    chunks = [
        make_chunk("one", "u1", "SAML is configured here"),
        make_chunk("two", "u1", "configure saml: SAML settings"),
        make_chunk("none", "u1", "release notes"),
    ]
    index = make_search_index([(make_doc("u1"), chunks, [SAME] * 3)])

    results = run(index.text_search("saml"))

    assert [r.chunk_id for r in results] == ["two", "one"]  # more matches first; no match left out
    assert results[0].score > results[1].score


def test_text_search_with_no_words_or_no_match_returns_empty(
    make_search_index: SearchIndexFactory,
) -> None:
    index = make_search_index([(make_doc("u1"), [make_chunk("a", "u1", "hello")], [SAME])])
    assert run(index.text_search("   !!! ")) == []
    assert run(index.text_search("absent")) == []


# --- end to end with FakeEmbedder -----------------------------------------------------------


def test_end_to_end_with_fake_embedder(make_search_index: SearchIndexFactory) -> None:
    embedder = FakeEmbedder(dimension=64)
    texts = {
        "saml": "how to configure SAML authentication in SeQuence",
        "notes": "release notes for case manager version 11",
    }
    chunks = [make_chunk(cid, "u1", text) for cid, text in texts.items()]
    vectors = run(embedder.embed(list(texts.values()), input_type="document"))
    index = make_search_index([(make_doc("u1"), chunks, vectors)], dimension=embedder.dimension)

    (query,) = run(embedder.embed(["configure SAML"], input_type="query"))

    assert run(index.vector_search(query, k=1))[0].chunk_id == "saml"
