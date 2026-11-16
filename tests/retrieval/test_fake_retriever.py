"""Tests for FakeRetriever (Search & Ranking fake)."""

import asyncio
from typing import Any

import pytest

from contracts.retrieval import Filters, Retriever, SearchResult
from kc.retrieval._fake_data import DEFAULT_CORPUS
from kc.retrieval.fake_retriever import FakeRetriever


def _item(chunk_id: str, text: str, **overrides: Any) -> SearchResult:
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "url": f"https://knowledgecenter.gcora.genpact.com/help/{chunk_id}",
        "title": "Title",
        "text": text,
        "score": 0.0,
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
        "heading_path": ["Section"],
    }
    base.update(overrides)
    return SearchResult(**base)


def _search(
    retriever: FakeRetriever, query: str, filters: Filters | None = None, k: int = 5
) -> list[SearchResult]:
    return asyncio.run(retriever.search(query, filters, k))


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.chunk_id for r in results]


# --- Protocol ----------------------------------------------------------------------------


def test_satisfies_retriever_protocol() -> None:
    retriever: Retriever = FakeRetriever()  # mypy checks the structural match
    results = asyncio.run(retriever.search("configure SAML authentication"))
    assert results
    assert all(isinstance(r, SearchResult) for r in results)


# --- Scoring parity with InMemoryIndexStore.text_search ----------------------------------


def test_scoring_matches_in_memory_index_store_algorithm() -> None:
    retriever = FakeRetriever(
        [
            _item("a", "saml saml config"),
            _item("b", "SAML"),
            _item("c", "unrelated words", title="saml", heading_path=["saml"]),
        ]
    )
    # Repeated query word counts once; occurrences in text are counted; title/heading ignored.
    results = _search(retriever, "SAML saml")
    assert [(r.chunk_id, r.score) for r in results] == [("a", 2.0), ("b", 1.0)]

    results = _search(retriever, "saml config")
    assert [(r.chunk_id, r.score) for r in results] == [("a", 3.0), ("b", 1.0)]


def test_is_case_insensitive() -> None:
    retriever = FakeRetriever([_item("a", "Configure SAML")])
    assert _search(retriever, "configure saml") == _search(retriever, "CONFIGURE SAML")


def test_no_match_returns_empty_list() -> None:
    assert _search(FakeRetriever(), "xyzzy nonexistent") == []


@pytest.mark.parametrize("query", ["", "   ", "?! ..."])
def test_empty_query_returns_empty_list_fake_behavior(query: str) -> None:
    # FakeRetriever decision, not a requirement of the shared contract.
    assert _search(FakeRetriever(), query) == []


def test_exact_technical_name_ranks_first() -> None:
    results = _search(FakeRetriever(), "Set-CoraSeQuenceSiteAuthentication")
    assert results[0].chunk_id == "fake-seq9-cmdlet-0"


# --- k, sorting, uniqueness --------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_raises(k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        _search(FakeRetriever(), "saml", k=k)


@pytest.mark.parametrize("k", [True, False, 5.0, "5", None])
def test_k_wrong_type_raises_type_error(k: Any) -> None:
    # Contract: a non-int k, including bool, is a TypeError (True is NOT treated as 1).
    with pytest.raises(TypeError, match="k must be an int"):
        _search(FakeRetriever(), "saml", k=k)


@pytest.mark.parametrize(("k", "error"), [(5.0, TypeError), (True, TypeError), (0, ValueError)])
def test_k_is_validated_before_the_query(k: Any, error: type[Exception]) -> None:
    # A query with no tokens returns [] early; an invalid k must still be rejected.
    with pytest.raises(error):
        _search(FakeRetriever(), "?!", k=k)


def test_results_are_cut_to_k() -> None:
    retriever = FakeRetriever([_item(f"c{i}", "saml") for i in range(5)])
    assert len(_search(retriever, "saml", k=2)) == 2


def test_default_k_is_5() -> None:
    retriever = FakeRetriever([_item(f"c{i}", "saml") for i in range(8)])
    assert len(asyncio.run(retriever.search("saml"))) == 5


def test_may_return_fewer_than_k() -> None:
    retriever = FakeRetriever([_item("a", "saml"), _item("b", "other")])
    assert _ids(_search(retriever, "saml", k=5)) == ["a"]


def test_sorted_by_score_desc_then_chunk_id_asc() -> None:
    retriever = FakeRetriever(
        [
            _item("zeta", "saml"),
            _item("alpha", "saml"),
            _item("mid", "saml saml"),
        ]
    )
    assert _ids(_search(retriever, "saml")) == ["mid", "alpha", "zeta"]


def test_results_have_unique_chunk_ids() -> None:
    results = _search(FakeRetriever(), "saml authentication expression release", k=50)
    assert len(_ids(results)) == len(set(_ids(results)))


def test_duplicate_chunk_id_in_corpus_raises() -> None:
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        FakeRetriever([_item("dup", "a"), _item("dup", "b")])


def test_is_deterministic() -> None:
    first = _search(FakeRetriever(), "saml authentication", k=10)
    second = _search(FakeRetriever(), "saml authentication", k=10)
    assert first == second


# --- Filters -----------------------------------------------------------------------------


@pytest.fixture
def filter_corpus() -> FakeRetriever:
    return FakeRetriever(
        [
            _item("seq9-howto", "saml", product="sequence", version="9", doc_type="how-to"),
            _item("seq8-ref", "saml", product="sequence", version="8", doc_type="reference"),
            _item(
                "orch11-rn", "saml", product="orchestration", version="11", doc_type="release-notes"
            ),
            _item("orch-none", "saml", product="orchestration", version=None, doc_type="overview"),
            _item("cm4-howto", "saml", product="case-manager", version="4", doc_type="how-to"),
        ]
    )


def test_no_filters_returns_all_matches(filter_corpus: FakeRetriever) -> None:
    assert len(_search(filter_corpus, "saml", Filters(), k=10)) == 5


def test_products_filter(filter_corpus: FakeRetriever) -> None:
    results = _search(filter_corpus, "saml", Filters(products=["sequence"]), k=10)
    assert sorted(_ids(results)) == ["seq8-ref", "seq9-howto"]


def test_or_within_field(filter_corpus: FakeRetriever) -> None:
    results = _search(filter_corpus, "saml", Filters(products=["sequence", "case-manager"]), k=10)
    assert sorted(_ids(results)) == ["cm4-howto", "seq8-ref", "seq9-howto"]


def test_and_across_fields(filter_corpus: FakeRetriever) -> None:
    filters = Filters(products=["sequence"], doc_types=["how-to"])
    assert _ids(_search(filter_corpus, "saml", filters, k=10)) == ["seq9-howto"]


def test_versions_filter_excludes_unversioned_items(filter_corpus: FakeRetriever) -> None:
    results = _search(filter_corpus, "saml", Filters(products=["orchestration"], versions=["11"]))
    assert _ids(results) == ["orch11-rn"]  # "orch-none" (version=None) is not included


def test_doc_types_filter(filter_corpus: FakeRetriever) -> None:
    results = _search(filter_corpus, "saml", Filters(doc_types=["reference", "overview"]), k=10)
    assert sorted(_ids(results)) == ["orch-none", "seq8-ref"]


def test_exclude_doc_types_filter(filter_corpus: FakeRetriever) -> None:
    results = _search(filter_corpus, "saml", Filters(exclude_doc_types=["release-notes"]), k=10)
    assert "orch11-rn" not in _ids(results)
    assert len(results) == 4


def test_cross_product_duplicate_is_separated_by_product_filter() -> None:
    results = _search(FakeRetriever(), "expression samples", Filters(products=["sequence"]))
    assert "fake-seq9-expr-0" in _ids(results)
    assert "fake-orch10-expr-0" not in _ids(results)
    assert all(r.product == "sequence" for r in results)


# --- Isolation (deep copies in and out) --------------------------------------------------


def test_mutating_returned_result_does_not_affect_later_searches() -> None:
    retriever = FakeRetriever([_item("a", "saml", heading_path=["Admin"])])
    (first,) = _search(retriever, "saml")
    first.text = "changed"
    first.heading_path.append("Injected")

    (second,) = _search(retriever, "saml")
    assert second.text == "saml"
    assert second.heading_path == ["Admin"]


def test_mutating_input_corpus_does_not_affect_retriever() -> None:
    item = _item("a", "saml", heading_path=["Admin"])
    corpus = [item]
    retriever = FakeRetriever(corpus)

    item.heading_path.append("Injected")
    item.text = "changed"
    corpus.append(_item("b", "saml"))

    (result,) = _search(retriever, "saml")
    assert result.chunk_id == "a"
    assert result.text == "saml"
    assert result.heading_path == ["Admin"]


def test_each_call_returns_new_objects() -> None:
    retriever = FakeRetriever([_item("a", "saml")])
    (first,) = _search(retriever, "saml")
    (second,) = _search(retriever, "saml")
    assert first is not second
    assert first.heading_path is not second.heading_path
    assert first == second


# --- Default corpus ----------------------------------------------------------------------


def test_default_corpus_is_marked_as_fake() -> None:
    assert all(item.chunk_id.startswith("fake-") for item in DEFAULT_CORPUS)


def test_default_corpus_covers_all_products_and_sensitive_cases() -> None:
    assert {item.product for item in DEFAULT_CORPUS} == {
        "sequence",
        "orchestration",
        "case-manager",
        "opsmanager",
    }
    assert any(item.doc_type == "release-notes" for item in DEFAULT_CORPUS)
    assert any(item.version is None for item in DEFAULT_CORPUS)
    expression_products = {
        item.product for item in DEFAULT_CORPUS if item.title == "Expression Samples"
    }
    assert expression_products == {"sequence", "orchestration"}


def test_default_corpus_has_unique_chunk_ids() -> None:
    ids = [item.chunk_id for item in DEFAULT_CORPUS]
    assert len(ids) == len(set(ids))
