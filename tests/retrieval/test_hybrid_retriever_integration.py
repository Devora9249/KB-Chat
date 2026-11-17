"""Integration (Test 37): HybridRetriever + real FakeEmbedder + InMemoryIndexStore.

Step 1: wiring, end-to-end search, empty index. Step 2: filters (product, version).

No mocks or stubs: documents are built and loaded only through Raizy's public test helper
(kc.indexing.testing). Offline: no database, no network, no API key.
"""

import asyncio
from typing import Any

from contracts.common import Product, Version
from contracts.retrieval import Filters, Retriever, SearchResult
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.testing import SampleDocument, sample_document, seed_index
from kc.retrieval.hybrid_retriever import _RRF_K, HybridRetriever
from kc.retrieval.settings import RetrievalSettings, create_hybrid_retriever

DIM = 64
MODEL = "fake-embedder-v1"
# Best possible contribution of ONE result list in RRF: rank 1 -> 1 / (_RRF_K + 1)
# (ranks start at 1). A chunk found by both searches always scores higher than this.
ONE_LIST_MAX = 1 / (_RRF_K + 1)

URL = "https://knowledgecenter.gcora.genpact.com/help/configure-saml-sequence"
TITLE = "Configure SAML Authentication"
SAML_TEXT = "Use SAML to sign in with your identity provider."
CMDLET_TEXT = "Run Set-CoraSeQuenceSiteAuthentication -AuthenticationType SAML."


def _components(
    candidate_k: int = 10,
) -> tuple[FakeEmbedder, InMemoryIndexStore, HybridRetriever]:
    embedder = FakeEmbedder(dimension=DIM, model_name=MODEL)
    store = InMemoryIndexStore(dimension=DIM, model_name=MODEL)
    # Explicit values (by env alias), no .env. The call stays on one line, so the ignore
    # sits where mypy reports the error.
    values: dict[str, Any] = {"RETRIEVAL_CANDIDATE_K": candidate_k, "RETRIEVAL_TIMEOUT_S": 5.0}
    settings = RetrievalSettings(_env_file=None, **values)  # type: ignore[call-arg]
    return embedder, store, create_hybrid_retriever(embedder, store, settings)


def _saml_document() -> SampleDocument:
    return sample_document(
        URL,
        TITLE,
        [SAML_TEXT, CMDLET_TEXT],
        product="sequence",
        version="9",
        doc_type="how-to",
    )


def _seeded() -> tuple[HybridRetriever, SampleDocument]:
    embedder, store, retriever = _components()
    sample = _saml_document()
    asyncio.run(seed_index(store, embedder, [sample]))
    return retriever, sample


# --- 1. wiring ---------------------------------------------------------------------------


def test_components_are_wired_with_matching_dimension_and_model() -> None:
    embedder, store, retriever = _components()
    assert embedder.dimension == store.dimension == DIM
    assert embedder.model_name == store.model_name == MODEL
    as_protocol: Retriever = retriever  # Retriever is not runtime_checkable: mypy checks this
    assert as_protocol is retriever


# --- 2 + 3. seed a document, search end to end -------------------------------------------


def test_exact_technical_name_finds_its_chunk_first() -> None:
    retriever, sample = _seeded()

    results = asyncio.run(retriever.search("Set-CoraSeQuenceSiteAuthentication", k=5))

    # vector_search returns every chunk, so both chunks come back, and nothing else.
    assert {r.chunk_id for r in results} <= set(sample.chunk_ids)
    top = results[0]
    assert CMDLET_TEXT in top.text
    # Only the cmdlet chunk matches the text search too, so it is the only one found by
    # BOTH searches: its score beats any single-list score, whatever the vector order.
    assert top.score > ONE_LIST_MAX
    assert all(r.score <= ONE_LIST_MAX for r in results[1:])


def test_results_keep_url_and_metadata() -> None:
    retriever, sample = _seeded()

    results = asyncio.run(retriever.search("configure SAML", k=5))

    assert results
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.chunk_id in sample.chunk_ids
        assert r.url == sample.url
        assert r.title == sample.title
        assert (r.product, r.version, r.doc_type) == (
            sample.product,
            sample.version,
            sample.doc_type,
        )
        assert r.heading_path == [TITLE]  # helper default
        assert 0 < r.score <= 2 * ONE_LIST_MAX
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len({r.chunk_id for r in results}) == len(results)


# --- 4. empty index ----------------------------------------------------------------------


def test_empty_index_returns_empty_list() -> None:
    _, _, retriever = _components()
    assert asyncio.run(retriever.search("configure SAML", k=5)) == []


# === Step 2: filters ======================================================================
#
# Every document below has ONE chunk with the same text, so each one matches the text
# search for EXPR_QUERY. With k=10 (> number of documents), every document allowed by the
# filters is therefore guaranteed to come back; set equality is justified, order is not.

KC_HELP = "https://knowledgecenter.gcora.genpact.com/help/"
EXPR_TITLE = "Expression Samples"
EXPR_TEXT = "Expression samples for string and date functions."
EXPR_QUERY = "expression samples"


def _expr_doc(slug: str, product: Product, version: Version | None) -> SampleDocument:
    return sample_document(
        KC_HELP + slug,
        EXPR_TITLE,
        [EXPR_TEXT],
        product=product,
        version=version,
        doc_type="how-to",
    )


SEQ_9 = _expr_doc("expression-samples-sequence-9", "sequence", "9")
SEQ_8 = _expr_doc("expression-samples-sequence-8", "sequence", "8")
SEQ_NO_VERSION = _expr_doc("expression-samples-sequence", "sequence", None)
ORCH_10 = _expr_doc("expression-samples-orchestration-10", "orchestration", "10")


def _seeded_with(
    documents: list[SampleDocument], *, candidate_k: int = 10
) -> tuple[FakeEmbedder, InMemoryIndexStore, HybridRetriever]:
    embedder, store, retriever = _components(candidate_k)
    asyncio.run(seed_index(store, embedder, documents))
    return embedder, store, retriever


def _search(retriever: HybridRetriever, filters: Filters | None) -> list[SearchResult]:
    return asyncio.run(retriever.search(EXPR_QUERY, filters, k=10))


def _assert_only(results: list[SearchResult], sample: SampleDocument) -> None:
    """Results are exactly the chunks of `sample` (and nothing from any other document)."""
    assert {r.chunk_id for r in results} == set(sample.chunk_ids)
    assert {r.url for r in results} == {sample.url}


# --- 1. product ---------------------------------------------------------------------------


def test_product_filter_separates_same_title_across_products() -> None:
    _, _, retriever = _seeded_with([SEQ_9, ORCH_10])

    unfiltered = _search(retriever, None)
    assert {r.url for r in unfiltered} == {SEQ_9.url, ORCH_10.url}

    filtered = _search(retriever, Filters(products=["sequence"]))
    _assert_only(filtered, SEQ_9)
    assert all(r.product == "sequence" for r in filtered)


# --- 2. version ---------------------------------------------------------------------------


def test_version_filter_returns_only_that_version_and_not_unversioned() -> None:
    _, _, retriever = _seeded_with([SEQ_9, SEQ_8, SEQ_NO_VERSION])

    results = _search(retriever, Filters(versions=["9"]))

    _assert_only(results, SEQ_9)
    assert SEQ_NO_VERSION.url not in {r.url for r in results}  # version=None never matches


# --- 3. product AND version ---------------------------------------------------------------


def test_product_and_version_filters_are_combined() -> None:
    _, _, retriever = _seeded_with([SEQ_9, SEQ_8, SEQ_NO_VERSION, ORCH_10])

    results = _search(retriever, Filters(products=["sequence"], versions=["9"]))

    _assert_only(results, SEQ_9)


# --- 4. no match --------------------------------------------------------------------------


def test_filter_without_matches_returns_empty_list() -> None:
    # Without filters vector_search returns every chunk; a filter with no match must
    # still end in [] (both searches empty, nothing to fuse).
    _, _, retriever = _seeded_with([SEQ_9, SEQ_8, SEQ_NO_VERSION, ORCH_10])
    assert _search(retriever, Filters(products=["opsmanager"])) == []


# --- 5. filters apply before candidate_k --------------------------------------------------


def test_filter_is_applied_before_choosing_candidates() -> None:
    query = "configure saml authentication portal"
    best = sample_document(
        KC_HELP + "configure-saml-orchestration",
        "Configure SAML",
        ["Configure SAML authentication portal."],  # every query word
        product="orchestration",
        version="10",
        doc_type="how-to",
    )
    allowed = sample_document(
        KC_HELP + "saml-reference-sequence",
        "SAML Reference",
        ["SAML settings reference table."],  # one query word only
        product="sequence",
        version="9",
        doc_type="how-to",
    )
    embedder, store, retriever = _seeded_with([best, allowed], candidate_k=1)

    # Precondition, checked on the real index: WITHOUT a filter, `best` is the single top
    # candidate of BOTH searches. Filtering after taking the top 1 would leave nothing.
    (query_vector,) = asyncio.run(embedder.embed([query], input_type="query"))
    top_vector = asyncio.run(store.vector_search(query_vector, k=1))
    top_text = asyncio.run(store.text_search(query, k=1))
    assert [r.chunk_id for r in top_vector] == list(best.chunk_ids)
    assert [r.chunk_id for r in top_text] == list(best.chunk_ids)

    results = asyncio.run(retriever.search(query, Filters(products=["sequence"]), k=1))

    _assert_only(results, allowed)
