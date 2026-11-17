"""Integration (Test 37): HybridRetriever + real FakeEmbedder + InMemoryIndexStore.

Step 1: wiring, end-to-end search, empty index. Step 2: filters (product, version).
Step 3: RRF fusion of the REAL index rankings.

No mocks or stubs: documents are built and loaded only through Raizy's public test helper
(kc.indexing.testing). Offline: no database, no network, no API key.
"""

import asyncio
from fractions import Fraction
from typing import Any

import pytest

from contracts.common import Product, Version
from contracts.retrieval import Filters, Retriever, SearchResult
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.testing import SampleDocument, sample_document, seed_index
from kc.retrieval.hybrid_retriever import _RRF_K, HybridRetriever
from kc.retrieval.rrf import reciprocal_rank_fusion
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


# === Step 3: RRF on the real index rankings ===============================================
#
# Tests 1-3 compare HybridRetriever with the two lists that the SAME store returns when it
# is called directly with the same query, filters and n = max(candidate_k, k).

RRF_QUERY = "configure saml authentication"
RRF_CANDIDATE_K = 3  # smaller than the corpus, so both lists are really cut to n


def _rrf_doc(
    slug: str, texts: list[str], product: Product, version: Version | None
) -> SampleDocument:
    return sample_document(
        KC_HELP + slug, slug, texts, product=product, version=version, doc_type="how-to"
    )


# 6 chunks in 5 documents; the query words appear in different amounts in each chunk.
RRF_CORPUS = [
    _rrf_doc(
        "saml-setup-sequence",
        [
            "Configure SAML authentication for the portal.",
            "SAML certificates must be renewed every year.",
        ],
        "sequence",
        "9",
    ),
    _rrf_doc("saml-orchestration", ["Configure SAML single sign-on."], "orchestration", "10"),
    _rrf_doc("ldap-sequence", ["Configure LDAP user synchronization."], "sequence", "9"),
    _rrf_doc("saml-redirect-fix", ["Fixed a SAML login redirect."], "sequence", "8"),
    _rrf_doc("expression-functions", ["Expression samples for strings."], "orchestration", "11"),
]
RRF_CHUNK_COUNT = sum(len(doc.chunk_ids) for doc in RRF_CORPUS)

RRF_CASES = [
    pytest.param(RRF_QUERY, None, 2, id="no-filter-k2"),
    pytest.param(RRF_QUERY, None, 5, id="no-filter-k5"),
    pytest.param(RRF_QUERY, Filters(products=["sequence"]), 3, id="sequence-k3"),
    pytest.param("saml", Filters(versions=["9", "10"]), 4, id="saml-versions-k4"),
]


def _direct_lists(
    embedder: FakeEmbedder,
    store: InMemoryIndexStore,
    query: str,
    filters: Filters | None,
    n: int,
) -> tuple[list[SearchResult], list[SearchResult]]:
    """The two candidate lists, straight from the index (same calls HybridRetriever makes)."""
    (vector,) = asyncio.run(embedder.embed([query], input_type="query"))
    vector_list = asyncio.run(store.vector_search(vector, filters, k=n))
    text_list = asyncio.run(store.text_search(query, filters, k=n))
    return vector_list, text_list


# --- 1. HybridRetriever == RRF of the real lists ------------------------------------------


@pytest.mark.parametrize(("query", "filters", "k"), RRF_CASES)
def test_search_equals_rrf_of_the_real_index_lists(
    query: str, filters: Filters | None, k: int
) -> None:
    embedder, store, retriever = _seeded_with(RRF_CORPUS, candidate_k=RRF_CANDIDATE_K)
    n = max(RRF_CANDIDATE_K, k)
    vector_list, text_list = _direct_lists(embedder, store, query, filters, n)
    assert vector_list and text_list  # both branches really contribute

    results = asyncio.run(retriever.search(query, filters, k=k))

    assert results == reciprocal_rank_fusion(vector_list, text_list, rrf_k=_RRF_K)[:k]


# --- 2. scores computed by hand from the real ranks ---------------------------------------


@pytest.mark.parametrize(("query", "filters", "k"), RRF_CASES)
def test_scores_are_exact_rrf_sums_of_real_ranks(
    query: str, filters: Filters | None, k: int
) -> None:
    embedder, store, retriever = _seeded_with(RRF_CORPUS, candidate_k=RRF_CANDIDATE_K)
    n = max(RRF_CANDIDATE_K, k)
    vector_list, text_list = _direct_lists(embedder, store, query, filters, n)

    # Independent of rrf.py: sum 1 / (_RRF_K + rank) over the lists, rank starting at 1.
    expected: dict[str, Fraction] = {}
    for ranked in (vector_list, text_list):
        ids = [r.chunk_id for r in ranked]
        assert len(ids) == len(set(ids))  # the index returns each chunk once
        for rank, chunk_id in enumerate(ids, start=1):
            expected[chunk_id] = expected.get(chunk_id, Fraction(0)) + Fraction(1, _RRF_K + rank)
    # Our ordering rule: score descending, ties by chunk_id; then the first k.
    expected_ids = sorted(expected, key=lambda chunk_id: (-expected[chunk_id], chunk_id))[:k]

    results = asyncio.run(retriever.search(query, filters, k=k))

    assert [r.chunk_id for r in results] == expected_ids
    for r in results:
        assert r.score == float(expected[r.chunk_id])


# --- 3. the candidate pool is max(candidate_k, k) ------------------------------------------


def test_candidate_pool_is_max_of_candidate_k_and_k() -> None:
    candidate_k = 2
    embedder, store, retriever = _seeded_with(RRF_CORPUS, candidate_k=candidate_k)

    # Precondition: without a filter vector_search returns every chunk, so the corpus
    # (6 chunks) can fill k=5 once each list asks for n = max(2, 5) = 5 candidates.
    all_vector, _ = _direct_lists(embedder, store, RRF_QUERY, None, n=100)
    assert len(all_vector) == RRF_CHUNK_COUNT >= 5
    # With only candidate_k=2 per list the pool could never reach 5 results.
    top2_vector, top2_text = _direct_lists(embedder, store, RRF_QUERY, None, n=candidate_k)
    pool_of_two = {r.chunk_id for r in top2_vector} | {r.chunk_id for r in top2_text}
    assert len(pool_of_two) < 5

    # k > candidate_k: the pool grows to k.
    assert len(asyncio.run(retriever.search(RRF_QUERY, k=5))) == 5

    # k < candidate_k: the pool stays candidate_k per list.
    (top,) = asyncio.run(retriever.search(RRF_QUERY, k=1))
    assert top.chunk_id in pool_of_two
    assert [top] == reciprocal_rank_fusion(top2_vector, top2_text, rrf_k=_RRF_K)[:1]


# --- 4. identical content in two documents -------------------------------------------------


def test_identical_content_in_two_documents_is_not_merged() -> None:
    # SEQ_9 and ORCH_10 have the same chunk text (cross-product duplicate) but their own
    # chunk ids, so they get the same vector and the same text score.
    embedder, store, retriever = _seeded_with([SEQ_9, ORCH_10])
    (seq_id,) = SEQ_9.chunk_ids
    (orch_id,) = ORCH_10.chunk_ids
    assert seq_id != orch_id
    low_id, high_id = sorted([seq_id, orch_id])

    # Precondition, checked on the real index: equal scores are ordered by chunk_id
    # (SearchIndex contract), so the lower id is rank 1 and the higher rank 2 in BOTH lists.
    vector_list, text_list = _direct_lists(embedder, store, EXPR_QUERY, None, n=10)
    assert [r.chunk_id for r in vector_list] == [low_id, high_id]
    assert [r.chunk_id for r in text_list] == [low_id, high_id]

    results = asyncio.run(retriever.search(EXPR_QUERY, k=10))

    assert [r.chunk_id for r in results] == [low_id, high_id]  # two results, not merged
    assert {r.url for r in results} == {SEQ_9.url, ORCH_10.url}
    assert [r.score for r in results] == [
        float(Fraction(2, _RRF_K + 1)),  # rank 1 in both lists
        float(Fraction(2, _RRF_K + 2)),  # rank 2 in both lists
    ]


# === Test 38: document lifecycle (update / delete) ========================================
#
# A document is identified by its url: seed_index -> index_document -> upsert_document
# replaces ALL chunks (and their vectors) of that url at once; delete_document removes
# them. HybridRetriever keeps no cache, so the SAME instance sees every change.
#
# Two consequences for the assertions:
# - chunk ids are sha256(url | heading_path | part), so after an update part 0 keeps its
#   id: old vs new content is told apart by TEXT, plus the id of a removed part;
# - vector_search returns every stored chunk, so a query for an old word can still return
#   the document's NEW chunk: "old content is gone" is checked on texts and ids, not url.

LIFECYCLE_URL = KC_HELP + "zephyr-token-setup"
OTHER_URL = KC_HELP + "ldap-sync-guide"
OLD_WORD = "zephyrtoken"
NEW_WORD = "quasartoken"
OTHER_WORD = "ldapsync"


def _lifecycle_doc(texts: list[str]) -> SampleDocument:
    return sample_document(
        LIFECYCLE_URL, "Token Setup", texts, product="sequence", version="9", doc_type="how-to"
    )


OLD_DOC = _lifecycle_doc([f"Legacy {OLD_WORD} setup steps.", f"{OLD_WORD} troubleshooting."])
NEW_DOC = _lifecycle_doc([f"Updated {NEW_WORD} setup steps."])  # same url, one chunk only
OTHER_DOC = sample_document(
    OTHER_URL,
    "LDAP Sync",
    [f"Configure {OTHER_WORD} for user accounts."],
    product="orchestration",
    version="10",
    doc_type="how-to",
)


def _find(retriever: HybridRetriever, query: str) -> list[SearchResult]:
    return asyncio.run(retriever.search(query, k=10))


def _texts(results: list[SearchResult]) -> str:
    return " ".join(r.text for r in results).lower()


def test_initial_upsert_is_searchable() -> None:
    embedder, store, retriever = _components()
    asyncio.run(seed_index(store, embedder, [OLD_DOC]))

    results = _find(retriever, OLD_WORD)

    assert {r.chunk_id for r in results} == set(OLD_DOC.chunk_ids)
    assert {r.url for r in results} == {LIFECYCLE_URL}
    assert OLD_WORD in _texts(results)


def test_repeated_upsert_replaces_old_content() -> None:
    embedder, store, retriever = _components()
    asyncio.run(seed_index(store, embedder, [OLD_DOC]))
    assert OLD_WORD in _texts(_find(retriever, OLD_WORD))  # old content searchable first

    # Same url = same document. Part 0 keeps its id; part 1 no longer exists.
    assert NEW_DOC.url == OLD_DOC.url
    assert NEW_DOC.chunk_ids[0] == OLD_DOC.chunk_ids[0]
    removed_id = OLD_DOC.chunk_ids[1]
    asyncio.run(seed_index(store, embedder, [NEW_DOC]))

    new_results = _find(retriever, NEW_WORD)
    assert {r.chunk_id for r in new_results} == set(NEW_DOC.chunk_ids)
    assert {r.url for r in new_results} == {LIFECYCLE_URL}
    assert NEW_WORD in _texts(new_results)

    old_results = _find(retriever, OLD_WORD)
    assert OLD_WORD not in _texts(old_results)  # neither the text nor the vector survived
    assert removed_id not in {r.chunk_id for r in old_results}
    assert {r.chunk_id for r in old_results} <= set(NEW_DOC.chunk_ids)


def test_delete_removes_the_document_from_later_searches() -> None:
    embedder, store, retriever = _components()
    asyncio.run(seed_index(store, embedder, [OLD_DOC]))
    assert {r.url for r in _find(retriever, OLD_WORD)} == {LIFECYCLE_URL}

    asyncio.run(store.delete_document(LIFECYCLE_URL))

    # It was the only document: both searches are now empty.
    assert _find(retriever, OLD_WORD) == []


def test_other_documents_survive_update_and_delete() -> None:
    embedder, store, retriever = _components()
    asyncio.run(seed_index(store, embedder, [OLD_DOC, OTHER_DOC]))

    def assert_other_is_found() -> None:
        results = _find(retriever, OTHER_WORD)
        assert set(OTHER_DOC.chunk_ids) <= {r.chunk_id for r in results}
        assert OTHER_WORD in _texts(results)

    assert_other_is_found()

    asyncio.run(seed_index(store, embedder, [NEW_DOC]))  # update the lifecycle document
    assert_other_is_found()

    asyncio.run(store.delete_document(LIFECYCLE_URL))  # delete the lifecycle document
    assert_other_is_found()
    remaining = _find(retriever, f"{OLD_WORD} {NEW_WORD} {OTHER_WORD}")
    assert {r.url for r in remaining} == {OTHER_URL}
