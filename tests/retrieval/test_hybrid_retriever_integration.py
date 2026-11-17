"""Integration (Test 37, step 1): HybridRetriever + real FakeEmbedder + InMemoryIndexStore.

No mocks or stubs: documents are built and loaded only through Raizy's public test helper
(kc.indexing.testing). Offline: no database, no network, no API key.
"""

import asyncio

from contracts.retrieval import Retriever, SearchResult
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


def _components() -> tuple[FakeEmbedder, InMemoryIndexStore, HybridRetriever]:
    embedder = FakeEmbedder(dimension=DIM, model_name=MODEL)
    store = InMemoryIndexStore(dimension=DIM, model_name=MODEL)
    # Explicit values, no .env. One line, so the ignore sits where mypy reports the error.
    settings = RetrievalSettings(_env_file=None, RETRIEVAL_CANDIDATE_K=10, RETRIEVAL_TIMEOUT_S=5.0)  # type: ignore[call-arg]
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
