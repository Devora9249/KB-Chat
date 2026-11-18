"""Synthetic retriever for SMOKE-TESTING the evaluation pipeline (Search & Ranking).

`create_evaluation_retriever(cases)` builds a small in-memory index from the Golden Set
itself and returns a `contracts.retrieval.Retriever` over it, marked `is_synthetic = True`.
It lets the evaluation runner check end to end that loading cases, indexing, retrieval and
scoring are wired correctly, without a crawled corpus, a database or an embedding provider.

NOT a quality measurement. Each document is built from the very questions that point to
it, and `FakeEmbedder` vectors are hashed bags of words, not semantic. Recall/MRR computed
on this retriever are artificially high and say nothing about real retrieval quality. Real
evaluation needs the real corpus, the real embedder and the real storage backend
(PgIndexStore); reports should show `is_synthetic` so the two are never confused.

The factory is typed to return `Retriever`, so mypy does not know `is_synthetic`. Read it
with a default, which also works for real retrievers that do not have the attribute:

    retriever = await create_evaluation_retriever(cases)
    synthetic = getattr(retriever, "is_synthetic", False)

Synthetic corpus (deterministic, no I/O besides the in-memory index):
- one document per unique URL in `expected_urls`, in order of first appearance;
- one chunk per (case, URL) pair, text = that case's question, in case order (a URL listed
  twice in the same case still gives one chunk);
- negative cases (`expected_urls=[]`) add nothing; no cases -> a valid, empty index;
- every document: doc_type="other", version=None, title derived from the URL slug;
- product: the non-None `EvalCase.product` values of the cases that point to the URL
  (must be one valid `Product`); only when none exists, `_FALLBACK_PRODUCTS` (explicit,
  agreed mapping). Conflicting, invalid or unresolvable products raise ValueError.
"""

from collections.abc import Sequence
from typing import Any, TypeGuard, get_args

from contracts.common import Product
from contracts.evaluation import EvalCase
from contracts.retrieval import Filters, Retriever, SearchResult
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.in_memory_index_store import InMemoryIndexStore
from kc.indexing.testing import SampleDocument, sample_document, seed_index
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RELEASE_NOTES_WEIGHT,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_TIMEOUT_S,
)
from kc.retrieval.settings import RetrievalSettings, create_hybrid_retriever

__all__ = ["SyntheticEvaluationRetriever", "create_evaluation_retriever"]

_KC_HELP = "https://knowledgecenter.gcora.genpact.com/help/"

# Products for Golden Set URLs whose cases are all product-agnostic (product=None).
# Agreed explicitly; never inferred from the URL or the title.
_FALLBACK_PRODUCTS: dict[str, Product] = {
    _KC_HELP + "job-execution-service-jes-overview": "sequence",
    _KC_HELP + "installation-deployment-and-configuration-job-execution-service-jes-overview": (
        "orchestration"
    ),
    # Today a Golden Set case already gives "sequence" for this URL; this entry is only a
    # fallback for when no case that expects it provides a product.
    _KC_HELP + "email-delivery-issues": "sequence",
}

_PRODUCTS: frozenset[str] = frozenset(get_args(Product))


class SyntheticEvaluationRetriever:
    """A `Retriever` over the synthetic corpus. `search` delegates to HybridRetriever."""

    def __init__(self, retriever: Retriever, documents: Sequence[SampleDocument]) -> None:
        self._retriever = retriever
        self._corpus = tuple(documents)

    @property
    def is_synthetic(self) -> bool:
        """Always True: results come from a synthetic corpus, not the Knowledge Center."""
        return True

    @property
    def _documents(self) -> tuple[SampleDocument, ...]:
        """Internal, for this domain's tests only: the synthetic documents that were indexed.

        Not part of the evaluation API (SampleDocument is a test helper type that may
        change); callers use `search` and `is_synthetic` only.
        """
        return self._corpus

    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        """See `contracts.retrieval.Retriever.search` (same validation and errors)."""
        return await self._retriever.search(query, filters, k)


async def create_evaluation_retriever(cases: Sequence[EvalCase]) -> Retriever:
    """Build a synthetic smoke-test retriever from `cases` (see the module docstring).

    The result is a `SyntheticEvaluationRetriever`: `is_synthetic` is True and its scores
    are NOT retrieval-quality measurements. Async on purpose: it is awaited inside the
    runner's event loop (no asyncio.run here).

    Raises ValueError for a URL with conflicting, invalid or unresolvable products, or a
    URL that is not a Knowledge Center article URL (from `sample_document`).
    """
    documents = _build_documents(cases)

    embedder = FakeEmbedder()
    store = InMemoryIndexStore(dimension=embedder.dimension, model_name=embedder.model_name)
    await seed_index(store, embedder, documents)

    # Explicit values + no .env file: neither a local .env nor RETRIEVAL_* environment
    # variables can change the synthetic run (init arguments win over the environment).
    values: dict[str, Any] = {
        "RETRIEVAL_CANDIDATE_K": DEFAULT_CANDIDATE_K,
        "RETRIEVAL_TIMEOUT_S": DEFAULT_TIMEOUT_S,
        "RETRIEVAL_RELEASE_NOTES_WEIGHT": DEFAULT_RELEASE_NOTES_WEIGHT,
        # No reranker is passed; set explicitly anyway so the environment cannot break us.
        "RETRIEVAL_RERANK_CANDIDATES": DEFAULT_RERANK_CANDIDATES,
        "RETRIEVAL_RERANK_TIMEOUT_S": DEFAULT_RERANK_TIMEOUT_S,
    }
    settings = RetrievalSettings(_env_file=None, **values)  # type: ignore[call-arg]
    retriever = create_hybrid_retriever(embedder, store, settings)
    return SyntheticEvaluationRetriever(retriever, documents)


def _build_documents(cases: Sequence[EvalCase]) -> list[SampleDocument]:
    questions: dict[str, list[str]] = {}  # url -> chunk texts; dict keeps first-seen order
    products: dict[str, set[str]] = {}  # url -> non-None products of the cases
    for case in cases:
        for url in dict.fromkeys(case.expected_urls):  # unique within the case, in order
            questions.setdefault(url, []).append(case.question)
            seen = products.setdefault(url, set())
            if case.product is not None:
                seen.add(case.product)

    return [
        sample_document(
            url,
            _title(url),
            texts,
            product=_resolve_product(url, products[url]),
            version=None,
            doc_type="other",
        )
        for url, texts in questions.items()
    ]


def _resolve_product(url: str, products: set[str]) -> Product:
    valid: set[Product] = set()
    for product in sorted(products):
        if not _is_product(product):
            raise ValueError(
                f"invalid product {product!r} for {url!r}; expected one of {sorted(_PRODUCTS)}"
            )
        valid.add(product)
    if len(valid) > 1:
        raise ValueError(f"conflicting products {sorted(valid)} for {url!r}")
    if valid:
        return valid.pop()
    if url in _FALLBACK_PRODUCTS:
        return _FALLBACK_PRODUCTS[url]
    raise ValueError(
        f"no product for {url!r}: every case that expects it has product=None and the URL "
        "is not in the explicit fallback mapping"
    )


def _is_product(value: str) -> TypeGuard[Product]:
    return value in _PRODUCTS


def _title(url: str) -> str:
    return url.removeprefix(_KC_HELP).replace("-", " ")
