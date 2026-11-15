"""Tests for the Search & Ranking contracts (Filters, SearchResult, Retriever)."""

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.retrieval import Filters, Retriever, SearchResult

LIST_FIELDS = ["products", "versions", "doc_types", "exclude_doc_types"]


def _result_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_id": "c1",
        "url": "https://knowledgecenter.gcora.genpact.com/help/saml-configuration",
        "title": "SAML Configuration",
        "text": "To configure SAML...",
        "score": 0.87,
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
        "heading_path": ["Administration", "Authentication", "SAML"],
    }
    base.update(overrides)
    return base


# --- Filters ---------------------------------------------------------------


def test_filters_defaults_are_none() -> None:
    f = Filters()
    assert f.products is None
    assert f.versions is None
    assert f.doc_types is None
    assert f.exclude_doc_types is None


def test_filters_accepts_valid_values() -> None:
    f = Filters(
        products=["orchestration", "sequence"],
        versions=["10", "9"],
        doc_types=["how-to"],
        exclude_doc_types=["release-notes"],
    )
    assert f.products == ["orchestration", "sequence"]
    assert f.exclude_doc_types == ["release-notes"]


@pytest.mark.parametrize("field", LIST_FIELDS)
def test_filters_rejects_empty_list(field: str) -> None:
    with pytest.raises(ValueError, match="empty list"):
        Filters(**{field: []})


def test_filters_rejects_doc_type_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Filters(doc_types=["how-to", "release-notes"], exclude_doc_types=["release-notes"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("products", ["Sequence"]),
        ("versions", ["9.x"]),
        ("versions", ["9.0"]),
        ("doc_types", ["guide"]),
    ],
)
def test_filters_rejects_values_outside_closed_set(field: str, value: list[Any]) -> None:
    with pytest.raises(ValidationError):
        Filters(**{field: value})


def test_filters_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Filters(product=["sequence"])  # type: ignore[call-arg]


def test_filters_is_frozen() -> None:
    f = Filters(products=["sequence"])
    with pytest.raises(ValidationError):
        f.products = ["orchestration"]


# --- SearchResult ----------------------------------------------------------


def test_search_result_valid() -> None:
    r = SearchResult(**_result_kwargs())
    assert r.product == "sequence"
    assert r.version == "9"


def test_search_result_version_none_is_allowed() -> None:
    assert SearchResult(**_result_kwargs(version=None)).version is None


@pytest.mark.parametrize("field", ["product", "version", "doc_type", "heading_path"])
def test_search_result_new_fields_are_required(field: str) -> None:
    kwargs = _result_kwargs()
    del kwargs[field]
    with pytest.raises(ValidationError):
        SearchResult(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [("product", "Sequence"), ("version", "9.x"), ("doc_type", "guide")],
)
def test_search_result_rejects_values_outside_closed_set(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        SearchResult(**_result_kwargs(**{field: value}))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_search_result_rejects_nan_and_infinity(score: float) -> None:
    with pytest.raises(ValidationError):
        SearchResult(**_result_kwargs(score=score))


# --- Retriever protocol ----------------------------------------------------


class _StubRetriever:
    """Minimal structural implementation, used only to type-check the protocol."""

    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        return [SearchResult(**_result_kwargs())][:k]


def test_retriever_protocol_is_satisfied_structurally() -> None:
    retriever: Retriever = _StubRetriever()  # mypy verifies the signature matches
    results = asyncio.run(retriever.search("How do I configure SAML?", Filters(), k=1))
    assert len(results) == 1
