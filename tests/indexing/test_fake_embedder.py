"""Contract tests for FakeEmbedder (draft storage contract D1-D5)."""

import asyncio
import math
from typing import Any

import pytest

from kc.indexing._provisional_contracts import InputType
from kc.indexing.fake_embedder import FakeEmbedder
from kc.indexing.vector_math import cosine_similarity, norm


def _embed(embedder: FakeEmbedder, texts: list[str], input_type: InputType) -> list[list[float]]:
    return asyncio.run(embedder.embed(texts, input_type=input_type))


# --- D1: model identity -------------------------------------------------------------


def test_exposes_model_name_and_dimension() -> None:
    embedder = FakeEmbedder()
    assert embedder.model_name == "fake-embedder-v1"
    assert embedder.dimension == 256
    custom = FakeEmbedder(dimension=64, model_name="fake-x")
    assert (custom.model_name, custom.dimension) == ("fake-x", 64)


@pytest.mark.parametrize("input_type", ["query", "document"])
def test_every_vector_has_declared_dimension(input_type: InputType) -> None:
    embedder = FakeEmbedder(dimension=64)
    vectors = _embed(embedder, ["a", "b c", "", "!!!"], input_type)
    assert all(len(v) == embedder.dimension for v in vectors)


# --- one vector per text, same order, embed([]) ---------------------------------------


def test_one_vector_per_text_in_input_order() -> None:
    embedder = FakeEmbedder()
    texts = ["first text", "second text", "third"]
    together = _embed(embedder, texts, "document")
    one_by_one = [_embed(embedder, [t], "document")[0] for t in texts]
    assert together == one_by_one


@pytest.mark.parametrize("input_type", ["query", "document"])
def test_empty_input_returns_empty_list(input_type: InputType) -> None:
    assert _embed(FakeEmbedder(), [], input_type) == []


# --- D4: unit length ----------------------------------------------------------------


@pytest.mark.parametrize("text", ["Configure SAML authentication", "x", "", "   !!! "])
@pytest.mark.parametrize("input_type", ["query", "document"])
def test_vectors_are_unit_length_even_for_empty_text(text: str, input_type: InputType) -> None:
    (vector,) = _embed(FakeEmbedder(), [text], input_type)
    assert math.isclose(norm(vector), 1.0)


# --- determinism --------------------------------------------------------------------


def test_is_deterministic_across_instances() -> None:
    text = "How do I configure SAML in SeQuence 9?"
    assert _embed(FakeEmbedder(), [text], "query") == _embed(FakeEmbedder(), [text], "query")


def test_case_insensitive() -> None:
    lower, upper = _embed(FakeEmbedder(), ["saml config", "SAML Config"], "document")
    assert lower == upper


# --- D2: query vs document ------------------------------------------------------------


def test_query_and_document_vectors_differ_but_stay_similar() -> None:
    embedder = FakeEmbedder()
    text = "configure SAML authentication"
    (as_query,) = _embed(embedder, [text], "query")
    (as_document,) = _embed(embedder, [text], "document")
    assert as_query != as_document
    assert cosine_similarity(as_query, as_document) > 0.85


def test_query_finds_related_document_over_unrelated() -> None:
    embedder = FakeEmbedder()
    (query,) = _embed(embedder, ["configure SAML authentication"], "query")
    related, unrelated = _embed(
        embedder,
        [
            "how to configure SAML authentication in SeQuence",
            "release notes for case manager version 11",
        ],
        "document",
    )
    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_invalid_input_type_raises() -> None:
    bad: Any = "passage"  # bypass mypy on purpose: simulates an untyped caller
    with pytest.raises(ValueError, match="input_type"):
        _embed(FakeEmbedder(), ["x"], bad)


# --- constructor validation -----------------------------------------------------------


def test_invalid_constructor_arguments_raise() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dimension=0)
    with pytest.raises(ValueError):
        FakeEmbedder(model_name="")
