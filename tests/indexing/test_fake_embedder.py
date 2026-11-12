import asyncio
import math

import pytest

from kc.indexing.fake_embedder import FakeEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are already normalized


def _embed(embedder: FakeEmbedder, texts: list[str]) -> list[list[float]]:
    return asyncio.run(embedder.embed(texts))


def test_returns_one_vector_per_text_with_configured_dim() -> None:
    vectors = _embed(FakeEmbedder(dim=64), ["a", "b c", "d"])
    assert len(vectors) == 3
    assert all(len(v) == 64 for v in vectors)


def test_empty_input_returns_empty_list() -> None:
    assert _embed(FakeEmbedder(), []) == []


def test_is_deterministic() -> None:
    text = "How do I configure SAML in SeQuence 9?"
    first = _embed(FakeEmbedder(), [text])
    second = _embed(FakeEmbedder(), [text])
    assert first == second


def test_vectors_are_unit_length() -> None:
    (vector,) = _embed(FakeEmbedder(), ["Configure SAML authentication"])
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)


def test_text_without_tokens_gives_zero_vector() -> None:
    (vector,) = _embed(FakeEmbedder(dim=16), ["   !!! "])
    assert vector == [0.0] * 16


def test_similar_texts_are_closer_than_unrelated_texts() -> None:
    query, related, unrelated = _embed(
        FakeEmbedder(),
        [
            "configure SAML authentication",
            "how to configure SAML authentication in SeQuence",
            "release notes for case manager version 11",
        ],
    )
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_case_insensitive() -> None:
    lower, upper = _embed(FakeEmbedder(), ["saml config", "SAML Config"])
    assert lower == upper


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)
