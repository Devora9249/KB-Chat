import asyncio

import pytest

from kc.indexing.batching import batched, embed_in_batches
from kc.indexing.fake_embedder import FakeEmbedder


def test_batched_splits_and_keeps_order() -> None:
    assert batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert batched([1, 2], 5) == [[1, 2]]
    assert batched([], 3) == []


def test_batched_invalid_size_raises() -> None:
    with pytest.raises(ValueError):
        batched([1], 0)


def test_embed_in_batches_matches_single_call_and_counts_calls() -> None:
    embedder = FakeEmbedder(dimension=32)
    texts = [f"text number {i}" for i in range(7)]
    calls: list[int] = []

    async def counting_embed(batch: list[str]) -> list[list[float]]:
        calls.append(len(batch))
        return await embedder.embed(batch, input_type="document")

    batched_vectors = asyncio.run(embed_in_batches(counting_embed, texts, batch_size=3))
    single_call = asyncio.run(embedder.embed(texts, input_type="document"))

    assert batched_vectors == single_call
    assert calls == [3, 3, 1]


def test_embed_in_batches_detects_wrong_vector_count() -> None:
    async def broken_embed(batch: list[str]) -> list[list[float]]:
        return [[0.0]]  # always one vector, regardless of batch size

    with pytest.raises(ValueError, match="returned 1 vectors for a batch of 2"):
        asyncio.run(embed_in_batches(broken_embed, ["a", "b"], batch_size=2))
