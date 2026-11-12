"""Split work into fixed-size batches while preserving order.

Embedding APIs limit how many texts fit in one request; batching keeps us under
that limit and guarantees vector i still belongs to text i.
"""

from collections.abc import Awaitable, Callable, Sequence
from itertools import batched as _batched

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def batched[T](items: Sequence[T], size: int) -> list[list[T]]:
    """[1,2,3,4,5], 2 -> [[1,2],[3,4],[5]]. The last batch may be smaller."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [list(batch) for batch in _batched(items, size)]


async def embed_in_batches(embed: EmbedFn, texts: list[str], batch_size: int) -> list[list[float]]:
    """Call `embed` once per batch (sequentially) and concatenate results in order.

    `embed` is any async callable like `Embedder.embed`, passed in as a parameter
    so this helper does not depend on a concrete embedder.
    """
    vectors: list[list[float]] = []
    for batch in batched(texts, batch_size):
        result = await embed(batch)
        if len(result) != len(batch):
            raise ValueError(
                f"embedder returned {len(result)} vectors for a batch of {len(batch)} texts"
            )
        vectors.extend(result)
    return vectors
