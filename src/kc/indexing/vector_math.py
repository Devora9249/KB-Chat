"""Pure-Python vector helpers (no numpy): normalization, cosine similarity, top-k.

Used by in-memory search and tests. pgvector does the same math inside Postgres
for the real index.
"""

import heapq
import math
from collections.abc import Sequence


def _check_same_dim(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def normalize(vector: Sequence[float]) -> list[float]:
    """Scale to unit length. A zero vector is returned unchanged (as zeros)."""
    length = norm(vector)
    if length == 0.0:
        return [0.0] * len(vector)
    return [x / length for x in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between a and b, in [-1, 1]. 0.0 if either is a zero vector."""
    _check_same_dim(a, b)
    denom = norm(a) * norm(b)
    if denom == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    # Clamp tiny floating-point overshoot (e.g. 1.0000000000000002).
    return max(-1.0, min(1.0, dot / denom))


def top_k(
    query: Sequence[float], vectors: Sequence[Sequence[float]], k: int
) -> list[tuple[int, float]]:
    """Return up to k (index, score) pairs, best cosine score first.

    Ties keep the original order (lower index first), so results are deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0 or not vectors:
        return []
    scored = ((i, cosine_similarity(query, v)) for i, v in enumerate(vectors))
    return heapq.nsmallest(k, scored, key=lambda item: (-item[1], item[0]))
