"""Deterministic, offline Embedder for tests and local development.

Uses the "hashing trick": each lowercase word token is hashed to a fixed slot in the
vector (with a +/- sign), counts are accumulated, and the vector is L2-normalized.
Texts that share words therefore get a higher cosine similarity, so search results
in tests are meaningful, not random. No network, no API key, no cost.

Satisfies the `Embedder` protocol from `contracts/` structurally:
    async def embed(self, texts: list[str]) -> list[list[float]]
"""

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    """Bag-of-words hashing embedder. Same text -> same vector, across runs and machines."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            # hashlib (not built-in hash()) so results are stable across processes.
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector  # empty / token-less text -> zero vector
        return [x / norm for x in vector]
