"""Deterministic, offline Embedder for tests and local development.

How a vector is built (the "hashing trick"):
1. Split the text into lowercase word tokens.
2. Hash each token (blake2b, stable across processes) to a slot in the vector and a sign;
   add +1 or -1 there. Texts sharing words push the same slots, so they end up close.
3. Add one hidden marker token for `input_type` with a small weight (0.5), so the same
   text gives different vectors as a query and as a document, while staying very similar
   (a query still finds its matching document).
4. L2-normalize to unit length.

The marker also guarantees a non-zero vector: word counts are whole numbers and the
marker adds +/-0.5, so no slot sum can cancel to zero. Every output is unit length,
even for empty or punctuation-only text.

Satisfies the draft `Embedder` protocol (storage contract D1-D5).
"""

import hashlib
import re
from typing import get_args

from kc.indexing._provisional_contracts import InputType
from kc.indexing.vector_math import normalize

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MARKER_WEIGHT = 0.5
# "\x00" can never appear inside a \w+ token, so markers never collide with real words
# by name (they may still share a slot via hashing, which is harmless).
_MARKER_PREFIX = "\x00input_type="
_VALID_INPUT_TYPES: tuple[str, ...] = get_args(InputType)


class FakeEmbedder:
    """Bag-of-words hashing embedder. Same (text, input_type) -> same vector, always."""

    def __init__(self, dimension: int = 256, model_name: str = "fake-embedder-v1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not model_name:
            raise ValueError("model_name must not be empty")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        # mypy already rejects bad literals; this catches untyped callers at runtime.
        if input_type not in _VALID_INPUT_TYPES:
            raise ValueError(f"input_type must be one of {_VALID_INPUT_TYPES}, got {input_type!r}")
        return [self._embed_one(text, input_type) for text in texts]

    def _embed_one(self, text: str, input_type: InputType) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            self._add(vector, token, 1.0)
        self._add(vector, _MARKER_PREFIX + input_type, _MARKER_WEIGHT)
        return normalize(vector)

    def _add(self, vector: list[float], token: str, weight: float) -> None:
        # hashlib (not built-in hash()) so results are stable across processes and machines.
        value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest())
        index = value % self._dimension
        sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
        vector[index] += sign * weight
