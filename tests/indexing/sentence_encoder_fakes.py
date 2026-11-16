"""Fakes for the local sentence-transformers embedders (bge-m3, multilingual-e5).

They stand in for the real model, so these tests are fast, need no download and run in CI.
"""

import hashlib
import os
import threading
import time
from typing import Any, ClassVar

from kc.indexing.sentence_transformer_embedder import DISABLE_CONVERSION_ENV_VAR

DIMENSION = 1024


def unit_vector(text: str) -> list[float]:
    """Deterministic one-hot vector per text (unit length, dimension 1024)."""
    vector = [0.0] * DIMENSION
    vector[int(hashlib.sha256(text.encode()).hexdigest(), 16) % DIMENSION] = 1.0
    return vector


class FakeEncoder:
    """Stands in for SentenceTransformer: records calls and concurrency, can misbehave."""

    def __init__(
        self,
        *,
        raise_on_encode: BaseException | None = None,
        rows: list[list[float]] | None = None,
        block: threading.Event | None = None,
    ) -> None:
        self.raise_on_encode = raise_on_encode
        self.rows = rows
        self.block = block
        self.started = threading.Event()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            self.calls.append((list(sentences), kwargs))
            self.started.set()
            if self.block is not None:
                self.block.wait(timeout=5)
            else:
                time.sleep(0.005)  # long enough for concurrent calls to overlap if unguarded
            if self.raise_on_encode is not None:
                raise self.raise_on_encode
            return self.rows if self.rows is not None else [unit_vector(s) for s in sentences]
        finally:
            with self._guard:
                self._active -= 1


class CountingFactory:
    def __init__(self, encoder: FakeEncoder | None = None, error: BaseException | None = None):
        self.encoder = encoder or FakeEncoder()
        self.error = error
        self.calls = 0

    def __call__(self) -> FakeEncoder:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.encoder


class RecordingSentenceTransformer:
    """Replaces sentence_transformers.SentenceTransformer; records how it was built."""

    instances: ClassVar[list["RecordingSentenceTransformer"]] = []

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        # What transformers would see while loading:
        self.conversion_env = os.environ.get(DISABLE_CONVERSION_ENV_VAR)
        RecordingSentenceTransformer.instances.append(self)
