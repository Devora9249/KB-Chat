"""Local embedder for BAAI/bge-m3 (dense, 1024 dimensions, multilingual incl. Hebrew).

Implements contracts.storage.Embedder. Needs the optional extra:
    uv sync --extra local-embeddings

Design
- Lazy loading: the ~2.3 GB model loads on the first non-empty `embed` call, not in the
  constructor.
- Not blocking the event loop: encoding is CPU-bound, so it runs in a worker thread via
  `asyncio.to_thread`.
- One encoding at a time: an asyncio.Lock queues concurrent calls (the CPU is saturated
  anyway). A threading.Lock inside the worker thread also guards the model: if a caller is
  cancelled, the asyncio.Lock is released at once while its thread keeps running until the
  encode finishes; the threading.Lock makes the next call wait for that thread.
- input_type is accepted (callers MUST pass it) but IGNORED: per the official model card,
  bge-m3 "no longer requires adding instructions to the queries".
- Output check: exactly one vector per text, each of length 1024 and unit norm
  (normalize_embeddings=True); anything else is a model fault -> EmbeddingError.
- Pinned model version: the model is always loaded at MODEL_REVISION, the official BAAI
  `main` commit, so every machine uses exactly the same weights.
- No background safetensors conversion: BAAI's `main` only has pytorch_model.bin. After
  loading it, `transformers` starts a background thread that downloads a converted
  model.safetensors (~2.1 GB) "for next time", which is never used at a pinned revision.
  `transformers` has no parameter to turn this off, only the environment variable
  DISABLE_SAFETENSORS_CONVERSION (read at load time). The loader sets it to "1" right
  before loading, unless it is already set explicitly; this applies to the whole process,
  which is intended (no background downloads in production).

Errors (contracts.storage rules)
- Wrapped in EmbeddingError with `raise ... from e`, ONLY these infrastructure failures:
  loading: ImportError (extra not installed), OSError (download / Hugging Face Hub
  errors, which all subclass OSError), MemoryError, RuntimeError (torch);
  encoding: RuntimeError (torch, incl. out-of-memory), MemoryError.
- ValueError for invalid input (bad input_type) is raised as is.
- No retries here: loading from the local cache does not need them, and a failed download
  is reported to the caller.
- asyncio.CancelledError is never caught.
"""

import asyncio
import math
import os
import threading
from collections.abc import Callable
from typing import Any, Protocol, cast, get_args

from contracts.storage import EmbeddingError, InputType

MODEL_NAME = "BAAI/bge-m3"
# Pinned commit of BAAI/bge-m3 `main` on the Hugging Face Hub (official BAAI weights).
# Changing it changes the vectors: it REQUIRES a full re-index (coordinate with Chaya).
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION = 1024
# transformers setting that stops the background safetensors download (see module docstring).
DISABLE_CONVERSION_ENV_VAR = "DISABLE_SAFETENSORS_CONVERSION"
_VALID_INPUT_TYPES: tuple[str, ...] = get_args(InputType)
_NORM_TOLERANCE = 1e-3


class SentenceEncoder(Protocol):
    """The part of sentence_transformers.SentenceTransformer that we use."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any: ...


EncoderFactory = Callable[[], SentenceEncoder]


def _load_sentence_transformer(device: str) -> SentenceEncoder:
    """Load bge-m3 at the pinned revision, without the background safetensors download."""
    os.environ.setdefault(DISABLE_CONVERSION_ENV_VAR, "1")  # an explicit setting wins
    from sentence_transformers import SentenceTransformer  # optional extra: import lazily

    return cast(
        SentenceEncoder,
        SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device),
    )


class BgeM3Embedder:
    """contracts.storage.Embedder backed by a local BAAI/bge-m3 model."""

    def __init__(
        self,
        *,
        batch_size: int = 16,
        device: str = "cpu",
        encoder_factory: EncoderFactory | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size
        self._encoder_factory: EncoderFactory = encoder_factory or (
            lambda: _load_sentence_transformer(device)
        )
        self._encoder: SentenceEncoder | None = None
        self._thread_lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None
        self._async_lock_loop: asyncio.AbstractEventLoop | None = None

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        if input_type not in _VALID_INPUT_TYPES:
            raise ValueError(f"input_type must be one of {_VALID_INPUT_TYPES}, got {input_type!r}")
        if not texts:
            return []
        batch = list(texts)  # copy: the caller may change its list while we wait
        async with self._lock_for_running_loop():
            return await asyncio.to_thread(self._embed_blocking, batch)

    # --- internals ------------------------------------------------------------------------

    def _lock_for_running_loop(self) -> asyncio.Lock:
        # An asyncio.Lock belongs to one event loop. The application has one loop; tests may
        # reuse an instance across several asyncio.run() calls, so keep one lock per loop.
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    def _embed_blocking(self, texts: list[str]) -> list[list[float]]:
        """Runs in a worker thread."""
        with self._thread_lock:
            encoder = self._load_if_needed()
            try:
                raw = encoder.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except (RuntimeError, MemoryError) as e:
                raise EmbeddingError(f"{MODEL_NAME}: encoding {len(texts)} texts failed") from e
        return _validated(raw, expected_count=len(texts))

    def _load_if_needed(self) -> SentenceEncoder:
        """Called with _thread_lock held, so the model is loaded at most once."""
        if self._encoder is None:
            try:
                self._encoder = self._encoder_factory()
            except ImportError as e:
                raise EmbeddingError(
                    f"{MODEL_NAME}: sentence-transformers is not installed; "
                    "run `uv sync --extra local-embeddings`"
                ) from e
            except (OSError, MemoryError, RuntimeError) as e:
                raise EmbeddingError(f"{MODEL_NAME}: loading the model failed") from e
        return self._encoder


def _validated(raw: Any, *, expected_count: int) -> list[list[float]]:
    rows = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    if len(rows) != expected_count:
        raise EmbeddingError(f"{MODEL_NAME}: got {len(rows)} vectors for {expected_count} texts")
    vectors: list[list[float]] = []
    for i, row in enumerate(rows):
        vector = [float(x) for x in row]
        if len(vector) != DIMENSION:
            raise EmbeddingError(
                f"{MODEL_NAME}: vector {i} has dimension {len(vector)}, expected {DIMENSION}"
            )
        length = math.sqrt(sum(x * x for x in vector))
        if abs(length - 1.0) > _NORM_TOLERANCE:
            raise EmbeddingError(f"{MODEL_NAME}: vector {i} is not normalized (norm={length:.4f})")
        vectors.append(vector)
    return vectors
