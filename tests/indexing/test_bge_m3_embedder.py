"""Tests for BgeM3Embedder.

Most tests inject a fake encoder (fast, no download, run in CI). Tests marked real_model
load the real BAAI/bge-m3 and run only with KC_REAL_EMBEDDER=1 (see conftest.py).
"""

import asyncio
import sys
import threading
import types
from typing import Any

import pytest
from sentence_encoder_fakes import (
    CountingFactory,
    FakeEncoder,
    RecordingSentenceTransformer,
)
from sentence_encoder_fakes import unit_vector as _unit_vector

from contracts.storage import Embedder, EmbeddingError, InputType
from kc.indexing.bge_m3_embedder import (
    DIMENSION,
    DISABLE_CONVERSION_ENV_VAR,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REVISION,
    BgeM3Embedder,
    _load_sentence_transformer,
)
from kc.indexing.vector_math import cosine_similarity, norm


def _embed(embedder: BgeM3Embedder, texts: list[str], kind: InputType = "document") -> Any:
    return asyncio.run(embedder.embed(texts, input_type=kind))


# --- contract basics ------------------------------------------------------------------------


def test_satisfies_embedder_protocol_and_identity() -> None:
    embedder = BgeM3Embedder(encoder_factory=CountingFactory())
    assert isinstance(embedder, Embedder)
    assert (embedder.model_name, embedder.dimension) == ("BAAI/bge-m3@5617a9f", 1024)


def test_model_name_identifies_model_and_pinned_revision() -> None:
    # Format "<hub id>@<first 7 chars of the pinned revision>": a revision change must change
    # model_name, so the embedding_config guard demands a re-index.
    model_name = BgeM3Embedder(encoder_factory=CountingFactory()).model_name
    hub_id, _, short_revision = model_name.partition("@")
    assert hub_id == MODEL_NAME == "BAAI/bge-m3"
    assert short_revision == MODEL_REVISION[:7]
    assert len(short_revision) == 7
    assert MODEL_REVISION.startswith(short_revision)
    assert model_name == MODEL_ID


def test_constructor_does_not_load_and_empty_input_does_not_load() -> None:
    factory = CountingFactory()
    embedder = BgeM3Embedder(encoder_factory=factory)
    assert _embed(embedder, []) == []
    assert factory.calls == 0
    assert not embedder.is_loaded


def test_model_is_loaded_once_and_lazily() -> None:
    factory = CountingFactory()
    embedder = BgeM3Embedder(encoder_factory=factory)
    _embed(embedder, ["a"])
    _embed(embedder, ["b", "c"])
    assert factory.calls == 1
    assert embedder.is_loaded


def test_returns_one_unit_vector_per_text_in_order_and_asks_for_normalization() -> None:
    factory = CountingFactory()
    texts = ["first", "second", "third"]
    vectors = _embed(BgeM3Embedder(encoder_factory=factory, batch_size=7), texts)
    assert vectors == [_unit_vector(t) for t in texts]
    _, kwargs = factory.encoder.calls[0]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 7


def test_input_type_is_ignored_but_validated() -> None:
    embedder = BgeM3Embedder(encoder_factory=CountingFactory())
    assert _embed(embedder, ["x"], "query") == _embed(embedder, ["x"], "document")
    bad: Any = "passage"
    with pytest.raises(ValueError, match="input_type"):
        _embed(embedder, ["x"], bad)


def test_invalid_batch_size_raises() -> None:
    with pytest.raises(ValueError):
        BgeM3Embedder(batch_size=0)


# --- loader: pinned revision, no background safetensors conversion ---------------------------


_RecordingSentenceTransformer = RecordingSentenceTransformer  # fixture in conftest.py


def test_loader_uses_pinned_revision_and_disables_background_conversion(
    fake_sentence_transformers: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_CONVERSION_ENV_VAR, raising=False)

    _load_sentence_transformer("cpu")

    (loaded,) = _RecordingSentenceTransformer.instances
    assert loaded.model_name_or_path == "BAAI/bge-m3"
    assert loaded.kwargs == {"revision": MODEL_REVISION, "device": "cpu"}
    assert MODEL_REVISION == "5617a9f61b028005a4858fdac845db406aefb181"
    assert loaded.conversion_env == "1"  # set BEFORE transformers loads the weights


def test_loader_respects_an_explicit_conversion_setting(
    fake_sentence_transformers: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DISABLE_CONVERSION_ENV_VAR, "0")
    _load_sentence_transformer("cpu")
    assert _RecordingSentenceTransformer.instances[0].conversion_env == "0"


def test_default_embedder_loads_through_the_pinned_loader(
    fake_sentence_transformers: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_CONVERSION_ENV_VAR, raising=False)
    embedder = BgeM3Embedder(device="cpu")  # no encoder_factory: the real loader path
    with pytest.raises(AttributeError):  # the recording fake has no encode(); load happened
        _embed(embedder, ["a"])
    (loaded,) = _RecordingSentenceTransformer.instances
    assert loaded.kwargs["revision"] == MODEL_REVISION


# --- output validation (model faults -> EmbeddingError) -------------------------------------


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[1.0] + [0.0] * (DIMENSION - 1)], "got 1 vectors for 2 texts"),
        ([[1.0, 0.0], [0.0, 1.0]], "dimension 2, expected 1024"),
        ([[2.0] + [0.0] * (DIMENSION - 1)] * 2, "not normalized"),
    ],
    ids=["wrong-count", "wrong-dimension", "not-normalized"],
)
def test_bad_model_output_raises_embedding_error(rows: list[list[float]], message: str) -> None:
    embedder = BgeM3Embedder(encoder_factory=CountingFactory(FakeEncoder(rows=rows)))
    with pytest.raises(EmbeddingError, match=message):
        _embed(embedder, ["a", "b"])


# --- error wrapping: only specific infrastructure errors ------------------------------------


@pytest.mark.parametrize(
    "error",
    [OSError("hub unreachable"), MemoryError(), RuntimeError("torch load"), ImportError("no st")],
    ids=["OSError", "MemoryError", "RuntimeError", "ImportError"],
)
def test_loading_failures_are_wrapped_with_cause(error: BaseException) -> None:
    embedder = BgeM3Embedder(encoder_factory=CountingFactory(error=error))
    with pytest.raises(EmbeddingError) as exc_info:
        _embed(embedder, ["a"])
    assert exc_info.value.__cause__ is error


# --- timeouts --------------------------------------------------------------------------------


def test_load_timeouts_are_wrapped_but_cancellation_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Python's own TimeoutError (a subclass of OSError).
    timeout = TimeoutError("model download timed out")
    with pytest.raises(EmbeddingError) as exc_info:
        _embed(BgeM3Embedder(encoder_factory=CountingFactory(error=timeout)), ["a"])
    assert exc_info.value.__cause__ is timeout

    # 2. The HTTP client's timeout (NOT an OSError), as huggingface_hub re-raises it after its
    #    retries. A fake httpx2 module keeps this test runnable where the extra is missing (CI).
    fake_httpx2 = types.ModuleType("httpx2")

    class TransportError(Exception):
        pass

    class ConnectTimeout(TransportError):
        pass

    fake_httpx2.TransportError = TransportError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx2", fake_httpx2)
    client_timeout = ConnectTimeout("connect timed out")
    assert not isinstance(client_timeout, OSError)
    with pytest.raises(EmbeddingError) as exc_info:
        _embed(BgeM3Embedder(encoder_factory=CountingFactory(error=client_timeout)), ["a"])
    assert exc_info.value.__cause__ is client_timeout

    # 3. Cancellation is never wrapped.
    cancelled = BgeM3Embedder(encoder_factory=CountingFactory(error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        _embed(cancelled, ["a"])


def test_real_httpx2_timeouts_are_wrapped() -> None:
    httpx2 = pytest.importorskip("httpx2", reason="needs the local-embeddings extra")
    for error in (httpx2.ConnectTimeout("connect"), httpx2.ReadTimeout("read")):
        with pytest.raises(EmbeddingError) as exc_info:
            _embed(BgeM3Embedder(encoder_factory=CountingFactory(error=error)), ["a"])
        assert exc_info.value.__cause__ is error


def test_caller_timeout_surfaces_as_timeout_error_not_embedding_error() -> None:
    # A timeout the CALLER sets cancels embed() with CancelledError (never wrapped), so the
    # caller sees TimeoutError from its own `asyncio.timeout`, not EmbeddingError.
    release = threading.Event()
    embedder = BgeM3Embedder(encoder_factory=CountingFactory(FakeEncoder(block=release)))

    async def scenario() -> None:
        try:
            async with asyncio.timeout(0.05):
                await embedder.embed(["slow"], input_type="query")
        finally:
            release.set()  # let the worker thread finish

    with pytest.raises(TimeoutError) as exc_info:
        asyncio.run(scenario())
    assert not isinstance(exc_info.value, EmbeddingError)


@pytest.mark.parametrize(
    "error", [RuntimeError("torch failed"), MemoryError()], ids=["RuntimeError", "MemoryError"]
)
def test_encoding_failures_are_wrapped_with_cause(error: BaseException) -> None:
    embedder = BgeM3Embedder(encoder_factory=CountingFactory(FakeEncoder(raise_on_encode=error)))
    with pytest.raises(EmbeddingError) as exc_info:
        _embed(embedder, ["a"])
    assert exc_info.value.__cause__ is error


def test_unexpected_errors_are_not_wrapped() -> None:
    # No generic `except Exception`: a programming error must surface as itself.
    embedder = BgeM3Embedder(
        encoder_factory=CountingFactory(FakeEncoder(raise_on_encode=TypeError("bug")))
    )
    with pytest.raises(TypeError, match="bug"):
        _embed(embedder, ["a"])


def test_failed_load_can_be_retried_by_the_caller() -> None:
    factory = CountingFactory(error=OSError("offline"))
    embedder = BgeM3Embedder(encoder_factory=factory)
    with pytest.raises(EmbeddingError):
        _embed(embedder, ["a"])
    factory.error = None  # network is back
    assert len(_embed(embedder, ["a"])) == 1
    assert factory.calls == 2


# --- concurrency and cancellation -----------------------------------------------------------


def test_concurrent_calls_never_run_the_model_in_parallel() -> None:
    encoder = FakeEncoder()
    embedder = BgeM3Embedder(encoder_factory=CountingFactory(encoder))
    calls = [[f"call {c} text {t}" for t in range(1 + c % 4)] for c in range(12)]

    async def scenario() -> list[list[list[float]]]:
        return list(await asyncio.gather(*(embedder.embed(t, input_type="query") for t in calls)))

    results = asyncio.run(scenario())

    assert encoder.max_active == 1
    for texts, vectors in zip(calls, results, strict=True):
        assert vectors == [_unit_vector(t) for t in texts]


def test_cancellation_is_not_wrapped_and_next_call_waits_for_the_running_thread() -> None:
    release = threading.Event()
    encoder = FakeEncoder(block=release)
    embedder = BgeM3Embedder(encoder_factory=CountingFactory(encoder))

    async def scenario() -> list[list[float]]:
        first = asyncio.create_task(embedder.embed(["slow"], input_type="query"))
        assert await asyncio.to_thread(encoder.started.wait, 5)  # encode is running now
        first.cancel()
        with pytest.raises(asyncio.CancelledError):  # NOT EmbeddingError
            await first
        encoder.block = None  # next encode is fast
        second = asyncio.create_task(embedder.embed(["next"], input_type="query"))
        await asyncio.sleep(0.02)
        assert not second.done()  # still waiting for the cancelled call's thread
        release.set()
        return await second

    assert asyncio.run(scenario()) == [_unit_vector("next")]
    assert encoder.max_active == 1


# --- real model (KC_REAL_EMBEDDER=1) ----------------------------------------------------------


@pytest.mark.real_model
def test_real_model_vectors_have_dimension_1024_and_unit_length(
    bge_m3_embedder: BgeM3Embedder,
) -> None:
    vectors = _embed(bge_m3_embedder, ["How do I configure SAML?", "איך מגדירים SAML?", ""])
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == DIMENSION
        assert abs(norm(vector) - 1.0) < 1e-3
    assert bge_m3_embedder.model_name == MODEL_ID


@pytest.mark.real_model
def test_real_model_hebrew_question_finds_the_english_document(
    bge_m3_embedder: BgeM3Embedder,
) -> None:
    documents = [
        "Configuring SAML authentication for Cora SeQuence: set up the identity provider.",
        "Release notes for Cora Case Manager version 11: new features and fixed issues.",
        "Installing Cora Orchestration on Windows Server: prerequisites and setup steps.",
    ]
    doc_vectors = _embed(bge_m3_embedder, documents, "document")
    (query,) = _embed(bge_m3_embedder, ["איך מגדירים אימות SAML ב-SeQuence?"], "query")

    scores = [cosine_similarity(query, d) for d in doc_vectors]

    assert scores.index(max(scores)) == 0, f"scores: {scores}"


@pytest.mark.real_model
def test_real_model_ignores_input_type(bge_m3_embedder: BgeM3Embedder) -> None:
    (as_query,) = _embed(bge_m3_embedder, ["configure SAML"], "query")
    (as_document,) = _embed(bge_m3_embedder, ["configure SAML"], "document")
    assert cosine_similarity(as_query, as_document) > 0.9999
