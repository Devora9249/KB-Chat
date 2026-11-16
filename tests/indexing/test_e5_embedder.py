"""Tests for E5Embedder (intfloat/multilingual-e5-large).

Most tests inject a fake encoder (fast, no download, run in CI). Tests marked real_model
load the real model (~2.27 GB on first run) and run only with KC_REAL_EMBEDDER=1.
The shared machinery (locks, error wrapping) is tested in depth in test_bge_m3_embedder.py;
here the e5 specifics plus the key shared guarantees for this class.
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
    unit_vector,
)

from contracts.storage import Embedder, EmbeddingError, InputType
from kc.indexing.e5_embedder import (
    DIMENSION,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REVISION,
    E5Embedder,
    _load_sentence_transformer,
)
from kc.indexing.sentence_transformer_embedder import DISABLE_CONVERSION_ENV_VAR
from kc.indexing.vector_math import cosine_similarity, norm


def _embed(embedder: E5Embedder, texts: list[str], kind: InputType = "document") -> Any:
    return asyncio.run(embedder.embed(texts, input_type=kind))


# --- contract basics ------------------------------------------------------------------------


def test_satisfies_embedder_protocol_and_identity() -> None:
    embedder = E5Embedder(encoder_factory=CountingFactory())
    assert isinstance(embedder, Embedder)
    assert embedder.model_name == "intfloat/multilingual-e5-large@3d7cfbd" == MODEL_ID
    assert embedder.dimension == DIMENSION == 1024  # same column as bge-m3
    assert MAX_TOKENS == 512


def test_model_name_identifies_model_and_pinned_revision() -> None:
    hub_id, _, short_revision = E5Embedder(encoder_factory=CountingFactory()).model_name.partition(
        "@"
    )
    assert hub_id == MODEL_NAME
    assert MODEL_REVISION.startswith(short_revision)
    assert len(short_revision) == 7


def test_constructor_does_not_load_and_model_loads_once() -> None:
    factory = CountingFactory()
    embedder = E5Embedder(encoder_factory=factory)
    assert _embed(embedder, []) == []
    assert factory.calls == 0
    _embed(embedder, ["a"])
    _embed(embedder, ["b"], "query")
    assert factory.calls == 1
    assert embedder.is_loaded


# --- prefixes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_type", "prefix"), [("query", "query: "), ("document", "passage: ")]
)
def test_prefix_is_added_from_input_type(input_type: InputType, prefix: str) -> None:
    factory = CountingFactory()
    embedder = E5Embedder(encoder_factory=factory, batch_size=5)
    texts = ["How do I configure SAML?", "איך מגדירים SAML?"]

    vectors = _embed(embedder, texts, input_type)

    (sent, kwargs) = factory.encoder.calls[0]
    assert sent == [prefix + t for t in texts]  # exactly "query: " / "passage: ", once
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 5
    assert vectors == [unit_vector(prefix + t) for t in texts]


def test_query_and_document_of_the_same_text_differ() -> None:
    embedder = E5Embedder(encoder_factory=CountingFactory())
    assert _embed(embedder, ["x"], "query") != _embed(embedder, ["x"], "document")


def test_callers_list_is_not_modified() -> None:
    texts = ["a", "b"]
    _embed(E5Embedder(encoder_factory=CountingFactory()), texts, "query")
    assert texts == ["a", "b"]


def test_invalid_input_type_raises_value_error_before_loading() -> None:
    factory = CountingFactory()
    bad: Any = "passage"
    with pytest.raises(ValueError, match="input_type"):
        _embed(E5Embedder(encoder_factory=factory), ["x"], bad)
    assert factory.calls == 0


def test_invalid_batch_size_raises() -> None:
    with pytest.raises(ValueError):
        E5Embedder(batch_size=0)


# --- loader --------------------------------------------------------------------------------


def test_loader_uses_pinned_revision(
    fake_sentence_transformers: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DISABLE_CONVERSION_ENV_VAR, raising=False)
    _load_sentence_transformer("cpu")
    (loaded,) = RecordingSentenceTransformer.instances
    assert loaded.model_name_or_path == "intfloat/multilingual-e5-large"
    assert loaded.kwargs == {"revision": MODEL_REVISION, "device": "cpu"}
    assert MODEL_REVISION == "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"


def test_default_embedder_loads_through_the_pinned_loader(
    fake_sentence_transformers: type,
) -> None:
    with pytest.raises(AttributeError):  # the recording fake has no encode(); load happened
        _embed(E5Embedder(device="cpu"), ["a"])
    (loaded,) = RecordingSentenceTransformer.instances
    assert loaded.kwargs["revision"] == MODEL_REVISION


# --- output validation and errors ----------------------------------------------------------


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
    embedder = E5Embedder(encoder_factory=CountingFactory(FakeEncoder(rows=rows)))
    with pytest.raises(EmbeddingError, match=message):
        _embed(embedder, ["a", "b"])


@pytest.mark.parametrize(
    "error",
    [
        OSError("hub unreachable"),
        TimeoutError("download timed out"),
        MemoryError(),
        RuntimeError("torch load"),
        ImportError("no st"),
    ],
    ids=["OSError", "TimeoutError", "MemoryError", "RuntimeError", "ImportError"],
)
def test_loading_failures_are_wrapped_with_cause(error: BaseException) -> None:
    with pytest.raises(EmbeddingError, match=MODEL_NAME) as exc_info:
        _embed(E5Embedder(encoder_factory=CountingFactory(error=error)), ["a"])
    assert exc_info.value.__cause__ is error


def test_http_client_timeout_is_wrapped_but_cancellation_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_httpx2 = types.ModuleType("httpx2")

    class TransportError(Exception):
        pass

    class ReadTimeout(TransportError):
        pass

    fake_httpx2.TransportError = TransportError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx2", fake_httpx2)
    timeout = ReadTimeout("read timed out")
    with pytest.raises(EmbeddingError) as exc_info:
        _embed(E5Embedder(encoder_factory=CountingFactory(error=timeout)), ["a"])
    assert exc_info.value.__cause__ is timeout

    cancelled = E5Embedder(encoder_factory=CountingFactory(error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        _embed(cancelled, ["a"])


@pytest.mark.parametrize(
    "error", [RuntimeError("torch failed"), MemoryError()], ids=["RuntimeError", "MemoryError"]
)
def test_encoding_failures_are_wrapped_with_cause(error: BaseException) -> None:
    embedder = E5Embedder(encoder_factory=CountingFactory(FakeEncoder(raise_on_encode=error)))
    with pytest.raises(EmbeddingError) as exc_info:
        _embed(embedder, ["a"])
    assert exc_info.value.__cause__ is error


def test_unexpected_errors_are_not_wrapped() -> None:
    embedder = E5Embedder(
        encoder_factory=CountingFactory(FakeEncoder(raise_on_encode=TypeError("bug")))
    )
    with pytest.raises(TypeError, match="bug"):
        _embed(embedder, ["a"])


# --- concurrency and cancellation -----------------------------------------------------------


def test_concurrent_calls_never_run_the_model_in_parallel() -> None:
    encoder = FakeEncoder()
    embedder = E5Embedder(encoder_factory=CountingFactory(encoder))
    calls = [[f"call {c} text {t}" for t in range(1 + c % 3)] for c in range(10)]

    async def scenario() -> list[list[list[float]]]:
        return list(await asyncio.gather(*(embedder.embed(t, input_type="query") for t in calls)))

    results = asyncio.run(scenario())

    assert encoder.max_active == 1
    for texts, vectors in zip(calls, results, strict=True):
        assert vectors == [unit_vector("query: " + t) for t in texts]


def test_caller_timeout_surfaces_as_timeout_error_not_embedding_error() -> None:
    release = threading.Event()
    embedder = E5Embedder(encoder_factory=CountingFactory(FakeEncoder(block=release)))

    async def scenario() -> None:
        try:
            async with asyncio.timeout(0.05):
                await embedder.embed(["slow"], input_type="query")
        finally:
            release.set()  # let the worker thread finish

    with pytest.raises(TimeoutError) as exc_info:
        asyncio.run(scenario())
    assert not isinstance(exc_info.value, EmbeddingError)


# --- real model (KC_REAL_EMBEDDER=1) ----------------------------------------------------------


@pytest.mark.real_model
def test_real_model_vectors_have_dimension_1024_and_unit_length(
    e5_embedder: E5Embedder,
) -> None:
    vectors = _embed(e5_embedder, ["How do I configure SAML?", "איך מגדירים SAML?", ""])
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == DIMENSION
        assert abs(norm(vector) - 1.0) < 1e-3
    assert e5_embedder.model_name == MODEL_ID


@pytest.mark.real_model
def test_real_model_hebrew_question_finds_the_english_document(
    e5_embedder: E5Embedder,
) -> None:
    documents = [
        "Configuring SAML authentication for Cora SeQuence: set up the identity provider.",
        "Release notes for Cora Case Manager version 11: new features and fixed issues.",
        "Installing Cora Orchestration on Windows Server: prerequisites and setup steps.",
    ]
    doc_vectors = _embed(e5_embedder, documents, "document")
    (query,) = _embed(e5_embedder, ["איך מגדירים אימות SAML ב-SeQuence?"], "query")

    scores = [cosine_similarity(query, d) for d in doc_vectors]

    assert scores.index(max(scores)) == 0, f"scores: {scores}"


@pytest.mark.real_model
def test_real_model_uses_the_prefix(e5_embedder: E5Embedder) -> None:
    # Unlike bge-m3, the same text as query and as passage gives DIFFERENT vectors.
    (as_query,) = _embed(e5_embedder, ["configure SAML"], "query")
    (as_document,) = _embed(e5_embedder, ["configure SAML"], "document")
    assert cosine_similarity(as_query, as_document) < 0.9999
