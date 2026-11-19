"""Tests for the search settings (RetrievalSettings) and the HybridRetriever factory."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from contracts.retrieval import Filters, Retriever, SearchResult
from contracts.storage import InputType
from kc.retrieval.fake_reranker import FakeReranker
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RELEASE_NOTES_WEIGHT,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_TIMEOUT_S,
)
from kc.retrieval.settings import (
    DEFAULT_RERANK_MODEL,
    RetrievalSettings,
    create_hybrid_retriever,
    create_reranker,
    load_retrieval_settings,
    open_reranker,
)
from kc.retrieval.voyage_reranker import VoyageReranker

ENV_NAMES = (
    "RETRIEVAL_CANDIDATE_K",
    "RETRIEVAL_TIMEOUT_S",
    "RETRIEVAL_RELEASE_NOTES_WEIGHT",
    "RETRIEVAL_RERANK_CANDIDATES",
    "RETRIEVAL_RERANK_TIMEOUT_S",
    "RETRIEVAL_RERANKER",
    "RETRIEVAL_RERANK_MODEL",
    "VOYAGE_API_KEY",
)
FAKE_KEY = "test-key-not-real"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Building / opening / closing a VoyageReranker must never touch the network.
    async def fail(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"real network call to {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)


def _load() -> RetrievalSettings:
    # _env_file=None: ignore any developer's local .env so tests are reproducible.
    # mypy does not know pydantic-settings' init arguments (same ignore as tests/db).
    return RetrievalSettings(_env_file=None)  # type: ignore[call-arg]


# --- defaults and loading ----------------------------------------------------------------


def test_defaults_come_from_hybrid_retriever() -> None:
    settings = _load()
    assert settings.candidate_k == DEFAULT_CANDIDATE_K == 50
    assert settings.timeout_s == DEFAULT_TIMEOUT_S == 10.0
    assert settings.release_notes_weight == DEFAULT_RELEASE_NOTES_WEIGHT == 1.0
    assert settings.rerank_candidates == DEFAULT_RERANK_CANDIDATES == 20
    assert settings.rerank_timeout_s == DEFAULT_RERANK_TIMEOUT_S == 3.0
    assert settings.reranker == "none"
    assert settings.rerank_model == DEFAULT_RERANK_MODEL == "rerank-3-lite"
    assert settings.voyage_api_key is None


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "20")
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", "2.5")
    settings = _load()
    assert settings.candidate_k == 20
    assert settings.timeout_s == 2.5


def test_whole_number_timeout_is_read_as_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", "5")
    settings = _load()
    assert settings.timeout_s == 5.0
    assert isinstance(settings.timeout_s, float)


def test_unrelated_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_SOMETHING_ELSE", "x")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://kc:kc@localhost/kc")
    assert _load().candidate_k == DEFAULT_CANDIDATE_K


def test_load_retrieval_settings_reads_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Run from an empty folder, so load_retrieval_settings() finds no .env file.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "7")
    settings = load_retrieval_settings()
    assert isinstance(settings, RetrievalSettings)
    assert settings.candidate_k == 7
    assert settings.timeout_s == DEFAULT_TIMEOUT_S


def test_load_retrieval_settings_reads_dot_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("RETRIEVAL_TIMEOUT_S=3.5\n", encoding="utf-8")
    assert load_retrieval_settings().timeout_s == 3.5


# --- invalid values ----------------------------------------------------------------------


@pytest.mark.parametrize("value", ["abc", "5.0", "5.5", "true", ""])
def test_candidate_k_not_an_int_in_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_CANDIDATE_K"):
        _load()


@pytest.mark.parametrize("value", [True, False, 5.0, 5.5])
def test_candidate_k_bool_or_float_is_rejected(value: Any) -> None:
    # In lax mode pydantic would turn True into 1 and 5.0 into 5; we reject both.
    with pytest.raises(ValidationError, match="candidate_k must be an int"):
        RetrievalSettings(_env_file=None, RETRIEVAL_CANDIDATE_K=value)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_candidate_k_below_one_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_CANDIDATE_K"):
        _load()


@pytest.mark.parametrize("value", ["0", "0.0", "-1", "-0.5"])
def test_timeout_s_not_positive_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_TIMEOUT_S"):
        _load()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_timeout_s_nan_and_inf_are_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_TIMEOUT_S"):
        _load()


@pytest.mark.parametrize("value", ["abc", "true", ""])
def test_timeout_s_not_a_number_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_TIMEOUT_S"):
        _load()


@pytest.mark.parametrize("value", [True, False])
def test_timeout_s_bool_is_rejected(value: bool) -> None:
    with pytest.raises(ValidationError, match="timeout_s must be a number"):
        RetrievalSettings(_env_file=None, RETRIEVAL_TIMEOUT_S=value)  # type: ignore[call-arg]


@pytest.mark.parametrize(("value", "expected"), [("0.5", 0.5), ("1", 1.0), ("0.001", 0.001)])
def test_release_notes_weight_reads_from_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: float
) -> None:
    monkeypatch.setenv("RETRIEVAL_RELEASE_NOTES_WEIGHT", value)
    assert _load().release_notes_weight == expected


@pytest.mark.parametrize(
    "value", ["0", "0.0", "-0.1", "1.01", "2", "nan", "inf", "-inf", "abc", "true", ""]
)
def test_release_notes_weight_invalid_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_RELEASE_NOTES_WEIGHT", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_RELEASE_NOTES_WEIGHT"):
        _load()


@pytest.mark.parametrize("value", [True, False])
def test_release_notes_weight_bool_is_rejected(value: bool) -> None:
    with pytest.raises(ValidationError, match="release_notes_weight must be a number"):
        RetrievalSettings(_env_file=None, RETRIEVAL_RELEASE_NOTES_WEIGHT=value)  # type: ignore[call-arg]


def test_rerank_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_CANDIDATES", "30")
    monkeypatch.setenv("RETRIEVAL_RERANK_TIMEOUT_S", "1")
    settings = _load()
    assert settings.rerank_candidates == 30
    assert settings.rerank_timeout_s == 1.0
    assert isinstance(settings.rerank_timeout_s, float)


@pytest.mark.parametrize("value", ["abc", "5.0", "5.5", "true", "", "0", "-1"])
def test_rerank_candidates_invalid_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_CANDIDATES", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_RERANK_CANDIDATES"):
        _load()


@pytest.mark.parametrize("value", [True, False, 5.0, 5.5])
def test_rerank_candidates_bool_or_float_is_rejected(value: Any) -> None:
    with pytest.raises(ValidationError, match="rerank_candidates must be an int"):
        RetrievalSettings(_env_file=None, RETRIEVAL_RERANK_CANDIDATES=value)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", ["0", "0.0", "-1", "nan", "inf", "-inf", "abc", "true", ""])
def test_rerank_timeout_s_invalid_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_TIMEOUT_S", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_RERANK_TIMEOUT_S"):
        _load()


@pytest.mark.parametrize("value", [True, False])
def test_rerank_timeout_s_bool_is_rejected(value: bool) -> None:
    with pytest.raises(ValidationError, match="rerank_timeout_s must be a number"):
        RetrievalSettings(_env_file=None, RETRIEVAL_RERANK_TIMEOUT_S=value)  # type: ignore[call-arg]


# --- factory -----------------------------------------------------------------------------


class _Embedder:
    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dimension(self) -> int:
        return 1

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        return [[1.0] for _ in texts]


class _Index:
    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return []

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return []


def test_factory_passes_both_settings_to_the_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "12")
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_S", "4.5")
    embedder, index = _Embedder(), _Index()

    retriever = create_hybrid_retriever(embedder, index, _load())

    assert retriever._candidate_k == 12
    assert retriever._timeout_s == 4.5
    assert retriever._embedder is embedder
    assert retriever._index is index
    as_protocol: Retriever = retriever  # mypy: the factory returns a Retriever
    assert as_protocol is retriever


def test_factory_with_default_settings_uses_the_defaults() -> None:
    retriever = create_hybrid_retriever(_Embedder(), _Index(), _load())
    assert retriever._candidate_k == DEFAULT_CANDIDATE_K
    assert retriever._timeout_s == DEFAULT_TIMEOUT_S
    assert retriever._release_notes_weight == DEFAULT_RELEASE_NOTES_WEIGHT


def test_factory_passes_the_release_notes_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RELEASE_NOTES_WEIGHT", "0.25")
    retriever = create_hybrid_retriever(_Embedder(), _Index(), _load())
    assert retriever._release_notes_weight == 0.25


def test_factory_without_a_reranker_keeps_the_default_none() -> None:
    retriever = create_hybrid_retriever(_Embedder(), _Index(), _load())
    assert retriever._reranker is None
    assert retriever._rerank_candidates == DEFAULT_RERANK_CANDIDATES
    assert retriever._rerank_timeout_s == DEFAULT_RERANK_TIMEOUT_S


def test_factory_passes_the_reranker_and_rerank_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_CANDIDATES", "8")
    monkeypatch.setenv("RETRIEVAL_RERANK_TIMEOUT_S", "0.5")
    reranker = FakeReranker()
    retriever = create_hybrid_retriever(_Embedder(), _Index(), _load(), reranker=reranker)
    assert retriever._reranker is reranker
    assert retriever._rerank_candidates == 8
    assert retriever._rerank_timeout_s == 0.5


def test_factory_reranker_is_keyword_only() -> None:
    untyped: Any = create_hybrid_retriever  # mypy already rejects this call; check at runtime
    with pytest.raises(TypeError):
        untyped(_Embedder(), _Index(), _load(), FakeReranker())


# --- reranker selection: RETRIEVAL_RERANKER, RETRIEVAL_RERANK_MODEL, VOYAGE_API_KEY --------


def _voyage(monkeypatch: pytest.MonkeyPatch, *, key: str | None = FAKE_KEY) -> RetrievalSettings:
    monkeypatch.setenv("RETRIEVAL_RERANKER", "voyage")
    if key is not None:
        monkeypatch.setenv("VOYAGE_API_KEY", key)
    return _load()


def test_reranker_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_MODEL", "  rerank-3  ")
    settings = _voyage(monkeypatch)
    assert settings.reranker == "voyage"
    assert settings.rerank_model == "rerank-3"  # stripped
    assert settings.voyage_api_key is not None
    assert settings.voyage_api_key.get_secret_value() == FAKE_KEY


@pytest.mark.parametrize("value", ["none", "fake", "voyage"])
def test_valid_reranker_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", value)
    assert _load().reranker == value


@pytest.mark.parametrize("value", ["Voyage", "bge", "", "NONE"])
def test_invalid_reranker_value_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_RERANKER"):
        _load()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_rerank_model_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_MODEL", value)
    with pytest.raises(ValidationError, match="RETRIEVAL_RERANK_MODEL"):
        _load()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_voyage_key_means_not_set(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", value)
    assert _load().voyage_api_key is None


def test_voyage_key_is_masked_in_repr_and_str(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _voyage(monkeypatch)
    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings)


def test_validation_errors_do_not_echo_input_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", FAKE_KEY)
    monkeypatch.setenv("RETRIEVAL_RERANKER", "secret-looking-value")
    with pytest.raises(ValidationError) as info:
        _load()
    assert "secret-looking-value" not in str(info.value)
    assert FAKE_KEY not in str(info.value)


# --- create_reranker: local only -----------------------------------------------------------


def test_create_reranker_none_by_default() -> None:
    assert create_reranker(_load()) is None


def test_create_reranker_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", "fake")
    assert isinstance(create_reranker(_load()), FakeReranker)


def test_create_reranker_voyage_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="requires VOYAGE_API_KEY") as info:
        create_reranker(_voyage(monkeypatch, key=None))
    assert FAKE_KEY not in str(info.value)


def test_create_reranker_voyage_builds_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_MODEL", "rerank-3")
    monkeypatch.setenv("RETRIEVAL_RERANK_TIMEOUT_S", "1.5")
    reranker = create_reranker(_voyage(monkeypatch))  # the fixture fails on any request

    assert isinstance(reranker, VoyageReranker)
    assert repr(reranker) == "VoyageReranker(model='rerank-3', timeout_s=1.5)"
    asyncio.run(reranker.aclose())


def test_create_reranker_timeout_reaches_every_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANK_TIMEOUT_S", "1.5")
    seen: list[Any] = []

    # Replaces the network-blocking fixture for this test only: answers locally and
    # records the timeout httpx attached to the request. Still no real network.
    async def respond(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"data": [{"index": 0, "relevance_score": 0.5}]})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    reranker = create_reranker(_voyage(monkeypatch))
    assert isinstance(reranker, VoyageReranker)
    item = SearchResult(
        chunk_id="a",
        url="https://knowledgecenter.gcora.genpact.com/help/a",
        title="A",
        text="text a",
        score=0.5,
        product="sequence",
        version=None,
        doc_type="how-to",
        heading_path=["A"],
    )

    async def scenario() -> None:
        try:
            await reranker.rerank("q", [item], k=1)
            await reranker.rerank("q", [item], k=1)
        finally:
            await reranker.aclose()

    asyncio.run(scenario())
    expected = {"connect": 1.5, "read": 1.5, "write": 1.5, "pool": 1.5}
    assert seen == [expected, expected]


def test_create_reranker_fake_logs_a_warning_but_does_not_block(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", "fake")
    with caplog.at_level(logging.WARNING, logger="kc.retrieval.settings"):
        reranker = create_reranker(_load())
    assert isinstance(reranker, FakeReranker)  # a warning only, not a block
    (record,) = [r for r in caplog.records if r.name == "kc.retrieval.settings"]
    assert record.levelno == logging.WARNING
    assert "RETRIEVAL_RERANKER=fake" in record.getMessage()
    assert "development and tests only" in record.getMessage()


@pytest.mark.parametrize("value", ["none", "voyage"])
def test_create_reranker_other_values_log_no_fake_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", value)
    monkeypatch.setenv("VOYAGE_API_KEY", FAKE_KEY)
    with caplog.at_level(logging.WARNING, logger="kc.retrieval.settings"):
        reranker = create_reranker(_load())
    assert "development and tests only" not in caplog.text
    if isinstance(reranker, VoyageReranker):
        asyncio.run(reranker.aclose())


@pytest.mark.parametrize("value", ["fake", "voyage"])
def test_create_reranker_returns_a_new_object_each_time(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", value)
    monkeypatch.setenv("VOYAGE_API_KEY", FAKE_KEY)
    settings = _load()
    first, second = create_reranker(settings), create_reranker(settings)
    assert first is not None and second is not None and first is not second
    for reranker in (first, second):
        if isinstance(reranker, VoyageReranker):
            asyncio.run(reranker.aclose())


def test_create_reranker_wires_into_the_hybrid_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", "fake")
    settings = _load()
    retriever = create_hybrid_retriever(
        _Embedder(), _Index(), settings, reranker=create_reranker(settings)
    )
    assert isinstance(retriever._reranker, FakeReranker)


# --- open_reranker: create, then always close (no network) ----------------------------------


def test_open_reranker_voyage_opens_and_closes_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _voyage(monkeypatch)

    async def scenario() -> VoyageReranker:
        async with open_reranker(settings) as reranker:
            assert isinstance(reranker, VoyageReranker)
            assert not reranker._client.is_closed
        return reranker

    assert asyncio.run(scenario())._client.is_closed


@pytest.mark.parametrize(("value", "expected"), [("none", type(None)), ("fake", FakeReranker)])
def test_open_reranker_none_and_fake(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: type
) -> None:
    monkeypatch.setenv("RETRIEVAL_RERANKER", value)
    settings = _load()

    async def scenario() -> None:
        async with open_reranker(settings) as reranker:
            assert isinstance(reranker, expected)

    asyncio.run(scenario())


def test_open_reranker_config_error_is_raised_on_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _voyage(monkeypatch, key=None)
    entered = False

    async def scenario() -> None:
        nonlocal entered
        async with open_reranker(settings):
            entered = True

    with pytest.raises(ValueError, match="requires VOYAGE_API_KEY"):
        asyncio.run(scenario())
    assert not entered


def test_open_reranker_closes_and_reraises_the_same_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _voyage(monkeypatch)
    error = ValueError("inside the block")
    opened: list[VoyageReranker] = []

    async def scenario() -> None:
        async with open_reranker(settings) as reranker:
            assert isinstance(reranker, VoyageReranker)
            opened.append(reranker)
            raise error

    with pytest.raises(ValueError) as info:
        asyncio.run(scenario())
    assert info.value is error
    assert opened[0]._client.is_closed


def test_open_reranker_closes_on_cancellation_and_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _voyage(monkeypatch)
    opened: list[VoyageReranker] = []

    async def scenario() -> None:
        started = asyncio.Event()

        async def lifespan() -> None:
            async with open_reranker(settings) as reranker:
                assert isinstance(reranker, VoyageReranker)
                opened.append(reranker)
                started.set()
                await asyncio.sleep(10.0)

        task = asyncio.create_task(lifespan())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    asyncio.run(scenario())
    assert opened[0]._client.is_closed


@pytest.mark.parametrize("inside", ["cancel", "error"])
def test_close_failure_never_replaces_the_original_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, inside: str
) -> None:
    settings = _voyage(monkeypatch)

    async def failing_close(self: VoyageReranker) -> None:
        raise OSError(f"close failed {FAKE_KEY}")

    monkeypatch.setattr(VoyageReranker, "aclose", failing_close)
    error = ValueError("inside the block")

    async def scenario() -> asyncio.Task[None]:
        started = asyncio.Event()

        async def lifespan() -> None:
            async with open_reranker(settings):
                started.set()
                if inside == "error":
                    raise error
                await asyncio.sleep(10.0)

        task = asyncio.create_task(lifespan())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        if inside == "cancel":
            task.cancel()
        await asyncio.wait([task])
        return task

    with caplog.at_level(logging.WARNING, logger="kc.retrieval.settings"):
        task = asyncio.run(scenario())

    if inside == "cancel":
        assert task.cancelled()  # the cancellation is kept, not replaced by OSError
    else:
        assert task.exception() is error
    assert "closing the reranker failed (OSError)" in caplog.text
    assert FAKE_KEY not in caplog.text


def test_close_failure_on_a_normal_exit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _voyage(monkeypatch)
    close_error = OSError("close failed")

    async def failing_close(self: VoyageReranker) -> None:
        raise close_error

    monkeypatch.setattr(VoyageReranker, "aclose", failing_close)

    async def scenario() -> None:
        async with open_reranker(settings):
            pass

    with pytest.raises(OSError) as info:
        asyncio.run(scenario())
    assert info.value is close_error
