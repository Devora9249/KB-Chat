"""Tests for the search settings (RetrievalSettings) and the HybridRetriever factory."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from contracts.retrieval import Filters, Retriever, SearchResult
from contracts.storage import InputType
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RELEASE_NOTES_WEIGHT,
    DEFAULT_TIMEOUT_S,
)
from kc.retrieval.settings import (
    RetrievalSettings,
    create_hybrid_retriever,
    load_retrieval_settings,
)

ENV_NAMES = ("RETRIEVAL_CANDIDATE_K", "RETRIEVAL_TIMEOUT_S", "RETRIEVAL_RELEASE_NOTES_WEIGHT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


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
