"""Tests for the API wiring (fake vs rag answerer) and the error mapping (503 / 500)."""

import asyncio
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contracts.generation import Answer, AskRequest
from contracts.retrieval import Filters, RetrievalError, SearchResult
from kc.api.app import (
    DEADLINE_EXCEEDED,
    GENERATION_UNAVAILABLE,
    SEARCH_UNAVAILABLE,
    create_app,
)
from kc.api.dependencies import get_answerer, get_llm, get_retriever, get_settings
from kc.api.settings import ApiSettings
from kc.generation.fake_llm import FakeLlmClient
from kc.generation.llm import GenerationError, LlmAnswer

QUESTION = {"question": "How do I configure SAML in SeQuence 9?"}


def settings(**overrides: object) -> ApiSettings:
    return ApiSettings.model_validate(overrides)


class OneResultRetriever:
    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id="saml#0",
                url="https://knowledgecenter.gcora.genpact.com/help/configure-saml",
                title="Configure SAML",
                text="Open the Authentication page.",
                score=1.0,
                product="sequence",
                version="9",
                doc_type="how-to",
                heading_path=["Security"],
            )
        ]


class FailingRetriever:
    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        raise RetrievalError("vector search failed: connection refused to db-host:5432")


class AnswererRaising:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def answer(self, request: AskRequest) -> Answer:
        raise self._error


class SlowAnswerer:
    async def answer(self, request: AskRequest) -> Answer:
        await asyncio.sleep(5)
        return Answer(text="too late", status="not_found")


@pytest.fixture
def app() -> FastAPI:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings()
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # raise_server_exceptions=False: see the 500 response a real client would get.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# --- wiring -----------------------------------------------------------------------------


def test_default_answerer_is_the_fake(client: TestClient) -> None:
    answer = Answer.model_validate(client.post("/v1/ask", json=QUESTION).json())
    assert answer.text.startswith("[FAKE]")


def test_rag_answerer_uses_injected_retriever_and_llm(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: settings(answerer="rag")
    app.dependency_overrides[get_retriever] = OneResultRetriever
    app.dependency_overrides[get_llm] = lambda: FakeLlmClient(
        answer=LlmAnswer(status="answered", text="Open it [1].", cited_passage_ids=[1])
    )
    response = client.post("/v1/ask", json=QUESTION)
    assert response.status_code == 200
    answer = Answer.model_validate(response.json())
    assert answer.status == "answered"
    assert answer.citations[0].url.endswith("/configure-saml")


def test_rag_answerer_works_with_the_default_fakes(app: FastAPI, client: TestClient) -> None:
    # FakeRetriever + FakeLlmClient, no overrides: the full path runs end to end.
    app.dependency_overrides[get_settings] = lambda: settings(answerer="rag")
    response = client.post("/v1/ask", json={"question": "How do I configure SAML authentication?"})
    assert response.status_code == 200
    Answer.model_validate(response.json())


# --- 503: temporary failures ------------------------------------------------------------


def test_retrieval_error_is_503_with_a_generic_message(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: settings(answerer="rag")
    app.dependency_overrides[get_retriever] = FailingRetriever
    response = client.post("/v1/ask", json=QUESTION)
    assert response.status_code == 503
    assert response.json() == {"detail": SEARCH_UNAVAILABLE}
    assert "db-host" not in response.text  # internal details never reach the user


def test_generation_error_is_503(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_answerer] = lambda: AnswererRaising(GenerationError("529"))
    response = client.post("/v1/ask", json=QUESTION)
    assert response.status_code == 503
    assert response.json() == {"detail": GENERATION_UNAVAILABLE}


def test_deadline_exceeded_is_503(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: settings(request_timeout_s=0.05)
    app.dependency_overrides[get_answerer] = SlowAnswerer
    response = client.post("/v1/ask", json=QUESTION)
    assert response.status_code == 503
    assert response.json() == {"detail": DEADLINE_EXCEEDED}


# --- 500: bugs --------------------------------------------------------------------------


def test_unexpected_error_is_500(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_answerer] = lambda: AnswererRaising(ValueError("bug"))
    assert client.post("/v1/ask", json=QUESTION).status_code == 500


def test_timeout_error_not_from_our_deadline_is_500(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_answerer] = lambda: AnswererRaising(TimeoutError("inner"))
    assert client.post("/v1/ask", json=QUESTION).status_code == 500


# --- settings ---------------------------------------------------------------------------


def test_settings_defaults() -> None:
    defaults = ApiSettings(_env_file=None)
    assert defaults.answerer == "fake"
    assert defaults.request_timeout_s == 45.0
    assert defaults.rag_k == 5


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_ANSWERER", "rag")
    monkeypatch.setenv("API_REQUEST_TIMEOUT_S", "30")
    monkeypatch.setenv("API_RAG_K", "8")
    loaded = ApiSettings(_env_file=None)
    assert (loaded.answerer, loaded.request_timeout_s, loaded.rag_k) == ("rag", 30.0, 8)


@pytest.mark.parametrize(
    "bad",
    [{"answerer": "claude"}, {"request_timeout_s": 0}, {"rag_k": 0}],
)
def test_settings_reject_invalid_values(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        settings(**bad)
