"""Tests for the HTTP layer: /health and /v1/ask."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contracts.generation import Answer, AskRequest
from kc.api.app import create_app
from kc.api.dependencies import get_answerer


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_in_contract_shape(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": "How do I configure SAML in SeQuence 9?"})
    assert response.status_code == 200
    answer = Answer.model_validate(response.json())  # fails if the shape is wrong
    assert answer.status == "answered"
    assert len(answer.citations) == 2


def test_ask_empty_question_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_missing_question_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/ask", json={})
    assert response.status_code == 422


class StubAnswerer:
    """Proves the endpoint uses whatever answerer is injected."""

    async def answer(self, request: AskRequest) -> Answer:
        return Answer(text=f"stub: {request.question}", status="not_found")


def test_ask_uses_injected_answerer(app: FastAPI, client: TestClient) -> None:
    app.dependency_overrides[get_answerer] = StubAnswerer
    response = client.post("/v1/ask", json={"question": "anything at all here"})
    assert response.json()["text"] == "stub: anything at all here"
