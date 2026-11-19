"""Dependency providers for the API: the composition root.

This is the ONLY place that picks concrete implementations. Everything else depends on
interfaces (`Retriever`, `LlmClient`, `Answerer`). Tests replace any provider with
`app.dependency_overrides`.

Integration day:
- get_retriever(): FakeRetriever -> the shared HybridRetriever (built in the lifespan)
"""

from functools import lru_cache
from typing import Annotated

from anthropic import AsyncAnthropic
from fastapi import Depends

from contracts.retrieval import Retriever
from kc.api.settings import ApiSettings
from kc.generation.answerer import Answerer
from kc.generation.claude_llm import ClaudeLlmClient
from kc.generation.fake_answerer import FakeAnswerer
from kc.generation.fake_llm import FakeLlmClient
from kc.generation.llm import LlmClient
from kc.generation.rag_answerer import RagAnswerer
from kc.retrieval.fake_retriever import FakeRetriever

# The SDK's defaults (10 min timeout, 2 retries) could keep a call running long after the
# API's overall deadline (API_REQUEST_TIMEOUT_S, 45s) has given up. Worst case here:
# 2 attempts x 20s, plus a short backoff, stays inside it.
LLM_TIMEOUT_S = 20.0
LLM_MAX_RETRIES = 1


@lru_cache
def get_settings() -> ApiSettings:
    """Read settings once per process (not on every request)."""
    return ApiSettings()


@lru_cache
def get_anthropic_client() -> AsyncAnthropic:
    """One SDK client per process: it holds the connection pool, so it is reused.

    The API key is read by the SDK from ANTHROPIC_API_KEY. Creating the client does not
    need the key; only a real call does.
    """
    return AsyncAnthropic(timeout=LLM_TIMEOUT_S, max_retries=LLM_MAX_RETRIES)


def get_retriever() -> Retriever:
    return FakeRetriever()


def get_llm(settings: Annotated[ApiSettings, Depends(get_settings)]) -> LlmClient:
    if settings.llm == "fake":
        return FakeLlmClient()
    return ClaudeLlmClient(
        get_anthropic_client(), model=settings.llm_model, effort=settings.llm_effort
    )


def get_answerer(
    settings: Annotated[ApiSettings, Depends(get_settings)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm: Annotated[LlmClient, Depends(get_llm)],
) -> Answerer:
    if settings.answerer == "fake":
        return FakeAnswerer()
    return RagAnswerer(retriever, llm, k=settings.rag_k)
