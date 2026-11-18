"""Dependency providers for the API: the composition root.

This is the ONLY place that picks concrete implementations. Everything else depends on
interfaces (`Retriever`, `LlmClient`, `Answerer`). Tests replace any provider with
`app.dependency_overrides`.

Integration day:
- get_retriever(): FakeRetriever -> the shared HybridRetriever (built in the lifespan)
- get_llm():       FakeLlmClient -> the real Claude client
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from contracts.retrieval import Retriever
from kc.api.settings import ApiSettings
from kc.generation.answerer import Answerer
from kc.generation.fake_answerer import FakeAnswerer
from kc.generation.fake_llm import FakeLlmClient
from kc.generation.llm import LlmClient
from kc.generation.rag_answerer import RagAnswerer
from kc.retrieval.fake_retriever import FakeRetriever


@lru_cache
def get_settings() -> ApiSettings:
    """Read settings once per process (not on every request)."""
    return ApiSettings()


def get_retriever() -> Retriever:
    return FakeRetriever()


def get_llm() -> LlmClient:
    return FakeLlmClient()


def get_answerer(
    settings: Annotated[ApiSettings, Depends(get_settings)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm: Annotated[LlmClient, Depends(get_llm)],
) -> Answerer:
    if settings.answerer == "fake":
        return FakeAnswerer()
    return RagAnswerer(retriever, llm, k=settings.rag_k)
