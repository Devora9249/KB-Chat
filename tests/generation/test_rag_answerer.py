"""Tests for RagAnswerer: search -> prompt -> model -> checked citations."""

import asyncio
from dataclasses import dataclass, field

import pytest

from contracts.generation import Answer, AskRequest
from contracts.retrieval import Filters, RetrievalError, SearchResult
from kc.generation.fake_llm import FakeLlmClient
from kc.generation.llm import GenerationError, LlmAnswer
from kc.generation.prompts import SYSTEM_PROMPT
from kc.generation.rag_answerer import NOT_FOUND_TEXT, RagAnswerer


def make_result(n: int) -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk-{n}",
        url=f"https://knowledgecenter.gcora.genpact.com/help/page-{n}",
        title=f"Page {n}",
        text=f"Passage text number {n}.",
        score=1.0 / n,
        product="sequence",
        version="9",
        doc_type="how-to",
        heading_path=["Section"],
    )


@dataclass
class StubRetriever:
    """Returns fixed results (or raises) and records how it was called."""

    results: list[SearchResult] = field(default_factory=list)
    error: Exception | None = None
    calls: list[tuple[str, Filters | None, int]] = field(default_factory=list)

    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        self.calls.append((query, filters, k))
        if self.error is not None:
            raise self.error
        return self.results


THREE_RESULTS = [make_result(1), make_result(2), make_result(3)]


def run(answerer: RagAnswerer, question: str = "How do I configure SAML?") -> Answer:
    return asyncio.run(answerer.answer(AskRequest(question=question)))


def llm_says(status: str, cited: list[int], text: str = "Answer [1].") -> FakeLlmClient:
    return FakeLlmClient(
        answer=LlmAnswer.model_validate(
            {"status": status, "text": text, "cited_passage_ids": cited}
        )
    )


# --- 1. no results ---------------------------------------------------------------------


def test_no_results_is_not_found_without_calling_the_model() -> None:
    llm = FakeLlmClient()
    answer = run(RagAnswerer(StubRetriever(results=[]), llm))
    assert answer.status == "not_found"
    assert answer.text == NOT_FOUND_TEXT
    assert llm.calls == []


# --- 2. what is sent -------------------------------------------------------------------


def test_retriever_gets_the_question_and_k() -> None:
    retriever = StubRetriever(results=THREE_RESULTS)
    run(RagAnswerer(retriever, FakeLlmClient(), k=3), question="What is SAML?")
    assert retriever.calls == [("What is SAML?", None, 3)]


def test_model_gets_the_system_prompt_passages_and_question() -> None:
    llm = FakeLlmClient()
    run(RagAnswerer(StubRetriever(results=THREE_RESULTS), llm), question="What is SAML?")
    assert len(llm.calls) == 1
    assert llm.calls[0].system == SYSTEM_PROMPT
    assert "Passage text number 3." in llm.calls[0].user_message
    assert "What is SAML?" in llm.calls[0].user_message


# --- 3. citations ----------------------------------------------------------------------


def test_cited_ids_map_to_the_right_passages() -> None:
    answer = run(RagAnswerer(StubRetriever(results=THREE_RESULTS), llm_says("answered", [3, 1])))
    assert answer.status == "answered"
    assert [(c.id, c.url, c.title, c.chunk_id) for c in answer.citations] == [
        (3, THREE_RESULTS[2].url, "Page 3", "chunk-3"),
        (1, THREE_RESULTS[0].url, "Page 1", "chunk-1"),
    ]


def test_invalid_and_duplicate_ids_are_dropped() -> None:
    answer = run(
        RagAnswerer(StubRetriever(results=THREE_RESULTS), llm_says("answered", [0, 2, 7, 2, -1]))
    )
    assert [c.id for c in answer.citations] == [2]


def test_answered_without_valid_citations_becomes_not_found() -> None:
    answer = run(
        RagAnswerer(StubRetriever(results=THREE_RESULTS), llm_says("answered", [9], "Made up."))
    )
    assert answer.status == "not_found"
    assert answer.text == NOT_FOUND_TEXT
    assert answer.citations == []


def test_answered_with_empty_citations_becomes_not_found() -> None:
    answer = run(RagAnswerer(StubRetriever(results=THREE_RESULTS), llm_says("answered", [])))
    assert answer.status == "not_found"


# --- 4. other statuses -----------------------------------------------------------------


def test_not_found_from_model_keeps_its_text_and_drops_citations() -> None:
    llm = llm_says("not_found", [1], "I can only answer questions about Cora products.")
    answer = run(RagAnswerer(StubRetriever(results=THREE_RESULTS), llm))
    assert answer.status == "not_found"
    assert answer.text == "I can only answer questions about Cora products."
    assert answer.citations == []


def test_needs_clarification_passes_through() -> None:
    llm = llm_says("needs_clarification", [], "Which product and version do you use?")
    answer = run(RagAnswerer(StubRetriever(results=THREE_RESULTS), llm))
    assert answer.status == "needs_clarification"
    assert answer.text == "Which product and version do you use?"
    assert answer.citations == []


# --- 5. errors -------------------------------------------------------------------------


def test_retrieval_error_propagates_and_model_is_not_called() -> None:
    llm = FakeLlmClient()
    retriever = StubRetriever(error=RetrievalError("search timed out"))
    with pytest.raises(RetrievalError):
        run(RagAnswerer(retriever, llm))
    assert llm.calls == []


def test_generation_error_propagates() -> None:
    llm = FakeLlmClient(error=GenerationError("overloaded"))
    with pytest.raises(GenerationError):
        run(RagAnswerer(StubRetriever(results=THREE_RESULTS), llm))


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        RagAnswerer(StubRetriever(), FakeLlmClient(), k=0)
