"""Shared test-data builders for tests/eval (contract objects with only the relevant fields)."""

from contracts.evaluation import EvalCase
from contracts.retrieval import SearchResult

KC = "https://knowledgecenter.gcora.genpact.com/help/"


def make_result(url: str, chunk_id: str = "c") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        url=url,
        title="t",
        text="t",
        score=1.0,
        product="sequence",
        version="9",
        doc_type="how-to",
        heading_path=[],
    )


def make_case(question: str = "q", expected_urls: list[str] | None = None) -> EvalCase:
    return EvalCase(
        question=question,
        expected_urls=[KC + "a"] if expected_urls is None else expected_urls,
        product="sequence",
        reference_answer=None,
    )
