"""The real answerer: retrieve passages, ask the model, check its citations.

Flow:
1. retriever.search(question, k)       -> no results: not_found, the model is NOT called
2. build the prompt (prompts.py)
3. llm.generate(...)                    -> LlmAnswer (status, text, cited_passage_ids)
4. check citations against the passages that were actually sent -> Answer (contract)

Grounding is enforced in code, not only in the prompt: cited ids that do not match a
passage are dropped, and an "answered" reply left without any valid citation becomes
not_found. Errors from the retriever (RetrievalError) and the model (GenerationError)
are not caught here; the API maps them to 503.
"""

import logging
from collections.abc import Sequence

from contracts.generation import Answer, AskRequest, Citation
from contracts.retrieval import Retriever, SearchResult
from kc.generation.llm import LlmAnswer, LlmClient
from kc.generation.prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)

DEFAULT_K = 5
NOT_FOUND_TEXT = "I didn't find an answer to this question in the Cora Knowledge Center."


class RagAnswerer:
    def __init__(self, retriever: Retriever, llm: LlmClient, *, k: int = DEFAULT_K) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._retriever = retriever
        self._llm = llm
        self._k = k

    async def answer(self, request: AskRequest) -> Answer:
        results = await self._retriever.search(request.question, k=self._k)
        if not results:
            return Answer(text=NOT_FOUND_TEXT, status="not_found")

        llm_answer = await self._llm.generate(
            system=SYSTEM_PROMPT,
            user_message=build_user_message(request.question, results),
        )
        return _to_answer(llm_answer, results)


def _valid_cited_ids(cited_ids: Sequence[int], passage_count: int) -> list[int]:
    """Keep ids that point to a sent passage (1..passage_count), first occurrence only."""
    valid: list[int] = []
    for passage_id in cited_ids:
        if 1 <= passage_id <= passage_count and passage_id not in valid:
            valid.append(passage_id)
    return valid


def _to_answer(llm_answer: LlmAnswer, results: Sequence[SearchResult]) -> Answer:
    if llm_answer.status != "answered":
        # not_found / needs_clarification: the text is a message, not a sourced claim.
        return Answer(text=llm_answer.text, status=llm_answer.status)

    valid_ids = _valid_cited_ids(llm_answer.cited_passage_ids, len(results))
    dropped = [i for i in llm_answer.cited_passage_ids if i not in valid_ids]
    if dropped:
        logger.warning("Dropped cited ids not matching any passage: %s", dropped)

    if not valid_ids:
        logger.warning("Model answered without any valid citation; returning not_found")
        return Answer(text=NOT_FOUND_TEXT, status="not_found")

    citations = [
        Citation(
            id=passage_id,
            url=results[passage_id - 1].url,
            title=results[passage_id - 1].title,
            chunk_id=results[passage_id - 1].chunk_id,
        )
        for passage_id in valid_ids
    ]
    return Answer(text=llm_answer.text, citations=citations, status="answered")
