"""Generation metrics: pure functions, no I/O, no LLM calls.

Retrieval metrics (eval/metrics.py) ask "did search find the right article?". These ask
questions about the *answer* the system wrote (`Answer`), using only what can be checked
mechanically against the Golden Set (`EvalCase`):

- `status_correct`: did the system answer when it should, and say "not found" when the
  knowledge base has no answer (negative cases)?
- `markers_valid`: is an answer's [n] markers backed by real citations (grounding)?
- `cites_expected_article`: does at least one citation point to a correct article?

What these do NOT check: whether the answer text is factually right. That needs the
human `reference_answer` and an LLM judge, which is a later step.

Functions that need a correct article (`cites_expected_article`) are defined only for
positive cases; for a negative case (empty `expected_urls`) they raise ValueError instead
of returning a misleading number, the same rule as in eval/metrics.py.
"""

import re

from contracts.evaluation import EvalCase
from contracts.generation import Answer, AnswerStatus

# "[1]", "[12]": the citation markers the answer text uses. Words in brackets like
# "[Note]" are not markers, so only digits are matched.
_MARKER = re.compile(r"\[(\d+)\]")


def expected_status(case: EvalCase) -> AnswerStatus:
    """The status a correct system should return for this case.

    A question with expected articles should be answered; a question with none (a negative
    case) should get "not_found". "needs_clarification" is never the expected status in
    the Golden Set: it is a legitimate system behaviour, but it fails this check.
    """
    return "answered" if case.expected_urls else "not_found"


def status_correct(answer: Answer, case: EvalCase) -> bool:
    """True if the answer's status is the one the Golden Set expects (see `expected_status`)."""
    return answer.status == expected_status(case)


def marker_ids(text: str) -> list[int]:
    """Marker numbers in the text, in order, each once: "see [2] and [1] and [2]" -> [2, 1]."""
    return list(dict.fromkeys(int(m) for m in _MARKER.findall(text)))


def dangling_markers(answer: Answer) -> list[int]:
    """Markers in the text that have no matching citation (the answer cites a source it lacks)."""
    cited = {citation.id for citation in answer.citations}
    return [m for m in marker_ids(answer.text) if m not in cited]


def markers_valid(answer: Answer) -> bool:
    """True if an "answered" answer has at least one marker and none of them dangle.

    Answers that make no claims ("not_found", "needs_clarification") have nothing to ground,
    so they pass trivially.
    """
    if answer.status != "answered":
        return True
    return bool(marker_ids(answer.text)) and not dangling_markers(answer)


def cites_expected_article(answer: Answer, case: EvalCase) -> bool:
    """True if at least one citation's URL is one of the case's expected articles."""
    if not case.expected_urls:
        raise ValueError("metric is undefined for a negative case (empty expected_urls)")
    expected = set(case.expected_urls)
    return any(citation.url in expected for citation in answer.citations)
