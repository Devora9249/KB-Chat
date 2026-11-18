"""Retrieval evaluation: run Golden Set cases through a Retriever and score them.

- Positive cases (non-empty `expected_urls`) get Hit, Recall and Reciprocal Rank at
  article level (see `eval/metrics.py`).
- Negative cases (`expected_urls == []`) are run but not scored: "the system says it did
  not find an answer" is a property of the generated answer (generation eval).
- No `Filters` are passed: real users send none (`AskRequest` has none), and using the
  Golden Set's own `product` label as a filter would inflate the scores.
- A `RetrievalError` on one case is recorded and the run continues; any other exception
  is a bug and propagates.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from contracts.evaluation import EvalCase
from contracts.retrieval import RetrievalError, Retriever, SearchResult
from eval import metrics


@dataclass(frozen=True)
class CaseScores:
    hit: bool
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class CaseResult:
    """Outcome of one case. `scores` is None for negative and failed cases."""

    case: EvalCase
    retrieved_urls: tuple[str, ...] = ()
    scores: CaseScores | None = None
    error: str | None = None


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


@dataclass(frozen=True)
class RetrievalReport:
    """All case results plus aggregate metrics, averaged over scored cases only."""

    k: int
    results: tuple[CaseResult, ...]

    @property
    def scores(self) -> list[CaseScores]:
        return [r.scores for r in self.results if r.scores is not None]

    @property
    def num_negative(self) -> int:
        return sum(1 for r in self.results if not r.case.expected_urls and r.error is None)

    @property
    def num_errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def hit_rate(self) -> float | None:
        return _mean([float(s.hit) for s in self.scores])

    @property
    def mean_recall(self) -> float | None:
        return _mean([s.recall for s in self.scores])

    @property
    def mrr(self) -> float | None:
        return _mean([s.reciprocal_rank for s in self.scores])


def score_case(case: EvalCase, results: Sequence[SearchResult]) -> CaseResult:
    """Score one case from its search results (pure: no I/O)."""
    urls = metrics.ranked_unique_urls(results)
    if not case.expected_urls:
        return CaseResult(case=case, retrieved_urls=tuple(urls))
    scores = CaseScores(
        hit=metrics.hit(urls, case.expected_urls),
        recall=metrics.recall(urls, case.expected_urls),
        reciprocal_rank=metrics.reciprocal_rank(urls, case.expected_urls),
    )
    return CaseResult(case=case, retrieved_urls=tuple(urls), scores=scores)


async def evaluate_retrieval(
    retriever: Retriever, cases: Sequence[EvalCase], k: int
) -> RetrievalReport:
    """Evaluate every case in order.

    Sequential on purpose: deterministic, and a real retriever (embedding API, database)
    is not flooded. Bounded concurrency can be added when the set grows.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    results: list[CaseResult] = []
    for case in cases:
        try:
            found = await retriever.search(case.question, k=k)
        except RetrievalError as exc:
            results.append(CaseResult(case=case, error=f"{type(exc).__name__}: {exc}"))
        else:
            results.append(score_case(case, found))
    return RetrievalReport(k=k, results=tuple(results))
