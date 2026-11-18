"""Hybrid retriever: vector search + full-text search, fused with RRF, optionally reranked.

Depends only on the shared contracts (`Embedder`, `SearchIndex`) and the internal
`Reranker` protocol; the real or fake implementations are injected by the caller.
"""

import asyncio
import logging
import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Final, NoReturn

from contracts.common import DocType
from contracts.retrieval import Filters, RetrievalError, SearchResult
from contracts.storage import Embedder, EmbeddingError, IndexStoreError, SearchIndex
from kc.retrieval.reranker import Reranker, RerankError
from kc.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Defaults; RetrievalSettings (kc.retrieval.settings) reads the configured values from the
# environment and uses these when a variable is not set.
DEFAULT_CANDIDATE_K: Final = 50
DEFAULT_TIMEOUT_S: Final = 10.0
# Multiplier for the fused score of release notes (1.0 = no down-weighting). To be tuned
# with evaluation metrics; no value below 1.0 has been chosen yet.
DEFAULT_RELEASE_NOTES_WEIGHT: Final = 1.0
# How many of the top fused results are sent to the reranker (at least k), and its own
# time budget (separate from timeout_s). Used only when a reranker is configured.
DEFAULT_RERANK_CANDIDATES: Final = 20
DEFAULT_RERANK_TIMEOUT_S: Final = 3.0
_RELEASE_NOTES: Final[DocType] = "release-notes"
_RRF_K: Final = 60
# Same tokenization as FakeRetriever and InMemoryIndexStore.text_search.
_TOKEN_RE: Final = re.compile(r"\w+", re.UNICODE)
# Known infrastructure failures (agreed with Leah and Raizy): wrapped in RetrievalError.
# Everything else (ValueError, TypeError, other OSError, ...) is a bug and is NOT wrapped.
_INFRA_ERRORS: Final = (EmbeddingError, IndexStoreError, TimeoutError)

_Branch = tuple[str, "asyncio.Task[list[SearchResult]] | None"]


def _leaves(exc: BaseException) -> Iterator[BaseException]:
    """Depth-first leaf exceptions. Groups are only read, never modified or rebuilt."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaves(inner)
    else:
        yield exc


def _raise_translated(group: ExceptionGroup[Exception], branches: Sequence[_Branch]) -> NoReturn:
    """Turn the TaskGroup's ExceptionGroup into the error `search` raises.

    Must be called while `group` is being handled, so Python sets `__context__` of the
    raised error to `group` automatically.

    - Each branch's ORIGINAL exception is taken from its task (not from the group's
      order, which is completion order), so the result does not depend on timing.
    - Classification walks the leaves depth-first, vector branch before text branch.
    - First non-infrastructure leaf -> raised as is (same object, not wrapped).
    - Only infrastructure leaves -> RetrievalError; `__cause__` is the branch's original
      exception (one branch), or a NEW ExceptionGroup [vector, text] (both branches).
    - If the group cannot be matched exactly to the branch exceptions, it is re-raised
      unchanged, so nothing is ever swallowed.
    """
    failed: list[tuple[str, Exception]] = []
    for name, task in branches:
        if task is None or not task.done() or task.cancelled():
            continue
        exc = task.exception()
        if exc is None:
            continue
        if not isinstance(exc, Exception):
            raise group
        failed.append((name, exc))

    if not failed or {id(e) for e in group.exceptions} != {id(e) for _, e in failed}:
        raise group

    for _, exc in failed:
        for leaf in _leaves(exc):
            if not isinstance(leaf, _INFRA_ERRORS):
                raise leaf

    names = " and ".join(name for name, _ in failed)
    cause: Exception = (
        failed[0][1]
        if len(failed) == 1
        else ExceptionGroup("retrieval failures", [exc for _, exc in failed])
    )
    raise RetrievalError(f"{names} failed") from cause


def _check_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _check_timeout_s(timeout_s: float, name: str = "timeout_s") -> None:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        raise TypeError(f"{name} must be an int or float, got {type(timeout_s).__name__}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {timeout_s}")


def _check_release_notes_weight(weight: float) -> None:
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        raise TypeError(
            f"release_notes_weight must be an int or float, got {type(weight).__name__}"
        )
    if not math.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError(f"release_notes_weight must be finite and in (0, 1], got {weight}")


def _weight_release_notes(results: list[SearchResult], weight: float) -> list[SearchResult]:
    """Multiply the score of release notes by `weight`, then re-sort.

    Same order rule as RRF: score descending, ties by chunk_id ascending. With weight 1.0
    the list is returned unchanged, so the RRF result is kept exactly.
    """
    if weight == 1.0:
        return results
    # model_copy(update=...) skips SearchResult validation. Safe here: the score is a
    # finite number >= 0 (an RRF score, or a reranker score already checked by
    # _check_rerank_output) and `weight` was validated as finite in (0, 1], so the product
    # is finite and >= 0 too.
    weighted = [
        r.model_copy(update={"score": r.score * weight}) if r.doc_type == _RELEASE_NOTES else r
        for r in results
    ]
    return sorted(weighted, key=lambda r: (-r.score, r.chunk_id))


def _check_rerank_output(pool: Sequence[SearchResult], reranked: Sequence[SearchResult]) -> None:
    """Raise ValueError if `reranked` breaks the Reranker protocol for this `pool`.

    The reranker is called with k=len(pool) and the pool has unique ids, so its output
    must be a full permutation of the pool, with finite scores >= 0. A violation is a
    reranker bug: it is raised, never turned into a fallback. The order is not checked
    (the caller re-sorts after weighting); the other fields are trusted (protocol).
    """
    ids = [r.chunk_id for r in reranked]
    if len(ids) != len(pool):
        raise ValueError(f"reranker returned {len(ids)} results for {len(pool)} candidates")
    duplicates = sorted(i for i, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"reranker returned duplicate chunk_ids: {duplicates}")
    expected = {r.chunk_id for r in pool}
    if set(ids) != expected:
        missing = sorted(expected - set(ids))
        extra = sorted(set(ids) - expected)
        raise ValueError(
            f"reranker output does not match the candidates: missing {missing}, extra {extra}"
        )
    for result in reranked:
        score: object = result.score  # model_copy(update=...) can store anything here
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(
                f"reranker score for {result.chunk_id!r} must be a number, "
                f"got {type(score).__name__}"
            )
        try:
            finite = math.isfinite(score)
        except OverflowError:  # an int too large for a float, e.g. 10**400
            raise ValueError(
                f"reranker score for {result.chunk_id!r} must be finite, "
                "got an int too large for a float"
            ) from None
        if not finite:
            raise ValueError(f"reranker score for {result.chunk_id!r} must be finite, got {score}")
        if score < 0:
            raise ValueError(f"reranker score for {result.chunk_id!r} must be >= 0, got {score}")


class HybridRetriever:
    """Retriever that combines vector search and full-text search.

    One instance is meant to live for the whole application. It holds only its
    dependencies and settings, set once in the constructor, and owns no resources
    (no close()): the embedder and index are managed by the caller. `search` keeps all
    per-request data in local variables, so one instance is safe for concurrent calls.

    Args:
        embedder: turns the query into a vector.
        index: runs vector_search and text_search.
        candidate_k: how many candidates to request from EACH search before fusion
            (the real request is max(candidate_k, k)).
        timeout_s: time budget for one search call, in seconds; when it expires both
            branches are cancelled (see `search` for why the real wait can be longer).
        release_notes_weight: multiplier for the score of results with doc_type
            "release-notes", in (0, 1]; 1.0 (the default) changes nothing. Applied to the
            fused RRF score and, when reranking succeeds, to the reranker score.
        reranker: optional; None (the default) keeps the RRF ranking exactly as before.
        rerank_candidates: how many of the top weighted RRF results are sent to the
            reranker (the real number is max(rerank_candidates, k), capped by how many
            results there are). Validated even when no reranker is set.
        rerank_timeout_s: the reranker's own time budget in seconds, separate from
            `timeout_s`; when it expires, search falls back to the RRF ranking.

    Raises:
        TypeError: `candidate_k` / `rerank_candidates` is not an int, or `timeout_s` /
            `release_notes_weight` / `rerank_timeout_s` is not an int or float (bool is
            rejected for all of them).
        ValueError: `candidate_k` / `rerank_candidates` < 1, `timeout_s` /
            `rerank_timeout_s` is not finite or is <= 0, or `release_notes_weight` is not
            finite or not in (0, 1].
    """

    def __init__(
        self,
        embedder: Embedder,
        index: SearchIndex,
        *,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        release_notes_weight: float = DEFAULT_RELEASE_NOTES_WEIGHT,
        reranker: Reranker | None = None,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
        rerank_timeout_s: float = DEFAULT_RERANK_TIMEOUT_S,
    ) -> None:
        _check_positive_int(candidate_k, "candidate_k")
        _check_timeout_s(timeout_s)
        _check_release_notes_weight(release_notes_weight)
        _check_positive_int(rerank_candidates, "rerank_candidates")
        _check_timeout_s(rerank_timeout_s, "rerank_timeout_s")
        self._embedder = embedder
        self._index = index
        self._candidate_k = candidate_k
        self._timeout_s = timeout_s
        self._release_notes_weight = release_notes_weight
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._rerank_timeout_s = rerank_timeout_s

    async def search(
        self, query: str, filters: Filters | None = None, k: int = 5
    ) -> list[SearchResult]:
        """Return up to `k` results, best first (contract: `Retriever.search`).

        Flow:
        1. Validate `k` (TypeError / ValueError), before any I/O.
        2. A query with no `\\w+` token returns [] without calling the dependencies
           ("_" counts as a token, because `\\w` includes the underscore).
        3. Run two branches concurrently, each asking for n = max(candidate_k, k)
           candidates with the same `filters` object:
           - vector: embed([query], input_type="query"), then vector_search;
           - text: text_search(query) (starts without waiting for the embedding).
        4. Fuse both lists with RRF (rank-based).
        5. Multiply the score of release notes by `release_notes_weight` and re-sort
           (score descending, ties by chunk_id); skipped when the weight is 1.0.
           Without a reranker, the returned `score` is this weighted RRF score.
        6. Only when a reranker is configured and step 5 gave any result: send the top
           max(rerank_candidates, k) of that list to `reranker.rerank(..., k=len(pool))`
           under its own budget `rerank_timeout_s`; check the output (a full permutation
           of the pool, scores finite and >= 0, else ValueError); apply the release-notes
           weight ONCE MORE, now to the reranker scores, and re-sort. The returned `score`
           is then the weighted reranker score. The two weightings are in different score
           spaces (the reranker replaces the RRF score, never reuses it), so each returned
           score is weighted exactly once.
           Fallback: if the reranker raises RerankError or TimeoutError (including its own
            budget expiring), a warning is logged with only the error type and the pool
            size (no query text, no exception details), and the result of step 5 is used,
            exactly as if no reranker were configured.
        7. Return the first `k` (cut AFTER weighting, so a result below `k` can move up).

        One time budget (`timeout_s`) covers both branches; checks 1-2 run before it.
        `timeout_s` is when both branches are CANCELLED, not a hard limit on the wait:
        `search` returns only after the cancelled work has actually finished, so if a
        dependency ignores cancellation (or keeps running in a thread that cancellation
        cannot stop), the real wait can be longer than `timeout_s`. Reranking has its own
        budget after that, so the worst case is about `timeout_s + rerank_timeout_s`. The
        same caveat applies: a reranker that blocks the event loop (synchronous work
        without a thread) cannot be stopped by its budget, and one that swallows the
        cancellation and returns a result has that result used.

        Errors (contract: RetrievalError):
        - Infrastructure failures (EmbeddingError, IndexStoreError, any TimeoutError,
          including our own budget expiring) -> RetrievalError, original in `__cause__`;
          both branches failed -> `__cause__` is ExceptionGroup([vector_exc, text_exc]).
        - Any other error -> raised unchanged (the same object, not wrapped); if several,
          the first one depth-first, vector branch before text branch (a deterministic
          choice). Because it is re-raised while the TaskGroup's ExceptionGroup is being
          handled, Python sets its `__context__` to that group, REPLACING any context the
          error had before; its `__cause__` is kept. If the dependency raised it with
          `raise ... from ...`, `__suppress_context__` stays True, so a printed traceback
          shows the cause and not the group, although the group is still reachable via
          `__context__`.
        - Reranker: only RerankError and TimeoutError lead to the fallback. Any other
          error (ValueError, TypeError, an ExceptionGroup even if it holds a RerankError)
          is raised unchanged; an invalid reranker output raises ValueError.
        - asyncio.CancelledError is never converted. External cancellation always ends
          in CancelledError, even if a branch failed at the same time (that error is
          then kept in the CancelledError's `__cause__`), and also during reranking
          (never a fallback).
        When a branch fails, the TaskGroup cancels the other one and waits for it, so no
        task is left running after `search` returns or raises.

        No retries. If either branch fails, nothing is returned (no partial results and
        no text-only fallback). Note: the two index calls are separate reads; if a chunk
        is updated between them, RRF keeps the version from vector_search.
        """
        _check_positive_int(k, "k")
        if _TOKEN_RE.search(query) is None:
            return []
        n = max(self._candidate_k, k)

        # External cancellation must always end in CancelledError. But when a branch has
        # already failed, TaskGroup gives its errors priority and drops a CancelledError
        # that arrives meanwhile. To tell the two apart we run the branches in a separate
        # `runner` task: the cancel requests that TaskGroup and asyncio.timeout make
        # internally land on `runner`, never on this task (in Python 3.12 TaskGroup does
        # not always undo its own request). So if THIS task's cancelling() count went
        # up, someone outside cancelled us. Cancelling this task also cancels `runner`,
        # and we resume only after `runner` has finished, so no task is left behind.
        current = asyncio.current_task()
        cancels_before = current.cancelling() if current is not None else 0
        runner = asyncio.create_task(self._run_branches(query, filters, n))
        try:
            vector_results, text_results = await runner
        except asyncio.CancelledError as cancel:
            # If `runner` had already FINISHED with an error when the cancel arrived,
            # asyncio replaced that error with this CancelledError and the error would be
            # lost. Keep it in __cause__. The guards make `exception()` safe: it raises if
            # `runner` is not done or was itself cancelled (then there is nothing to keep).
            if runner.done() and not runner.cancelled() and cancel.__cause__ is None:
                cancel.__cause__ = runner.exception()  # None after a success: no change
            raise
        except Exception as exc:
            if current is not None and current.cancelling() > cancels_before:
                # Cancelled from outside while the branches were failing: cancellation
                # wins; the error raised meanwhile is kept in __cause__.
                raise asyncio.CancelledError() from exc
            raise

        fused = reciprocal_rank_fusion(vector_results, text_results, rrf_k=_RRF_K)
        ranked = _weight_release_notes(fused, self._release_notes_weight)  # RRF score space
        return await self._rerank(query, ranked, k)

    async def _rerank(self, query: str, ranked: list[SearchResult], k: int) -> list[SearchResult]:
        """Step 6 of `search`: optional reranking with fallback; returns the final top k."""
        if self._reranker is None or not ranked:
            return ranked[:k]  # exactly the behaviour without a reranker
        pool = ranked[: max(self._rerank_candidates, k)]
        try:
            # asyncio.timeout turns only ITS OWN cancellation into TimeoutError; an
            # external cancel stays CancelledError (not caught here: never a fallback).
            async with asyncio.timeout(self._rerank_timeout_s):
                reranked = await self._reranker.rerank(query, pool, k=len(pool))
        except (RerankError, TimeoutError) as exc:
            # Only the error TYPE and the pool size: no query text (user content) and no
            # exception message or traceback (a provider error may echo the request).
            logger.warning(
                "reranker failed (%s); falling back to the RRF ranking for %d candidates",
                type(exc).__name__,
                len(pool),
            )
            return ranked[:k]
        _check_rerank_output(pool, reranked)
        # SearchResult.score is a float; model_copy(update=...) can carry a valid int score
        # through unchanged, so convert it (finite, already checked). Other fields are kept.
        as_float = [
            r if isinstance(r.score, float) else r.model_copy(update={"score": float(r.score)})
            for r in reranked
        ]
        # Reranker score space: the weight is applied once more, to the NEW scores.
        final = _weight_release_notes(as_float, self._release_notes_weight)
        return sorted(final, key=lambda r: (-r.score, r.chunk_id))[:k]

    async def _run_branches(
        self, query: str, filters: Filters | None, n: int
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """Both branches under one time budget; errors translated (see `search`)."""
        vector_task: asyncio.Task[list[SearchResult]] | None = None
        text_task: asyncio.Task[list[SearchResult]] | None = None
        budget = asyncio.timeout(self._timeout_s)
        try:
            async with budget:
                async with asyncio.TaskGroup() as tg:
                    vector_task = tg.create_task(self._vector_branch(query, filters, n))
                    text_task = tg.create_task(self._index.text_search(query, filters, k=n))
        except TimeoutError as exc:
            # Branch errors always arrive inside an ExceptionGroup, so a bare TimeoutError
            # here comes from `budget`. Any TimeoutError is infrastructure, so wrap anyway.
            reason = "search timed out"
            if budget.expired():
                reason = f"search timed out after {self._timeout_s}s"
            raise RetrievalError(reason) from exc
        except ExceptionGroup as group:
            _raise_translated(
                group,
                (
                    # The vector branch can fail while embedding OR in vector_search.
                    ("vector branch (embedding or vector search)", vector_task),
                    ("text search", text_task),
                ),
            )

        assert vector_task is not None and text_task is not None  # both created above
        return vector_task.result(), text_task.result()

    async def _vector_branch(
        self, query: str, filters: Filters | None, n: int
    ) -> list[SearchResult]:
        vectors = await self._embedder.embed([query], input_type="query")
        if len(vectors) != 1:
            # The Embedder contract guarantees one vector per text: this is a dependency bug.
            raise ValueError(f"embedder returned {len(vectors)} vectors for 1 query")
        return await self._index.vector_search(vectors[0], filters, k=n)
