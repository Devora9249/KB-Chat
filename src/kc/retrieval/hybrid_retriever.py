"""Hybrid retriever: vector search + full-text search, fused with RRF.

Depends only on the shared contracts (`Embedder`, `SearchIndex`); the real or fake
implementations are injected by the caller.
"""

import asyncio
import math
import re
from collections.abc import Iterator, Sequence
from typing import Final, NoReturn

from contracts.common import DocType
from contracts.retrieval import Filters, RetrievalError, SearchResult
from contracts.storage import Embedder, EmbeddingError, IndexStoreError, SearchIndex
from kc.retrieval.rrf import reciprocal_rank_fusion

# Defaults; RetrievalSettings (kc.retrieval.settings) reads the configured values from the
# environment and uses these when a variable is not set.
DEFAULT_CANDIDATE_K: Final = 50
DEFAULT_TIMEOUT_S: Final = 10.0
# Multiplier for the fused score of release notes (1.0 = no down-weighting). To be tuned
# with evaluation metrics; no value below 1.0 has been chosen yet.
DEFAULT_RELEASE_NOTES_WEIGHT: Final = 1.0
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


def _check_timeout_s(timeout_s: float) -> None:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        raise TypeError(f"timeout_s must be an int or float, got {type(timeout_s).__name__}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be finite and > 0, got {timeout_s}")


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
    # model_copy(update=...) skips SearchResult validation. Safe here: the RRF score is a
    # positive finite float and `weight` was validated as finite in (0, 1], so the product
    # is finite too.
    weighted = [
        r.model_copy(update={"score": r.score * weight}) if r.doc_type == _RELEASE_NOTES else r
        for r in results
    ]
    return sorted(weighted, key=lambda r: (-r.score, r.chunk_id))


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
        release_notes_weight: multiplier for the fused score of results with
            doc_type "release-notes", in (0, 1]; 1.0 (the default) changes nothing.

    Raises:
        TypeError: `candidate_k` is not an int, or `timeout_s` / `release_notes_weight`
            is not an int or float (bool is rejected for all three).
        ValueError: `candidate_k` < 1, `timeout_s` is not finite or is <= 0, or
            `release_notes_weight` is not finite or not in (0, 1].
    """

    def __init__(
        self,
        embedder: Embedder,
        index: SearchIndex,
        *,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        release_notes_weight: float = DEFAULT_RELEASE_NOTES_WEIGHT,
    ) -> None:
        _check_positive_int(candidate_k, "candidate_k")
        _check_timeout_s(timeout_s)
        _check_release_notes_weight(release_notes_weight)
        self._embedder = embedder
        self._index = index
        self._candidate_k = candidate_k
        self._timeout_s = timeout_s
        self._release_notes_weight = release_notes_weight

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
           The returned `score` is then the weighted RRF score.
        6. Return the first `k` (cut AFTER weighting, so a result below `k` can move up).

        One time budget (`timeout_s`) covers both branches; checks 1-2 run before it.
        `timeout_s` is when both branches are CANCELLED, not a hard limit on the wait:
        `search` returns only after the cancelled work has actually finished, so if a
        dependency ignores cancellation (or keeps running in a thread that cancellation
        cannot stop), the real wait can be longer than `timeout_s`.

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
        - asyncio.CancelledError is never converted. External cancellation always ends
          in CancelledError, even if a branch failed at the same time (that error is
          then kept in the CancelledError's `__cause__`).
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
        return _weight_release_notes(fused, self._release_notes_weight)[:k]

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
