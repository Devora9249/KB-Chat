"""Reranker over Voyage AI's REST API (POST /v1/rerank), with plain httpx (no SDK).

PRIVACY: `rerank` sends the user's query and the candidate passage texts to Voyage. It is
used only when RETRIEVAL_RERANKER=voyage (the default is "none"); do not enable it before
the privacy decision is made.

Building and using are separate: the constructor and `aclose` make no request; only
`rerank` (with at least one candidate) sends exactly one request. The instance owns one
`httpx.AsyncClient`; close it with `aclose()` (`kc.retrieval.settings.open_reranker`
does that for the application).

Errors (Reranker protocol):
- invalid `k` -> TypeError / ValueError, before any request;
- timeout -> TimeoutError; network / transport / decoding failure, a non-200 status
  (including 400) or a response that breaks the expected schema -> RerankError
  (HybridRetriever then falls back to the RRF ranking);
- anything else (bugs, e.g. a non-SearchResult input or using a closed client) is NOT
  caught; asyncio.CancelledError is never caught or wrapped.
Messages never contain the API key, the query or the response body.

Limits: Voyage accepts at most 1,000 documents per request; more unique candidates raise
RerankError WITHOUT sending a request (fallback). The 600K total-token limit is not
checked locally (no tokenizer): an overflow comes back as HTTP 400 -> RerankError.

Logs: a non-200 status is also logged as one WARNING with the status code only (never the
body, headers, key, query or passages); exceeding the document limit is logged with the
document count and the limit only. Each kind (every status code, and the document limit)
is rate-limited separately: at most one WARNING per 60 seconds. The number of occurrences
suppressed in between is reported with the NEXT occurrence that is logged (no timer), so
a burst followed by silence leaves its count unreported.

ASSUMPTION (not verified against the live API yet): the response is
{"data": [{"index": int, "relevance_score": number}, ...]}. If it is different, every call
raises RerankError (fallback), never a crash.
"""

import asyncio
import logging
import math
import time
from collections import Counter
from collections.abc import Sequence
from typing import Final

import httpx
from pydantic import SecretStr

from contracts.retrieval import SearchResult
from kc.retrieval.reranker import RerankError

logger = logging.getLogger(__name__)

_BASE_URL: Final = "https://api.voyageai.com"
_RERANK_PATH: Final = "/v1/rerank"
_MAX_DOCUMENTS: Final = 1000  # Voyage: "The number of documents cannot exceed 1,000."
_LOG_INTERVAL_S: Final = 60.0  # at most one WARNING per log key per interval
_TOO_MANY_DOCUMENTS: Final = "too_many_documents"  # log key, separate from status codes
_monotonic = time.monotonic  # module-level so tests can replace the clock


def _suppressed_suffix(suppressed: int) -> str:
    return f"; {suppressed} more since last report" if suppressed else ""


def _status_category(status: int) -> str:
    if status in (401, 403):
        return "API key or access rejected"
    if status == 429:
        return "rate limit or quota exceeded"
    if 500 <= status <= 599:
        return "service error"
    return "request rejected"


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _check_timeout_s(timeout_s: float) -> None:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        raise TypeError(f"timeout_s must be an int or float, got {type(timeout_s).__name__}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be finite and > 0, got {timeout_s}")


def _parse_score(value: object, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RerankError(f"voyage rerank: relevance_score of result {index} is not a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an int too large for a float
        finite = False
    if not finite:
        raise RerankError(f"voyage rerank: relevance_score of result {index} is not finite")
    if value < 0:
        raise RerankError(f"voyage rerank: relevance_score of result {index} is negative")
    return float(value)


def _parse_scores(payload: object, n: int) -> list[float]:
    """Scores in the order of the documents sent. Any deviation raises RerankError."""
    if not isinstance(payload, dict):
        raise RerankError("voyage rerank: response is not a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise RerankError("voyage rerank: response has no 'data' list")
    if len(data) != n:
        raise RerankError(f"voyage rerank: {len(data)} results for {n} documents")
    indices: list[int] = []
    scores: dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            raise RerankError("voyage rerank: a result is not a JSON object")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RerankError("voyage rerank: a result has no int 'index'")
        indices.append(index)
        scores[index] = _parse_score(item.get("relevance_score"), index)
    # n results whose indices are exactly 0..n-1: each document scored once.
    if sorted(indices) != list(range(n)):
        missing = sorted(set(range(n)) - set(indices))
        unexpected = sorted(
            {i for i, count in Counter(indices).items() if count > 1 or not 0 <= i < n}
        )
        raise RerankError(
            f"voyage rerank: indices must cover 0..{n - 1}, "
            f"missing {missing}, duplicated or out of range {unexpected}"
        )
    return [scores[i] for i in range(n)]


def _retrieve_close_error(task: "asyncio.Task[None]") -> None:
    # Mark a close error as retrieved even if every awaiter was cancelled, so asyncio does
    # not log "Task exception was never retrieved". The error stays on the task.
    if not task.cancelled():
        task.exception()


class VoyageReranker:
    """Reranks with Voyage's /v1/rerank. Satisfies `Reranker` structurally."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model: str,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,  # tests only (MockTransport)
    ) -> None:
        if not isinstance(api_key, SecretStr):
            raise TypeError("api_key must be a SecretStr")
        key = api_key.get_secret_value().strip()
        if not key:
            raise ValueError("api_key must not be empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        _check_timeout_s(timeout_s)
        self._model = model.strip()
        self._timeout_s = timeout_s
        # No I/O here: an AsyncClient opens connections only on its first request.
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )
        self._close_task: asyncio.Task[None] | None = None
        # log key (an HTTP status code, or _TOO_MANY_DOCUMENTS) ->
        # (time of the last WARNING, occurrences suppressed since then)
        self._log_state: dict[int | str, tuple[float, int]] = {}

    def __repr__(self) -> str:
        return f"VoyageReranker(model={self._model!r}, timeout_s={self._timeout_s!r})"

    async def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        k: int,
    ) -> list[SearchResult]:
        """See `Reranker.rerank`. Duplicate chunk_ids: the first occurrence is used."""
        _check_k(k)
        unique: dict[str, SearchResult] = {}
        for result in results:
            unique.setdefault(result.chunk_id, result)
        if not unique:
            return []
        candidates = list(unique.values())
        if len(candidates) > _MAX_DOCUMENTS:  # checked before any request is sent
            self._log_too_many_documents(len(candidates))
            raise RerankError(
                f"voyage rerank: {len(candidates)} documents exceed the limit of {_MAX_DOCUMENTS}"
            )
        body = {"query": query, "documents": [r.text for r in candidates], "model": self._model}

        try:
            response = await self._client.post(_RERANK_PATH, json=body)
        except httpx.TimeoutException as exc:  # first: a subclass of TransportError
            raise TimeoutError("voyage rerank timed out") from exc
        except (httpx.TransportError, httpx.DecodingError) as exc:
            raise RerankError(f"voyage rerank transport error: {type(exc).__name__}") from exc

        if response.status_code != 200:
            self._log_http_status(response.status_code)  # rate-limited WARNING
            raise RerankError(f"voyage rerank failed with HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:  # JSONDecodeError / UnicodeDecodeError
            raise RerankError("voyage rerank returned invalid JSON") from exc

        scores = _parse_scores(payload, len(candidates))
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda p: (-p[0], p[1].chunk_id))
        return [r.model_copy(update={"score": s}, deep=True) for s, r in ranked[:k]]

    def _report_now(self, key: int | str) -> int | None:
        """Rate limiter: None = suppress (and count); an int = log now (suppressed count).

        One report per `key` per _LOG_INTERVAL_S: a report is due when at least
        _LOG_INTERVAL_S seconds passed since the last report of that key (exactly 60.0
        counts). The suppressed count is reported with the NEXT occurrence that logs,
        not by a timer: a burst followed by silence leaves its count unreported.
        Safe without a lock: no await between the check and the update.
        """
        now = _monotonic()
        last = self._log_state.get(key)
        if last is not None and now - last[0] < _LOG_INTERVAL_S:
            self._log_state[key] = (last[0], last[1] + 1)
            return None
        self._log_state[key] = (now, 0)
        return last[1] if last is not None else 0

    def _log_http_status(self, status: int) -> None:
        """WARNING with the status code and a fixed category only (never the body, headers,
        key, query or passages), rate-limited per status code."""
        suppressed = self._report_now(status)
        if suppressed is not None:
            logger.warning(
                "voyage rerank: %s (HTTP %d%s)",
                _status_category(status),
                status,
                _suppressed_suffix(suppressed),
            )

    def _log_too_many_documents(self, count: int) -> None:
        """WARNING with the document count and the limit only, rate-limited separately."""
        suppressed = self._report_now(_TOO_MANY_DOCUMENTS)
        if suppressed is not None:
            logger.warning(
                "voyage rerank: %d documents exceed the limit of %d%s",
                count,
                _MAX_DOCUMENTS,
                _suppressed_suffix(suppressed),
            )

    async def aclose(self) -> None:
        """Close the HTTP client once; safe to call again or concurrently.

        Every caller waits until the single close has finished; cancelling one caller
        cancels neither the close nor the other callers. A close error is raised to every
        caller (and never reported as "never retrieved").
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._client.aclose())
            self._close_task.add_done_callback(_retrieve_close_error)
        await asyncio.shield(self._close_task)
