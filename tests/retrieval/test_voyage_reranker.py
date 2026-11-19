"""Tests for VoyageReranker (Search & Ranking).

No network: every test uses httpx.MockTransport, and an autouse fixture makes any call
through the real transport fail the test.
"""

import asyncio
import gc
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from contracts.retrieval import Filters, SearchResult
from contracts.storage import InputType
from kc.retrieval.hybrid_retriever import HybridRetriever
from kc.retrieval.reranker import Reranker, RerankError
from kc.retrieval.voyage_reranker import VoyageReranker

FAKE_KEY = "test-key-not-real"
MODEL = "rerank-3-lite"
URL = "https://api.voyageai.com/v1/rerank"
LONG = 10.0

Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"real network call to {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)


def _item(chunk_id: str, text: str | None = None, **overrides: Any) -> SearchResult:
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "url": f"https://knowledgecenter.gcora.genpact.com/help/{chunk_id}",
        "title": f"Title {chunk_id}",
        "text": text if text is not None else f"text {chunk_id}",
        "score": 0.5,
        "product": "sequence",
        "version": "9",
        "doc_type": "how-to",
        "heading_path": ["Section", chunk_id],
    }
    base.update(overrides)
    return SearchResult(**base)


def _json_response(payload: Any, status: int = 200) -> httpx.Response:
    # json.dumps (allow_nan=True) so tests can send NaN / Infinity like a broken provider.
    return httpx.Response(
        status, content=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )


def _scored(*scores: float) -> dict[str, Any]:
    return {"data": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


class Recorder:
    """MockTransport handler that records every request and answers with `respond`."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)


def _reranker(
    handler: Handler, transport: httpx.AsyncBaseTransport | None = None
) -> VoyageReranker:
    return VoyageReranker(
        SecretStr(FAKE_KEY),
        model=MODEL,
        timeout_s=3.0,
        transport=transport if transport is not None else httpx.MockTransport(handler),
    )


def _rerank(
    handler: Handler, query: str, results: list[SearchResult], k: int = 10
) -> list[SearchResult]:
    async def scenario() -> list[SearchResult]:
        reranker = _reranker(handler)
        try:
            return await reranker.rerank(query, results, k=k)
        finally:
            await reranker.aclose()

    return asyncio.run(scenario())


def _error(handler: Handler, results: list[SearchResult] | None = None) -> BaseException:
    with pytest.raises(BaseException) as info:
        _rerank(handler, "saml", results if results is not None else [_item("a"), _item("b")])
    return info.value


ABC = [_item("a"), _item("b"), _item("c")]


# --- building is separate from using: no request without rerank ---------------------------


def test_construction_and_close_send_no_request() -> None:
    recorder = Recorder(lambda r: _json_response(_scored()))

    async def scenario() -> None:
        reranker = _reranker(recorder)
        await reranker.aclose()

    asyncio.run(scenario())
    assert recorder.requests == []


@pytest.mark.parametrize("results", [[], ()])
def test_empty_input_returns_empty_list_without_a_request(results: Any) -> None:
    recorder = Recorder(lambda r: _json_response(_scored()))
    assert _rerank(recorder, "saml", results) == []
    assert recorder.requests == []


@pytest.mark.parametrize(("k", "error"), [(True, TypeError), (5.0, TypeError), (0, ValueError)])
def test_invalid_k_raises_before_any_request(k: Any, error: type[Exception]) -> None:
    recorder = Recorder(lambda r: _json_response(_scored(1.0)))
    with pytest.raises(error):
        _rerank(recorder, "saml", [_item("a")], k=k)
    assert recorder.requests == []


def test_one_rerank_sends_exactly_one_request() -> None:
    recorder = Recorder(lambda r: _json_response(_scored(0.1, 0.2, 0.3)))
    _rerank(recorder, "saml", ABC)
    assert len(recorder.requests) == 1


# --- the request -----------------------------------------------------------------------------


def test_request_method_url_auth_and_body() -> None:
    recorder = Recorder(lambda r: _json_response(_scored(0.1, 0.2)))
    _rerank(recorder, "configure saml", [_item("a", "first text"), _item("b", "second text")])

    (request,) = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == URL
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert json.loads(request.content) == {
        "query": "configure saml",
        "documents": ["first text", "second text"],
        "model": MODEL,
    }  # no top_k: the cut to k is local


# --- success -----------------------------------------------------------------------------------


def test_scores_map_by_index_and_sort_descending() -> None:
    payload = {
        "data": [  # any order: mapped by index
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.2},
        ]
    }
    results = _rerank(lambda r: _json_response(payload), "saml", ABC)
    assert [(r.chunk_id, r.score) for r in results] == [("b", 0.9), ("c", 0.5), ("a", 0.2)]


@pytest.mark.parametrize(("k", "expected"), [(1, ["b"]), (2, ["b", "c"]), (3, ["b", "c", "a"])])
def test_output_is_cut_to_k(k: int, expected: list[str]) -> None:
    results = _rerank(lambda r: _json_response(_scored(0.2, 0.9, 0.5)), "saml", ABC, k=k)
    assert [r.chunk_id for r in results] == expected


def test_ties_are_ordered_by_chunk_id_and_zero_scores_are_kept() -> None:
    items = [_item("c"), _item("a"), _item("b")]
    results = _rerank(lambda r: _json_response(_scored(0.0, 0.0, 0.0)), "saml", items)
    assert [(r.chunk_id, r.score) for r in results] == [("a", 0.0), ("b", 0.0), ("c", 0.0)]


def test_int_scores_become_floats() -> None:
    results = _rerank(lambda r: _json_response(_scored(3, 1)), "saml", ABC[:2])
    assert [(r.chunk_id, r.score) for r in results] == [("a", 3.0), ("b", 1.0)]
    assert all(type(r.score) is float for r in results)


def test_other_fields_are_kept_and_outputs_are_new_objects() -> None:
    items = [_item("a", product="orchestration", version=None, doc_type="release-notes")]
    before = items[0].model_copy(deep=True)
    (result,) = _rerank(lambda r: _json_response(_scored(0.7)), "saml", items)

    assert result.model_dump(exclude={"score"}) == items[0].model_dump(exclude={"score"})
    assert result.score == 0.7
    assert result is not items[0]
    assert result.heading_path is not items[0].heading_path
    assert items[0] == before  # the input is not modified


def test_duplicate_chunk_id_sends_and_returns_the_first_occurrence() -> None:
    recorder = Recorder(lambda r: _json_response(_scored(0.1, 0.2)))
    items = [_item("a", "first"), _item("b", "bee"), _item("a", "later")]
    results = _rerank(recorder, "saml", items)

    assert json.loads(recorder.requests[0].content)["documents"] == ["first", "bee"]
    assert sorted(r.chunk_id for r in results) == ["a", "b"]
    assert next(r for r in results if r.chunk_id == "a").text == "first"


def test_satisfies_the_reranker_protocol() -> None:
    async def scenario() -> None:
        reranker: Reranker = _reranker(lambda r: _json_response(_scored()))  # mypy check
        assert isinstance(reranker, Reranker)
        assert isinstance(reranker, VoyageReranker)
        await reranker.aclose()

    asyncio.run(scenario())


# --- error mapping ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeout", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_httpx_timeouts_become_timeout_error(timeout: type[httpx.TimeoutException]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise timeout("slow", request=request)

    exc = _error(handler)
    assert type(exc) is TimeoutError
    assert isinstance(exc.__cause__, timeout)


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError])
def test_transport_errors_become_rerank_error(error: type[httpx.TransportError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("down", request=request)

    exc = _error(handler)
    assert type(exc) is RerankError
    assert error.__name__ in str(exc)
    assert isinstance(exc.__cause__, error)


def test_decoding_error_becomes_rerank_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.DecodingError("bad gzip", request=request)

    exc = _error(handler)
    assert type(exc) is RerankError
    assert isinstance(exc.__cause__, httpx.DecodingError)


@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 429, 500, 502, 503])
def test_non_200_status_becomes_rerank_error(status: int) -> None:
    exc = _error(lambda r: _json_response({"detail": f"error with {FAKE_KEY}"}, status=status))
    assert type(exc) is RerankError
    assert f"HTTP {status}" in str(exc)
    assert FAKE_KEY not in str(exc)  # the body is never copied into the message


@pytest.mark.parametrize("content", [b"not json", b"", b"\xff\xfe\x00"])
def test_invalid_json_becomes_rerank_error(content: bytes) -> None:
    exc = _error(lambda r: httpx.Response(200, content=content))
    assert type(exc) is RerankError
    assert "invalid JSON" in str(exc)


def _entry(index: Any, score: Any = 0.5) -> dict[str, Any]:
    return {"index": index, "relevance_score": score}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "not a JSON object"),
        ("text", "not a JSON object"),
        ({}, "no 'data' list"),
        ({"data": "x"}, "no 'data' list"),
        ({"data": [_entry(0), _entry(1)]}, "2 results for 3 documents"),
        ({"data": [_entry(0), _entry(1), _entry(2), _entry(3)]}, "4 results for 3 documents"),
        ({"data": [_entry(0), _entry(1), "x"]}, "a result is not a JSON object"),
        ({"data": [_entry(0), _entry(1), {"relevance_score": 0.5}]}, "no int 'index'"),
        ({"data": [_entry(0), _entry(1), _entry(True)]}, "no int 'index'"),
        ({"data": [_entry(0), _entry(1), _entry("2")]}, "no int 'index'"),
        ({"data": [_entry(0), _entry(1), _entry(2.0)]}, "no int 'index'"),
        # indices must be exactly 0..n-1
        ({"data": [_entry(0), _entry(0), _entry(1)]}, r"missing \[2\], duplicated or out of"),
        ({"data": [_entry(0), _entry(1), _entry(3)]}, r"missing \[2\], .* range \[3\]"),
        ({"data": [_entry(1), _entry(2), _entry(3)]}, r"missing \[0\], .* range \[3\]"),
        ({"data": [_entry(-1), _entry(1), _entry(2)]}, r"missing \[0\], .* range \[-1\]"),
        # scores must be finite numbers >= 0
        ({"data": [_entry(0), _entry(1), _entry(2, -0.1)]}, "result 2 is negative"),
        ({"data": [_entry(0), _entry(1), _entry(2, math.nan)]}, "result 2 is not finite"),
        ({"data": [_entry(0), _entry(1), _entry(2, math.inf)]}, "result 2 is not finite"),
        ({"data": [_entry(0), _entry(1), _entry(2, -math.inf)]}, "result 2 is not finite"),
        ({"data": [_entry(0), _entry(1), _entry(2, 10**400)]}, "result 2 is not finite"),
        ({"data": [_entry(0), _entry(1), _entry(2, None)]}, "result 2 is not a number"),
        ({"data": [_entry(0), _entry(1), _entry(2, True)]}, "result 2 is not a number"),
        ({"data": [_entry(0), _entry(1), _entry(2, "0.5")]}, "result 2 is not a number"),
        ({"data": [_entry(0), _entry(1), {"index": 2}]}, "result 2 is not a number"),
    ],
)
def test_response_that_breaks_the_schema_becomes_rerank_error(payload: Any, message: str) -> None:
    exc = _error(lambda r: _json_response(payload), ABC)
    assert type(exc) is RerankError
    assert re.search(message, str(exc)), str(exc)


# --- bugs are not hidden ---------------------------------------------------------------------


def test_non_search_result_input_raises_attribute_error() -> None:
    bad: list[Any] = [object()]
    exc = _error(lambda r: _json_response(_scored(1.0)), bad)
    assert type(exc) is AttributeError


@pytest.mark.parametrize("bug", [ValueError("bug"), KeyError("bug"), TypeError("bug")])
def test_unexpected_handler_error_propagates_unchanged(bug: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise bug

    assert _error(handler) is bug


def test_rerank_after_aclose_raises_runtime_error() -> None:
    async def scenario() -> None:
        reranker = _reranker(lambda r: _json_response(_scored(1.0)))
        await reranker.aclose()
        with pytest.raises(RuntimeError):
            await reranker.rerank("saml", [_item("a")], k=1)

    asyncio.run(scenario())


# --- cancellation ---------------------------------------------------------------------------


def test_cancel_during_the_request_is_not_wrapped() -> None:
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.sleep(LONG)
        return _json_response(_scored(1.0))

    async def scenario() -> None:
        reranker = _reranker(handler)
        task = asyncio.create_task(reranker.rerank("saml", [_item("a")], k=1))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()  # not RerankError, not TimeoutError
        await reranker.aclose()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


# --- the secret ------------------------------------------------------------------------------


def test_repr_does_not_contain_the_key() -> None:
    async def scenario() -> str:
        reranker = _reranker(lambda r: _json_response(_scored()))
        text = repr(reranker)
        await reranker.aclose()
        return text

    text = asyncio.run(scenario())
    assert FAKE_KEY not in text
    assert MODEL in text


def test_error_messages_never_contain_the_key() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    handlers: list[Handler] = [
        timeout,
        lambda r: _json_response({}, status=401),
        lambda r: httpx.Response(200, content=b"not json"),
        lambda r: _json_response({"data": []}),
    ]
    for handler in handlers:
        exc = _error(handler)
        assert FAKE_KEY not in str(exc)
        assert FAKE_KEY not in repr(exc)


@pytest.mark.parametrize("key", ["", "   "])
def test_blank_key_raises_value_error(key: str) -> None:
    with pytest.raises(ValueError, match="api_key must not be empty"):
        VoyageReranker(SecretStr(key), model=MODEL, timeout_s=3.0)


def test_plain_string_key_raises_type_error_without_echoing_it() -> None:
    plain: Any = "plain-secret-value"
    with pytest.raises(TypeError) as info:
        VoyageReranker(plain, model=MODEL, timeout_s=3.0)
    assert "plain-secret-value" not in str(info.value)


@pytest.mark.parametrize("model", ["", "   "])
def test_blank_model_raises_value_error(model: str) -> None:
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        VoyageReranker(SecretStr(FAKE_KEY), model=model, timeout_s=3.0)


@pytest.mark.parametrize(
    ("timeout_s", "error"),
    [
        (0, ValueError),
        (-1.0, ValueError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (True, TypeError),
        ("3", TypeError),
    ],
)
def test_invalid_timeout_raises(timeout_s: Any, error: type[Exception]) -> None:
    with pytest.raises(error, match="timeout_s"):
        VoyageReranker(SecretStr(FAKE_KEY), model=MODEL, timeout_s=timeout_s)


# --- lifecycle: aclose ------------------------------------------------------------------------


class ClosingTransport(httpx.MockTransport):
    """MockTransport whose aclose can wait for a gate and/or fail; counts the closes."""

    def __init__(
        self, *, gate: asyncio.Event | None = None, error: Exception | None = None
    ) -> None:
        super().__init__(lambda r: _json_response(_scored()))
        self.gate = gate
        self.error = error
        self.closes = 0
        self.close_started = asyncio.Event()

    async def aclose(self) -> None:
        self.closes += 1
        self.close_started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error


def test_aclose_closes_the_client_and_is_safe_to_repeat() -> None:
    async def scenario() -> None:
        transport = ClosingTransport()
        reranker = _reranker(lambda r: _json_response(_scored()), transport)
        await reranker.aclose()
        await reranker.aclose()
        assert reranker._client.is_closed
        assert transport.closes == 1

    asyncio.run(scenario())


def test_concurrent_aclose_closes_once_and_everyone_waits_for_it() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        transport = ClosingTransport(gate=gate)
        reranker = _reranker(lambda r: _json_response(_scored()), transport)
        callers = [asyncio.create_task(reranker.aclose()) for _ in range(3)]
        await asyncio.wait_for(transport.close_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert not any(c.done() for c in callers)  # nobody returns before the close ends

        gate.set()
        assert await asyncio.gather(*callers) == [None, None, None]
        assert transport.closes == 1
        assert reranker._client.is_closed

    asyncio.run(scenario())


def test_cancelling_one_caller_does_not_cancel_the_close_or_the_others() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        transport = ClosingTransport(gate=gate)
        reranker = _reranker(lambda r: _json_response(_scored()), transport)
        first, second = (asyncio.create_task(reranker.aclose()) for _ in range(2))
        await asyncio.wait_for(transport.close_started.wait(), timeout=1.0)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        await second  # completes normally
        assert transport.closes == 1
        assert reranker._client.is_closed

    asyncio.run(scenario())


def test_close_error_reaches_every_caller_and_later_calls() -> None:
    async def scenario() -> None:
        error = OSError("close failed")
        gate = asyncio.Event()
        transport = ClosingTransport(gate=gate, error=error)
        reranker = _reranker(lambda r: _json_response(_scored()), transport)
        callers = [asyncio.create_task(reranker.aclose()) for _ in range(2)]
        await asyncio.wait_for(transport.close_started.wait(), timeout=1.0)
        gate.set()
        outcomes = await asyncio.gather(*callers, return_exceptions=True)
        assert all(o is error for o in outcomes)
        with pytest.raises(OSError) as info:
            await reranker.aclose()
        assert info.value is error
        assert transport.closes == 1

    asyncio.run(scenario())


def test_close_error_is_never_reported_as_unretrieved_when_all_callers_are_cancelled() -> None:
    async def scenario() -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: reports.append(ctx))

        gate = asyncio.Event()
        transport = ClosingTransport(gate=gate, error=OSError("close failed"))
        reranker = _reranker(lambda r: _json_response(_scored()), transport)
        callers = [asyncio.create_task(reranker.aclose()) for _ in range(2)]
        await asyncio.wait_for(transport.close_started.wait(), timeout=1.0)
        for caller in callers:
            caller.cancel()
        for caller in callers:
            with pytest.raises(asyncio.CancelledError):
                await caller
        gate.set()  # the close now fails, with nobody waiting
        close_task = reranker._close_task
        assert close_task is not None
        await asyncio.wait([close_task])  # waits without retrieving the error
        assert transport.closes == 1
        del close_task, reranker, callers
        gc.collect()
        found = list(reports)

        # Sensitivity check: an unretrieved task error IS reported by this setup.
        async def fails() -> None:
            raise OSError("unretrieved")

        task = asyncio.create_task(fails())
        await asyncio.wait([task])  # done, error not retrieved
        del task
        gc.collect()
        assert len(reports) == len(found) + 1
        assert "never retrieved" in reports[-1]["message"]
        return found

    assert asyncio.run(scenario()) == []


# --- integration with HybridRetriever -------------------------------------------------------


class _Embedder:
    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dimension(self) -> int:
        return 1

    async def embed(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        return [[1.0] for _ in texts]


class _Index:
    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return [r.model_copy(deep=True) for r in ABC]  # RRF order: a, b, c

    async def text_search(
        self, query: str, filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return []


def _search_with(handler: Handler, query: str = "secret customer question") -> list[SearchResult]:
    async def scenario() -> list[SearchResult]:
        reranker = _reranker(handler)
        try:
            retriever = HybridRetriever(_Embedder(), _Index(), reranker=reranker)
            return await retriever.search(query, k=3)
        finally:
            await reranker.aclose()

    return asyncio.run(scenario())


def test_hybrid_retriever_follows_voyage_scores() -> None:
    results = _search_with(lambda r: _json_response(_scored(0.1, 0.5, 0.9)))
    assert [(r.chunk_id, r.score) for r in results] == [("c", 0.9), ("b", 0.5), ("a", 0.1)]


@pytest.mark.parametrize(
    "respond",
    [
        lambda r: _json_response({}, status=500),
        lambda r: _json_response({"data": [_entry(0)]}),
    ],
)
def test_hybrid_retriever_falls_back_on_voyage_failure(
    respond: Callable[[httpx.Request], httpx.Response], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        results = _search_with(respond)
    assert [r.chunk_id for r in results] == ["a", "b", "c"]  # the RRF ranking
    assert "RerankError" in caplog.text
    assert FAKE_KEY not in caplog.text
    assert "secret customer question" not in caplog.text


def test_hybrid_retriever_raises_a_handler_bug() -> None:
    bug = ValueError("bug")

    def handler(request: httpx.Request) -> httpx.Response:
        raise bug

    with pytest.raises(ValueError) as info:
        _search_with(handler)
    assert info.value is bug


# --- HTTP status WARNING: status code only, rate-limited per code ---------------------------

LOGGER = "kc.retrieval.voyage_reranker"
QUERY_MARKER = "secret-query-marker"
PASSAGE_MARKER = "secret-passage-marker"
BODY_MARKER = "secret-body-marker"
# Lowercase on purpose: compared with caplog.text.lower(), so "Authorization" and
# "authorization" (and "Bearer" / "bearer") are both caught.
SENSITIVE = (
    FAKE_KEY.lower(),
    QUERY_MARKER,
    PASSAGE_MARKER,
    BODY_MARKER,
    "bearer",
    "authorization",
)


def _assert_nothing_sensitive(caplog: pytest.LogCaptureFixture) -> None:
    text = caplog.text.lower()
    for marker in SENSITIVE:
        assert marker not in text, marker


def _failing_status(status: int) -> Handler:
    # The body deliberately carries sensitive text: it must never reach the log.
    return lambda r: _json_response({"detail": f"{BODY_MARKER} {FAKE_KEY}"}, status=status)


def _voyage_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER]


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (401, "API key or access rejected"),
        (403, "API key or access rejected"),
        (429, "rate limit or quota exceeded"),
        (500, "service error"),
        (503, "service error"),
        (400, "request rejected"),
        (302, "request rejected"),
    ],
)
def test_http_status_is_logged_with_the_code_and_category_only(
    status: int, category: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Captures every logger at DEBUG. Coverage limit: MockTransport bypasses httpcore, so
    # this checks our logger and httpx's, NOT httpcore's (which a real transport would use).
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RerankError, match=f"failed with HTTP {status}$"):
            _rerank(_failing_status(status), QUERY_MARKER, [_item("a", PASSAGE_MARKER)])

    (record,) = _voyage_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.getMessage() == f"voyage rerank: {category} (HTTP {status})"
    assert record.exc_info is None
    # httpx's own "HTTP Request: ..." record was really captured, so the check below
    # covers it (not vacuous).
    assert any(r.name == "httpx" for r in caplog.records)
    _assert_nothing_sensitive(caplog)


CAP = 0  # marker for a document-cap event in the rate-limiting sequence (not a status code)


def test_status_warnings_are_rate_limited_per_status_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = [1000.0]
    monkeypatch.setattr("kc.retrieval.voyage_reranker._monotonic", lambda: clock[0])
    status = [401]

    async def scenario() -> None:
        reranker = _reranker(lambda r: _json_response({}, status=status[0]))
        try:
            # (time, event) -> expected. CAP = a document-cap event (1001 documents, no
            # request) inside the 401 window: it must change neither the 401 window's
            # start time nor its suppressed count.
            for now, event in [
                (1000.0, 401),  # first 401: logged
                (1010.0, 401),  # suppressed (1)
                (1015.0, CAP),  # cap key: logged on its own, 401 state untouched
                (1020.0, 401),  # still suppressed (2): the cap event did not reset the window
                (1030.0, 429),  # another code: logged, separate window
                (1040.0, CAP),  # cap key: suppressed in its own window (from 1015.0)
                (1059.9, 401),  # 59.9 s after the last 401 report: suppressed (3)
                (1060.0, 401),  # exactly 60.0 s after 1000.0: logged, "3 more" (no cap events)
                (1061.0, 401),  # suppressed (1), new window from 1060.0
                (1120.0, 401),  # exactly 60.0 s again: logged, "1 more"
            ]:
                clock[0] = now
                with pytest.raises(RerankError):
                    if event == CAP:
                        await reranker.rerank("q", _many(1001), k=1)
                    else:
                        status[0] = event
                        await reranker.rerank("q", [_item("a")], k=1)
        finally:
            await reranker.aclose()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(scenario())

    assert [r.getMessage() for r in _voyage_records(caplog)] == [
        "voyage rerank: API key or access rejected (HTTP 401)",
        "voyage rerank: 1001 documents exceed the limit of 1000",
        "voyage rerank: rate limit or quota exceeded (HTTP 429)",
        "voyage rerank: API key or access rejected (HTTP 401; 3 more since last report)",
        "voyage rerank: API key or access rejected (HTTP 401; 1 more since last report)",
    ]


def _raises_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow", request=request)


def _raises_connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("down", request=request)


def test_success_returns_results_and_logs_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        results = _rerank(lambda r: _json_response(_scored(0.5)), "q", [_item("a")])
    assert [(r.chunk_id, r.score) for r in results] == [("a", 0.5)]
    assert _voyage_records(caplog) == []


def test_timeout_raises_timeout_error_and_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(TimeoutError):
            _rerank(_raises_timeout, "q", [_item("a")])
    assert _voyage_records(caplog) == []


def test_transport_error_raises_rerank_error_and_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(RerankError, match="transport error: ConnectError"):
            _rerank(_raises_connect_error, "q", [_item("a")])
    assert _voyage_records(caplog) == []


# --- document limit: 1,000 unique documents, checked before sending ------------------------


def _many(n: int) -> list[SearchResult]:
    return [_item(f"c{i:04d}") for i in range(n)]


def test_more_than_1000_unique_documents_raise_without_a_request() -> None:
    recorder = Recorder(lambda r: _json_response(_scored()))
    with pytest.raises(RerankError, match="1001 documents exceed the limit of 1000"):
        _rerank(recorder, "q", _many(1001))
    assert recorder.requests == []


def test_too_many_documents_is_logged_with_count_and_limit_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    items = [_item(f"c{i:04d}", PASSAGE_MARKER) for i in range(1001)]
    with caplog.at_level(logging.DEBUG):  # every logger (see the httpcore note above)
        with pytest.raises(RerankError, match="1001 documents exceed the limit of 1000"):
            _rerank(lambda r: _json_response(_scored()), QUERY_MARKER, items)

    (record,) = _voyage_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "voyage rerank: 1001 documents exceed the limit of 1000"
    assert record.exc_info is None
    _assert_nothing_sensitive(caplog)


def test_too_many_documents_warning_is_rate_limited_separately(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = [1000.0]
    monkeypatch.setattr("kc.retrieval.voyage_reranker._monotonic", lambda: clock[0])

    async def scenario() -> None:
        reranker = _reranker(lambda r: _json_response({}, status=401))
        try:
            for now, items in [
                (1000.0, _many(1001)),  # logged
                (1010.0, _many(1002)),  # suppressed (1)
                (1020.0, [_item("a")]),  # HTTP 401: its own key, logged
                (1059.9, _many(1001)),  # suppressed (2)
                (1060.0, _many(1003)),  # exactly 60.0 s: logged, "2 more"
            ]:
                clock[0] = now
                with pytest.raises(RerankError):
                    await reranker.rerank("q", items, k=1)
        finally:
            await reranker.aclose()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        asyncio.run(scenario())

    assert [r.getMessage() for r in _voyage_records(caplog)] == [
        "voyage rerank: 1001 documents exceed the limit of 1000",
        "voyage rerank: API key or access rejected (HTTP 401)",
        "voyage rerank: 1003 documents exceed the limit of 1000; 2 more since last report",
    ]


def test_exactly_1000_documents_are_sent() -> None:
    recorder = Recorder(lambda r: _json_response(_scored(*[0.5] * 1000)))
    assert len(_rerank(recorder, "q", _many(1000), k=5)) == 5
    (request,) = recorder.requests
    assert len(json.loads(request.content)["documents"]) == 1000


def test_limit_counts_unique_documents_after_removing_duplicates() -> None:
    recorder = Recorder(lambda r: _json_response(_scored(*[0.5] * 1000)))
    items = [*_many(1000), _item("c0000")]  # 1001 items, 1000 unique
    _rerank(recorder, "q", items, k=1)
    assert len(json.loads(recorder.requests[0].content)["documents"]) == 1000


@pytest.mark.parametrize(("k", "error"), [(True, TypeError), (0, ValueError)])
def test_k_is_still_checked_before_the_document_limit(k: Any, error: type[Exception]) -> None:
    recorder = Recorder(lambda r: _json_response(_scored()))
    with pytest.raises(error):
        _rerank(recorder, "q", _many(1001), k=k)
    assert recorder.requests == []


class _ManyIndex(_Index):
    def __init__(self, items: list[SearchResult]) -> None:
        self._items = items

    async def vector_search(
        self, vector: list[float], filters: Filters | None = None, *, k: int = 10
    ) -> list[SearchResult]:
        return [r.model_copy(deep=True) for r in self._items]


def test_hybrid_retriever_falls_back_when_the_pool_exceeds_the_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = Recorder(lambda r: _json_response(_scored()))
    items = _many(1001)

    async def scenario() -> list[SearchResult]:
        reranker = _reranker(recorder)
        try:
            retriever = HybridRetriever(
                _Embedder(), _ManyIndex(items), reranker=reranker, rerank_candidates=1001
            )
            return await retriever.search("q", k=3)
        finally:
            await reranker.aclose()

    with caplog.at_level(logging.WARNING):
        results = asyncio.run(scenario())

    assert [r.chunk_id for r in results] == ["c0000", "c0001", "c0002"]  # the RRF ranking
    assert recorder.requests == []
    assert "RerankError" in caplog.text


# --- concurrency and UTF-8 ------------------------------------------------------------------


def test_concurrent_reranks_are_all_in_flight_together_and_never_mix() -> None:
    n = 10

    async def scenario() -> None:
        in_flight = 0
        peak = 0
        all_in_flight = asyncio.Event()
        queries: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            if in_flight == n:
                all_in_flight.set()
            # Every request waits until ALL n are in flight at the same time; a request
            # that is not concurrent with the others times out and fails the test.
            await asyncio.wait_for(all_in_flight.wait(), timeout=2.0)
            body = json.loads(request.content)
            queries.append(body["query"])
            in_flight -= 1
            # Later documents score higher: each task gets its own documents reversed.
            return _json_response(_scored(*[float(i) for i in range(len(body["documents"]))]))

        reranker = _reranker(handler)
        try:
            inputs = [[_item(f"t{t}-{i}") for i in range(3)] for t in range(n)]
            outputs = await asyncio.gather(
                *(reranker.rerank(f"query {t}", inputs[t], k=3) for t in range(n))
            )
            assert peak == n  # proof: all n requests were in flight simultaneously
            assert sorted(queries) == sorted(f"query {t}" for t in range(n))
            for t, output in enumerate(outputs):
                assert [r.chunk_id for r in output] == [f"t{t}-2", f"t{t}-1", f"t{t}-0"]
            # The instance still works after the concurrent calls.
            all_in_flight.clear()
            in_flight, peak = n - 1, 0  # let a single request pass the barrier
            (only,) = await reranker.rerank("again", [_item("z")], k=1)
            assert only.chunk_id == "z"
        finally:
            await reranker.aclose()

    asyncio.run(scenario())


def test_utf8_and_hebrew_are_sent_and_returned_unchanged() -> None:
    query = "איך מגדירים SAML ב-SeQuence 9?"
    texts = ["הגדרת SAML ב-SeQuence 9", "Ünïcödé — café ✓", "שלום עולם"]
    recorder = Recorder(lambda r: _json_response(_scored(0.1, 0.3, 0.2)))
    items = [_item(f"h{i}", text) for i, text in enumerate(texts)]

    results = _rerank(recorder, query, items)

    (request,) = recorder.requests
    assert request.headers["content-type"] == "application/json"
    assert "איך מגדירים".encode() in request.content  # raw UTF-8, not \u escapes
    body = json.loads(request.content.decode("utf-8"))
    assert body["query"] == query
    assert body["documents"] == texts
    assert [(r.chunk_id, r.text) for r in results] == [
        ("h1", texts[1]),
        ("h2", texts[2]),
        ("h0", texts[0]),
    ]
