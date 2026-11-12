import asyncio
import random

import pytest

from kc.indexing.retry import retry_async


class Flaky:
    """Fails `failures` times with `error`, then returns "ok"."""

    def __init__(self, failures: int, error: type[Exception] = ConnectionError) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("boom")
        return "ok"


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_succeeds_first_try_without_sleeping() -> None:
    fn, sleep = Flaky(0), RecordingSleep()
    assert asyncio.run(retry_async(fn, sleep=sleep)) == "ok"
    assert fn.calls == 1
    assert sleep.delays == []


def test_retries_with_exponential_backoff() -> None:
    fn, sleep = Flaky(3), RecordingSleep()
    result = asyncio.run(retry_async(fn, attempts=4, base_delay=0.5, jitter=False, sleep=sleep))
    assert result == "ok"
    assert fn.calls == 4
    assert sleep.delays == [0.5, 1.0, 2.0]


def test_delay_is_capped_by_max_delay() -> None:
    fn, sleep = Flaky(4), RecordingSleep()
    asyncio.run(
        retry_async(fn, attempts=5, base_delay=1.0, max_delay=3.0, jitter=False, sleep=sleep)
    )
    assert sleep.delays == [1.0, 2.0, 3.0, 3.0]


def test_jitter_stays_within_bounds_and_is_reproducible_with_seed() -> None:
    def run() -> list[float]:
        sleep = RecordingSleep()
        asyncio.run(
            retry_async(Flaky(3), attempts=4, base_delay=1.0, sleep=sleep, rng=random.Random(42))
        )
        return sleep.delays

    first, second = run(), run()
    assert first == second
    for delay, cap in zip(first, [1.0, 2.0, 4.0], strict=True):
        assert 0.0 <= delay <= cap


def test_raises_last_error_when_all_attempts_fail() -> None:
    fn, sleep = Flaky(10), RecordingSleep()
    with pytest.raises(ConnectionError):
        asyncio.run(retry_async(fn, attempts=3, jitter=False, sleep=sleep))
    assert fn.calls == 3
    assert len(sleep.delays) == 2  # no sleep after the final failure


def test_non_retryable_error_propagates_immediately() -> None:
    fn, sleep = Flaky(1, error=ValueError), RecordingSleep()
    with pytest.raises(ValueError):
        asyncio.run(retry_async(fn, retry_on=(ConnectionError,), sleep=sleep))
    assert fn.calls == 1
    assert sleep.delays == []


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        asyncio.run(retry_async(Flaky(0), attempts=0))
    with pytest.raises(ValueError):
        asyncio.run(retry_async(Flaky(0), base_delay=-1.0))
