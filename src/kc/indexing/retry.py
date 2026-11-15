"""Async retry with exponential backoff and optional full jitter.

delay(attempt) = min(max_delay, base_delay * 2 ** attempt); with jitter, a random
value in [0, delay]. Standard library only.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable

SleepFn = Callable[[float], Awaitable[None]]


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    jitter: bool = True,
    sleep: SleepFn = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run `fn` up to `attempts` times; re-raise the last error if all attempts fail.

    Only exceptions in `retry_on` are retried; anything else propagates immediately.
    `sleep` and `rng` are injectable so tests run instantly and deterministically.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("delays must be non-negative")
    rand = rng or random.Random()

    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on:
            if attempt == attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2**attempt)
            if jitter:
                delay = rand.uniform(0.0, delay)
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
