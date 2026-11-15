"""The interface every answerer implements (fake now, Claude-based later).

Internal to the Generation domain: the API depends on this Protocol, never on a
concrete class, so implementations can be swapped via dependency injection.
"""

from typing import Protocol

from contracts.generation import Answer, AskRequest


class Answerer(Protocol):
    async def answer(self, request: AskRequest) -> Answer: ...
