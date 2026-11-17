"""HTTP client for Leah's `/v1/ask` endpoint (the real backend of the UI).

Same signature as `fake_api.ask`, so `app.py` can switch between them via KC_UI_BACKEND.

Errors never crash the UI: every failure (API down, timeout, non-200, invalid body,
invalid question) is raised as a single `ApiError` whose message is safe to show to
the user. Technical details go to the log, not to the chat.
"""

import logging
import os

import httpx
from pydantic import ValidationError

from contracts.generation import Answer, AskRequest

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.environ.get("KC_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
ASK_PATH = "/v1/ask"

# A real answer (retrieval + Claude) can take a while; connecting should not.
REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class ApiError(RuntimeError):
    """The API could not produce an answer. `str(error)` is a user-facing message."""


async def ask(question: str) -> Answer:
    """POST the question to `/v1/ask` and return the validated `Answer`.

    Raises `ApiError` (only) on any failure.
    """
    try:
        request = AskRequest(question=question)
    except ValidationError as error:
        raise ApiError("Please enter a question between 1 and 2000 characters.") from error

    try:
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(ASK_PATH, json=request.model_dump())
    except httpx.ConnectError as error:
        logger.warning("Cannot connect to API at %s: %s", API_BASE_URL, error)
        raise ApiError(
            f"Cannot reach the assistant API at {API_BASE_URL}. Is it running?"
        ) from error
    except httpx.TimeoutException as error:
        logger.warning("API request timed out: %s", error)
        raise ApiError("The assistant took too long to answer. Please try again.") from error
    except httpx.HTTPError as error:
        logger.warning("API request failed: %r", error)
        raise ApiError("Could not contact the assistant API. Please try again.") from error

    if response.status_code != httpx.codes.OK:
        logger.warning("API returned %s: %s", response.status_code, response.text[:500])
        if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
            raise ApiError("The assistant could not accept this question. Please rephrase it.")
        raise ApiError(
            f"The assistant API returned an error (HTTP {response.status_code}). Please try again."
        )

    try:
        return Answer.model_validate_json(response.content)
    except ValidationError as error:
        logger.warning("API returned an invalid Answer: %s", error)
        raise ApiError("The assistant API returned an unexpected response.") from error
