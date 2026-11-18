"""FastAPI application for the Knowledge Center Assistant.

Run locally:
    uv run uvicorn kc.api.app:app --reload
Interactive docs: http://127.0.0.1:8000/docs

Error mapping (agreed with Racheli and Raizy):
- RetrievalError, GenerationError, overall deadline expired -> 503 (temporary; try again)
- anything else                                             -> 500 (a bug; must be fixed)
Responses carry a generic message only; details go to the log, never to the user.
"""

import asyncio
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from contracts.generation import Answer, AskRequest
from contracts.retrieval import RetrievalError
from kc.api.dependencies import get_answerer, get_settings
from kc.api.settings import ApiSettings
from kc.generation.answerer import Answerer
from kc.generation.llm import GenerationError

logger = logging.getLogger(__name__)

SEARCH_UNAVAILABLE = "Search is temporarily unavailable. Please try again."
GENERATION_UNAVAILABLE = "The answer service is temporarily unavailable. Please try again."
DEADLINE_EXCEEDED = "The answer took too long. Please try again."


async def _retrieval_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Retrieval failed for %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": SEARCH_UNAVAILABLE})


async def _generation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Generation failed for %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": GENERATION_UNAVAILABLE})


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Center Assistant API",
        version="0.1.0",
        description="Answers questions about Genpact Cora products, with citations.",
    )
    app.add_exception_handler(RetrievalError, _retrieval_error_handler)
    app.add_exception_handler(GenerationError, _generation_error_handler)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check for Docker / deployment."""
        return {"status": "ok"}

    @app.post("/v1/ask")
    async def ask(
        request: AskRequest,
        answerer: Annotated[Answerer, Depends(get_answerer)],
        settings: Annotated[ApiSettings, Depends(get_settings)],
    ) -> Answer:
        """Answer a question from the Knowledge Center, with citations."""
        try:
            async with asyncio.timeout(settings.request_timeout_s) as deadline:
                return await answerer.answer(request)
        except TimeoutError:
            if not deadline.expired():
                raise  # a TimeoutError from deeper code is not our deadline: a bug -> 500
            logger.warning("Deadline of %ss exceeded for /v1/ask", settings.request_timeout_s)
            raise HTTPException(status_code=503, detail=DEADLINE_EXCEEDED) from None

    return app


app = create_app()
