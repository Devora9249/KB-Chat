"""FastAPI application for the Knowledge Center Assistant.

Run locally:
    uv run uvicorn kc.api.app:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from typing import Annotated

from fastapi import Depends, FastAPI

from contracts.generation import Answer, AskRequest
from kc.api.dependencies import get_answerer
from kc.generation.answerer import Answerer


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Center Assistant API",
        version="0.1.0",
        description="Answers questions about Genpact Cora products, with citations.",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check for Docker / deployment."""
        return {"status": "ok"}

    @app.post("/v1/ask")
    async def ask(
        request: AskRequest,
        answerer: Annotated[Answerer, Depends(get_answerer)],
    ) -> Answer:
        """Answer a question from the Knowledge Center, with citations."""
        return await answerer.answer(request)

    return app


app = create_app()
