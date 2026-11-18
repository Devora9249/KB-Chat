"""API settings, read from environment variables or the local `.env` file.

Environment variables:
    API_ANSWERER           "fake" (default): canned FakeAnswerer, every status triggerable
                           "rag": RagAnswerer (retriever + model, real grounding logic)
    API_REQUEST_TIMEOUT_S  overall deadline for one /v1/ask call, in seconds (default 45;
                           keep it below the UI's 60s client timeout)
    API_RAG_K              passages retrieved per question for "rag" (default 5)
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    answerer: Literal["fake", "rag"] = "fake"
    request_timeout_s: float = Field(default=45.0, gt=0, allow_inf_nan=False)
    rag_k: int = Field(default=5, ge=1)
