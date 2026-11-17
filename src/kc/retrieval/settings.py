"""Search settings, read from environment variables or the local `.env` file.

Same pattern as `kc.db.settings`: one settings class per domain, a `load_...()` helper, and
a factory that receives the settings (dependency injection, no global).

Environment variables (defaults come from `hybrid_retriever`, not repeated here):
    RETRIEVAL_CANDIDATE_K  candidates requested from EACH search before fusion (int >= 1)
    RETRIEVAL_TIMEOUT_S    time budget for one search call, in seconds (finite, > 0)
"""

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from contracts.storage import Embedder, SearchIndex
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_TIMEOUT_S,
    HybridRetriever,
)


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    candidate_k: int = Field(
        default=DEFAULT_CANDIDATE_K, ge=1, validation_alias="RETRIEVAL_CANDIDATE_K"
    )
    timeout_s: float = Field(
        default=DEFAULT_TIMEOUT_S, gt=0, allow_inf_nan=False, validation_alias="RETRIEVAL_TIMEOUT_S"
    )

    # pydantic's default (lax) mode would accept True as 1, and 5.0 or "5.0" as 5. The
    # retriever contract rejects them, so reject them here too, before pydantic converts.
    @field_validator("candidate_k", mode="before")
    @classmethod
    def _candidate_k_must_be_a_whole_int(cls, value: Any) -> Any:
        if isinstance(value, bool | float):
            raise ValueError(f"candidate_k must be an int, got {type(value).__name__}")
        if isinstance(value, str):
            try:
                return int(value)  # "5.0" and "abc" are rejected here
            except ValueError:
                raise ValueError(f"candidate_k must be an int, got {value!r}") from None
        return value

    @field_validator("timeout_s", mode="before")
    @classmethod
    def _timeout_s_must_not_be_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("timeout_s must be a number, got bool")
        return value


def load_retrieval_settings() -> RetrievalSettings:
    """Load settings from the environment / `.env` (defaults for anything not set)."""
    return RetrievalSettings()


def create_hybrid_retriever(
    embedder: Embedder, index: SearchIndex, settings: RetrievalSettings
) -> HybridRetriever:
    """Build the HybridRetriever with the configured candidate_k and timeout_s."""
    return HybridRetriever(
        embedder, index, candidate_k=settings.candidate_k, timeout_s=settings.timeout_s
    )
