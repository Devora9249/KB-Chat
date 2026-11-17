"""Search settings, read from environment variables or the local `.env` file.

Same pattern as `kc.db.settings`: one settings class per domain, a `load_...()` helper, and
a factory that receives the settings (dependency injection, no global).

Environment variables (defaults come from `hybrid_retriever`, not repeated here):
    RETRIEVAL_CANDIDATE_K  candidates requested from EACH search before fusion (int >= 1)
    RETRIEVAL_TIMEOUT_S    time budget for one search call, in seconds (finite, > 0)
    RETRIEVAL_RELEASE_NOTES_WEIGHT
                           multiplier for the score of release notes (finite, 0 < w <= 1;
                           1.0 = no down-weighting)
"""

from typing import Any

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from contracts.storage import Embedder, SearchIndex
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RELEASE_NOTES_WEIGHT,
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
    release_notes_weight: float = Field(
        default=DEFAULT_RELEASE_NOTES_WEIGHT,
        gt=0,
        le=1,
        allow_inf_nan=False,
        validation_alias="RETRIEVAL_RELEASE_NOTES_WEIGHT",
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

    @field_validator("timeout_s", "release_notes_weight", mode="before")
    @classmethod
    def _number_must_not_be_bool(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a number, got bool")
        return value


def load_retrieval_settings() -> RetrievalSettings:
    """Load settings from the environment / `.env` (defaults for anything not set)."""
    return RetrievalSettings()


def create_hybrid_retriever(
    embedder: Embedder, index: SearchIndex, settings: RetrievalSettings
) -> HybridRetriever:
    """Build the HybridRetriever with the configured settings."""
    return HybridRetriever(
        embedder,
        index,
        candidate_k=settings.candidate_k,
        timeout_s=settings.timeout_s,
        release_notes_weight=settings.release_notes_weight,
    )
