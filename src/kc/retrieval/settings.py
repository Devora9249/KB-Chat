"""Search settings, read from environment variables or the local `.env` file.

Same pattern as `kc.db.settings`: one settings class per domain, a `load_...()` helper, and
a factory that receives the settings (dependency injection, no global).

Environment variables (defaults come from `hybrid_retriever`, not repeated here):
    RETRIEVAL_CANDIDATE_K  candidates requested from EACH search before fusion (int >= 1)
    RETRIEVAL_TIMEOUT_S    time budget for one search call, in seconds (finite, > 0)
    RETRIEVAL_RELEASE_NOTES_WEIGHT
                           multiplier for the score of release notes (finite, 0 < w <= 1;
                           1.0 = no down-weighting)
    RETRIEVAL_RERANK_CANDIDATES
                           top results sent to the reranker, at least k (int >= 1); used
                           only when a reranker is passed to the factory
    RETRIEVAL_RERANK_TIMEOUT_S
                           the reranker's own time budget, in seconds (finite, > 0); on
                           expiry search falls back to the RRF ranking
    RETRIEVAL_RERANKER     which reranker `create_reranker` builds: "none" (default, no
                           reranking), "fake" (development and tests only) or "voyage".
                           PRIVACY: "voyage" sends user queries and passage texts to
                           Voyage AI; do not enable it before the privacy decision.
    RETRIEVAL_RERANK_MODEL Voyage rerank model (default "rerank-3-lite")
    VOYAGE_API_KEY         Voyage API key, required only for "voyage" (secret: masked in
                           repr and never echoed in validation errors)
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from contracts.storage import Embedder, SearchIndex
from kc.retrieval.fake_reranker import FakeReranker
from kc.retrieval.hybrid_retriever import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RELEASE_NOTES_WEIGHT,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_TIMEOUT_S,
    HybridRetriever,
)
from kc.retrieval.reranker import Reranker
from kc.retrieval.voyage_reranker import VoyageReranker

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL: Final = "rerank-3-lite"


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,  # never echo input values (VOYAGE_API_KEY) in errors
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
    rerank_candidates: int = Field(
        default=DEFAULT_RERANK_CANDIDATES, ge=1, validation_alias="RETRIEVAL_RERANK_CANDIDATES"
    )
    rerank_timeout_s: float = Field(
        default=DEFAULT_RERANK_TIMEOUT_S,
        gt=0,
        allow_inf_nan=False,
        validation_alias="RETRIEVAL_RERANK_TIMEOUT_S",
    )
    reranker: Literal["none", "fake", "voyage"] = Field(
        default="none", validation_alias="RETRIEVAL_RERANKER"
    )
    rerank_model: str = Field(
        default=DEFAULT_RERANK_MODEL, min_length=1, validation_alias="RETRIEVAL_RERANK_MODEL"
    )
    voyage_api_key: SecretStr | None = Field(default=None, validation_alias="VOYAGE_API_KEY")

    @field_validator("rerank_model", mode="before")
    @classmethod
    def _strip_model(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("voyage_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: Any) -> Any:
        # .env.example has `VOYAGE_API_KEY=` (empty): treat a blank key as "not set".
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw, str) and not raw.strip():
            return None
        return value

    # pydantic's default (lax) mode would accept True as 1, and 5.0 or "5.0" as 5. The
    # retriever contract rejects them, so reject them here too, before pydantic converts.
    @field_validator("candidate_k", "rerank_candidates", mode="before")
    @classmethod
    def _count_must_be_a_whole_int(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool | float):
            raise ValueError(f"{info.field_name} must be an int, got {type(value).__name__}")
        if isinstance(value, str):
            try:
                return int(value)  # "5.0" and "abc" are rejected here
            except ValueError:
                raise ValueError(f"{info.field_name} must be an int, got {value!r}") from None
        return value

    @field_validator("timeout_s", "release_notes_weight", "rerank_timeout_s", mode="before")
    @classmethod
    def _number_must_not_be_bool(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a number, got bool")
        return value


def load_retrieval_settings() -> RetrievalSettings:
    """Load settings from the environment / `.env` (defaults for anything not set)."""
    return RetrievalSettings()


def create_hybrid_retriever(
    embedder: Embedder,
    index: SearchIndex,
    settings: RetrievalSettings,
    *,
    reranker: Reranker | None = None,
) -> HybridRetriever:
    """Build the HybridRetriever with the configured settings.

    `reranker` is optional (keyword-only); None keeps the RRF ranking exactly as before.
    """
    return HybridRetriever(
        embedder,
        index,
        candidate_k=settings.candidate_k,
        timeout_s=settings.timeout_s,
        release_notes_weight=settings.release_notes_weight,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
        rerank_timeout_s=settings.rerank_timeout_s,
    )


def create_reranker(settings: RetrievalSettings) -> Reranker | None:
    """Build the configured reranker. Local only: no network call, no warm-up.

    "none" -> None (search keeps the RRF ranking); "fake" -> FakeReranker (development and
    tests only; logs a WARNING, but is not blocked: there is no production detection);
    "voyage" -> VoyageReranker, which must be closed (use `open_reranker`).
    Raises ValueError when "voyage" is selected without VOYAGE_API_KEY (the message names
    the variable, never a value).
    """
    if settings.reranker == "none":
        return None
    if settings.reranker == "fake":
        # A visible warning only, NOT a production block: the project has no environment
        # detection (same convention as API_ANSWERER=fake and KC_UI_BACKEND=fake).
        logger.warning(
            "RETRIEVAL_RERANKER=fake: FakeReranker (token overlap) is for development "
            "and tests only"
        )
        return FakeReranker()
    if settings.voyage_api_key is None:
        raise ValueError("RETRIEVAL_RERANKER=voyage requires VOYAGE_API_KEY")
    return VoyageReranker(
        settings.voyage_api_key,
        model=settings.rerank_model,
        timeout_s=settings.rerank_timeout_s,
    )


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


async def _close(reranker: Reranker | None) -> None:
    if isinstance(reranker, _AsyncClosable):
        await reranker.aclose()


async def _close_quietly(reranker: Reranker | None) -> None:
    """Try to close while another exception is propagating; never replace it.

    Catches Exception only: a NEW asyncio.CancelledError during the close propagates.
    """
    try:
        await _close(reranker)
    except Exception as exc:
        logger.warning("closing the reranker failed (%s)", type(exc).__name__)


@asynccontextmanager
async def open_reranker(settings: RetrievalSettings) -> AsyncIterator[Reranker | None]:
    """Create the configured reranker and always close it on exit (for the API lifespan).

        async with open_reranker(settings) as reranker:
            retriever = create_hybrid_retriever(embedder, index, settings, reranker=reranker)
            ...

    No network call when entering or leaving. A configuration error is raised on entry.
    If the block raises (including asyncio.CancelledError), the close is attempted and
    the ORIGINAL exception propagates unchanged (a close error is only logged); on a
    normal exit a close error propagates.
    """
    reranker = create_reranker(settings)
    try:
        yield reranker
    except BaseException:
        await _close_quietly(reranker)
        raise
    else:
        await _close(reranker)
