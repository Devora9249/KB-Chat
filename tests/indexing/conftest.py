"""Shared test setup for tests/indexing (Storage & Indexing).

Tests that need a REAL embedding model (downloads ~2.3 GB on first run, slow on CPU) are
marked `@pytest.mark.real_model`. They run only when explicitly requested:

    $env:KC_REAL_EMBEDDER = "1"; uv run pytest tests/indexing      # PowerShell
    KC_REAL_EMBEDDER=1 uv run pytest tests/indexing                # bash

and only if the optional extra is installed (`uv sync --extra local-embeddings`).
Everything else (and CI) uses FakeEmbedder.
"""

import os
from collections.abc import Iterator

import pytest

from kc.indexing.bge_m3_embedder import BgeM3Embedder

REAL_EMBEDDER_ENV_VAR = "KC_REAL_EMBEDDER"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"real_model: loads a real embedding model; runs only when {REAL_EMBEDDER_ENV_VAR}=1",
    )


@pytest.fixture(autouse=True)
def _skip_real_model_unless_requested(request: pytest.FixtureRequest) -> None:
    if not request.node.get_closest_marker("real_model"):
        return
    if os.environ.get(REAL_EMBEDDER_ENV_VAR) != "1":
        pytest.skip(f"real model test: set {REAL_EMBEDDER_ENV_VAR}=1 to run")
    pytest.importorskip(
        "sentence_transformers",
        reason="install the extra first: uv sync --extra local-embeddings",
    )


@pytest.fixture(scope="session")
def bge_m3_embedder() -> Iterator[BgeM3Embedder]:
    """ONE shared BgeM3Embedder for the whole run (the model is loaded once, lazily)."""
    yield BgeM3Embedder()
