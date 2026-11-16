"""Shared test setup for tests/indexing (Storage & Indexing).

Tests that need a REAL embedding model (bge-m3 and multilingual-e5-large download ~2.3 GB
each on first run, slow on CPU) are
marked `@pytest.mark.real_model`. They run only when explicitly requested:

    $env:KC_REAL_EMBEDDER = "1"; uv run pytest tests/indexing      # PowerShell
    KC_REAL_EMBEDDER=1 uv run pytest tests/indexing                # bash

and only if the optional extra is installed (`uv sync --extra local-embeddings`).
Everything else (and CI) uses FakeEmbedder.
"""

import os
import sys
import types
from collections.abc import Iterator

import pytest
from sentence_encoder_fakes import RecordingSentenceTransformer

from kc.indexing.bge_m3_embedder import BgeM3Embedder
from kc.indexing.e5_embedder import E5Embedder

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


@pytest.fixture
def fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> type[RecordingSentenceTransformer]:
    """A fake `sentence_transformers` module whose SentenceTransformer only records."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = RecordingSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    RecordingSentenceTransformer.instances.clear()
    return RecordingSentenceTransformer


@pytest.fixture(scope="session")
def bge_m3_embedder() -> Iterator[BgeM3Embedder]:
    """ONE shared BgeM3Embedder for the whole run (the model is loaded once, lazily)."""
    yield BgeM3Embedder()


@pytest.fixture(scope="session")
def e5_embedder() -> Iterator[E5Embedder]:
    """ONE shared E5Embedder for the whole run (~2.27 GB download on first use)."""
    yield E5Embedder()
