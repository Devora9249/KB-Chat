"""Shared fixtures for the API tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove API_* variables from the environment before every API test.

    A variable left in the developer's terminal (e.g. `$env:API_ANSWERER = "rag"`)
    would otherwise change the settings the tests run with. monkeypatch restores
    the variables after each test, so the terminal is not affected.
    """
    for name in list(os.environ):
        if name.upper().startswith("API_"):
            monkeypatch.delenv(name)
