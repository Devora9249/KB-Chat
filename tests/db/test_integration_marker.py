"""Checks that the integration-test switch in tests/db/conftest.py behaves as documented."""

import pytest
from conftest import INTEGRATION_ENV_VAR, integration_enabled


def test_integration_enabled_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INTEGRATION_ENV_VAR, raising=False)
    assert integration_enabled() is False
    monkeypatch.setenv(INTEGRATION_ENV_VAR, "0")
    assert integration_enabled() is False
    monkeypatch.setenv(INTEGRATION_ENV_VAR, "1")
    assert integration_enabled() is True


@pytest.mark.integration
def test_integration_marker_runs_only_when_enabled() -> None:
    # If we got here, the autouse fixture did not skip us, so the switch must be on.
    assert integration_enabled()
