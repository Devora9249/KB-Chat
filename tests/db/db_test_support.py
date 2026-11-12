"""Helpers shared by tests/db (kept out of conftest.py: several folders have a conftest.py,
so `from conftest import ...` could pick up the wrong one depending on collection order)."""

import os

INTEGRATION_ENV_VAR = "KC_INTEGRATION"


def integration_enabled() -> bool:
    return os.environ.get(INTEGRATION_ENV_VAR) == "1"
