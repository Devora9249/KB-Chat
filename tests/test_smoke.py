"""Smoke test so CI has at least one test before domains add their own."""

import contracts
import kc


def test_packages_import() -> None:
    assert kc is not None
    assert contracts is not None
