"""Dependency providers for the API.

Integration day: change the one line in `get_answerer` to return the real answerer.
Tests replace providers with `app.dependency_overrides`.
"""

from kc.generation.answerer import Answerer
from kc.generation.fake_answerer import FakeAnswerer


def get_answerer() -> Answerer:
    return FakeAnswerer()
