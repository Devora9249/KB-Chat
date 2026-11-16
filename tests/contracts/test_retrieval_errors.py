"""Error contract of Search & Ranking (contracts.retrieval.RetrievalError)."""

import pytest

import contracts
from contracts.retrieval import RetrievalError
from contracts.storage import EmbeddingError, IndexStoreError


def test_is_runtime_error_not_value_or_type_error() -> None:
    # `except ValueError` / `except TypeError` (caller bugs) must never swallow it.
    assert issubclass(RetrievalError, RuntimeError)
    assert not issubclass(RetrievalError, ValueError)
    assert not issubclass(RetrievalError, TypeError)


@pytest.mark.parametrize("storage_error", [EmbeddingError, IndexStoreError])
def test_is_unrelated_to_storage_errors(storage_error: type[Exception]) -> None:
    # Raw storage errors are not RetrievalError; they must be wrapped explicitly.
    assert not issubclass(RetrievalError, storage_error)
    assert not issubclass(storage_error, RetrievalError)


def test_message_is_kept() -> None:
    assert str(RetrievalError("embedding failed")) == "embedding failed"


def test_single_cause_is_kept_by_identity() -> None:
    # The pattern implementations must use: raise RetrievalError(...) from e
    original = IndexStoreError("db down")
    with pytest.raises(RetrievalError) as exc_info:
        try:
            raise original
        except IndexStoreError as e:
            raise RetrievalError("vector search failed") from e
    assert exc_info.value.__cause__ is original


def test_several_causes_are_kept_in_an_exception_group_in_order() -> None:
    first = EmbeddingError("embedding failed")
    second = IndexStoreError("text search failed")
    with pytest.raises(RetrievalError) as exc_info:
        raise RetrievalError("vector and text search failed") from ExceptionGroup(
            "retrieval failures", [first, second]
        )
    cause = exc_info.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert len(cause.exceptions) == 2
    assert cause.exceptions[0] is first
    assert cause.exceptions[1] is second


def test_is_exported_from_contracts_package() -> None:
    assert "RetrievalError" in contracts.__all__
    assert contracts.RetrievalError is RetrievalError
