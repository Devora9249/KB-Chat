"""Error contract of Storage & Indexing (consumed by Racheli's RetrievalError)."""

import pytest

from kc.indexing import _provisional_contracts as contracts
from kc.indexing._provisional_contracts import EmbeddingError, IndexStoreError


@pytest.mark.parametrize("error_type", [EmbeddingError, IndexStoreError])
def test_infrastructure_errors_are_runtime_errors_not_value_errors(
    error_type: type[Exception],
) -> None:
    # `except ValueError` (caller bugs) must never swallow an infrastructure failure.
    assert issubclass(error_type, RuntimeError)
    assert not issubclass(error_type, ValueError)


def test_errors_are_distinct() -> None:
    assert not issubclass(EmbeddingError, IndexStoreError)
    assert not issubclass(IndexStoreError, EmbeddingError)


def test_errors_are_exported_from_the_single_import_point() -> None:
    assert {"EmbeddingError", "IndexStoreError"} <= set(contracts.__all__)


def test_wrapping_keeps_the_original_error_in_cause() -> None:
    # The pattern implementations must use: raise IndexStoreError(...) from e
    original = ConnectionError("db down")
    with pytest.raises(IndexStoreError) as exc_info:
        try:
            raise original
        except ConnectionError as e:
            raise IndexStoreError("vector_search failed") from e
    assert exc_info.value.__cause__ is original
