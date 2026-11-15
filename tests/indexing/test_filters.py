"""Validation rules of Filters (D10, agreed with Racheli 2026-11-12).

Filters is Racheli's contract; it lives in the provisional file until contracts/ exists
(D11). These tests should move with it.
"""

import pytest
from pydantic import ValidationError

from kc.indexing._provisional_contracts import Filters

FIELDS = ["products", "versions", "doc_types", "exclude_doc_types"]


def test_all_fields_default_to_none() -> None:
    filters = Filters()
    assert all(getattr(filters, name) is None for name in FIELDS)


@pytest.mark.parametrize("field", FIELDS)
def test_empty_list_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} must be None"):
        Filters(**{field: []})


@pytest.mark.parametrize("field", FIELDS)
def test_non_empty_list_is_accepted(field: str) -> None:
    assert getattr(Filters(**{field: ["x"]}), field) == ["x"]


def test_same_value_in_doc_types_and_exclude_doc_types_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"overlap: \['release-notes'\]"):
        Filters(doc_types=["how-to", "release-notes"], exclude_doc_types=["release-notes"])


def test_disjoint_doc_types_and_exclude_doc_types_are_accepted() -> None:
    filters = Filters(doc_types=["how-to"], exclude_doc_types=["release-notes"])
    assert (filters.doc_types, filters.exclude_doc_types) == (["how-to"], ["release-notes"])


def test_validation_error_is_a_value_error() -> None:
    # Callers that catch ValueError (as agreed: "empty list = ValueError") still work.
    with pytest.raises(ValueError):
        Filters(products=[])
