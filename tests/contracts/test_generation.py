"""Tests for contracts/generation.py."""

import pytest
from pydantic import ValidationError

from contracts.generation import Citation


def make_citation(**overrides: object) -> Citation:
    data: dict[str, object] = {
        "id": 1,
        "url": "https://knowledgecenter.gcora.genpact.com/help/saml-configuration",
        "title": "Configure SAML",
        "chunk_id": "saml-configuration#0",
    }
    data.update(overrides)
    return Citation.model_validate(data)


def test_citation_valid() -> None:
    citation = make_citation()
    assert citation.id == 1
    assert citation.url.endswith("/help/saml-configuration")


def test_citation_missing_field_fails() -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate({"id": 1, "title": "x", "chunk_id": "c"})  # no url


def test_citation_id_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        make_citation(id=0)


def test_citation_json_round_trip() -> None:
    citation = make_citation()
    assert Citation.model_validate_json(citation.model_dump_json()) == citation
