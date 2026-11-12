"""Tests for contracts/generation.py."""

import pytest
from pydantic import ValidationError

from contracts.generation import Answer, Citation


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


def test_answer_answered_with_citation() -> None:
    answer = Answer(
        text="Use the SAML settings page [1].", citations=[make_citation()], status="answered"
    )
    assert answer.status == "answered"
    assert answer.citations[0].id == 1


def test_answer_invalid_status_fails() -> None:
    with pytest.raises(ValidationError):
        Answer.model_validate({"text": "x", "status": "maybe"})


def test_answer_answered_without_citations_fails() -> None:
    with pytest.raises(ValidationError, match="requires at least one citation"):
        Answer(text="Trust me.", status="answered")


def test_answer_not_found_needs_no_citations() -> None:
    answer = Answer(text="I didn't find it in the Knowledge Center.", status="not_found")
    assert answer.citations == []


def test_answer_json_round_trip() -> None:
    answer = Answer(text="See [1].", citations=[make_citation()], status="answered")
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
