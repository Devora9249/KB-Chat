"""Tests for contracts/evaluation.py."""

import pytest
from pydantic import ValidationError

from contracts.evaluation import EvalCase


def make_case(**overrides: object) -> EvalCase:
    data: dict[str, object] = {
        "question": "How do I configure SAML in SeQuence 9?",
        "expected_urls": ["https://knowledgecenter.gcora.genpact.com/help/saml-configuration"],
        "product": "sequence",
        "reference_answer": "Run Set-CoraSeQuenceSiteAuthentication ...",
    }
    data.update(overrides)
    return EvalCase.model_validate(data)


def test_valid_case() -> None:
    case = make_case()
    assert case.product == "sequence"
    assert case.expected_urls[0].endswith("/help/saml-configuration")


def test_product_and_reference_answer_accept_none() -> None:
    case = make_case(product=None, reference_answer=None)
    assert case.product is None
    assert case.reference_answer is None


@pytest.mark.parametrize("missing", ["question", "expected_urls", "product", "reference_answer"])
def test_missing_field_fails(missing: str) -> None:
    data = make_case().model_dump()
    del data[missing]
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


def test_expected_urls_must_be_a_list() -> None:
    with pytest.raises(ValidationError):
        make_case(expected_urls="https://knowledgecenter.gcora.genpact.com/help/x")


def test_json_round_trip() -> None:
    case = make_case()
    assert EvalCase.model_validate_json(case.model_dump_json()) == case