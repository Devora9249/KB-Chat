"""Quality checks on the real Golden Set file (eval/golden_set.jsonl).

These run in CI on every PR, so a broken question never reaches `main` and silently skews
the metrics. Each check lists *all* offending items, so one run shows everything to fix.
The last four checks guard the design of the set: dropping a whole product or all negative
cases makes a test fail, so such a change is made consciously.
"""

import json
import re
from collections import Counter
from typing import get_args

import pytest

from contracts.common import Product
from contracts.evaluation import EvalCase
from eval.golden_set import GOLDEN_SET_PATH, load_golden_set

VALID_PRODUCTS = set(get_args(Product))

# A full Knowledge Center article URL: https, lowercase slug, no trailing slash or "#".
KC_ARTICLE_URL = re.compile(
    r"https://knowledgecenter\.gcora\.genpact\.com/help/[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
)


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    return load_golden_set()


def test_no_unknown_keys() -> None:
    # Pydantic ignores extra keys, so a field the contract lacks would be silently dropped.
    allowed = set(EvalCase.model_fields)
    lines = GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
    unknown = {
        line_no: sorted(extra)
        for line_no, line in enumerate(lines, start=1)
        if line.strip() and (extra := set(json.loads(line)) - allowed)
    }
    assert unknown == {}


def test_questions_are_not_blank(cases: list[EvalCase]) -> None:
    assert [c for c in cases if not c.question.strip()] == []


def test_questions_are_unique(cases: list[EvalCase]) -> None:
    # Ignore case and extra spaces: "What is JES?" and "what  is jes?" are the same question.
    counts = Counter(" ".join(c.question.casefold().split()) for c in cases)
    assert [q for q, n in counts.items() if n > 1] == []


def test_products_are_valid(cases: list[EvalCase]) -> None:
    assert [c.product for c in cases if c.product not in VALID_PRODUCTS | {None}] == []


def test_urls_are_kc_articles(cases: list[EvalCase]) -> None:
    assert [u for c in cases for u in c.expected_urls if not KC_ARTICLE_URL.fullmatch(u)] == []


def test_no_duplicate_urls_within_a_case(cases: list[EvalCase]) -> None:
    assert [c.question for c in cases if len(set(c.expected_urls)) != len(c.expected_urls)] == []


def test_has_at_least_20_cases(cases: list[EvalCase]) -> None:
    assert len(cases) >= 20


def test_every_product_is_covered(cases: list[EvalCase]) -> None:
    assert {c.product for c in cases if c.product is not None} == VALID_PRODUCTS


def test_has_negative_cases(cases: list[EvalCase]) -> None:
    assert sum(1 for c in cases if not c.expected_urls) >= 2


def test_has_product_agnostic_case(cases: list[EvalCase]) -> None:
    assert any(c.product is None and c.expected_urls for c in cases)
