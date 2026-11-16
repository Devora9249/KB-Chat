"""Tests for the Golden Set loader (eval/golden_set.py), on small synthetic files."""

import json
from pathlib import Path

import pytest

from eval.golden_set import load_golden_set

VALID_LINE = json.dumps(
    {
        "question": "How do I configure SAML in SeQuence 9?",
        "expected_urls": ["https://knowledgecenter.gcora.genpact.com/help/saml"],
        "product": "sequence",
        "reference_answer": None,
    }
)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "golden_set.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_cases_and_skips_blank_lines(tmp_path: Path) -> None:
    cases = load_golden_set(write(tmp_path, f"\n{VALID_LINE}\n\n   \n{VALID_LINE}\n"))
    assert len(cases) == 2
    assert cases[0].question == "How do I configure SAML in SeQuence 9?"
    assert cases[0].product == "sequence"


@pytest.mark.parametrize(
    "bad_line",
    [
        "{not json",  # broken JSON
        '{"question": "What is JES?"}',  # missing required fields
    ],
)
def test_bad_line_reports_its_line_number(tmp_path: Path, bad_line: str) -> None:
    path = write(tmp_path, f"{VALID_LINE}\n{bad_line}\n")
    with pytest.raises(ValueError, match=r"golden_set\.jsonl, line 2:"):
        load_golden_set(path)
