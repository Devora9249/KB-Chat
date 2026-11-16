"""Load the Golden Set: questions with known correct answers, used to measure quality.

The file `eval/golden_set.jsonl` is JSON Lines: one `EvalCase` (see `contracts/evaluation.py`)
per line. `expected_urls == []` marks a negative case: the knowledge base has no answer and
the system is expected to say so (`Answer.status == "not_found"`).

Data-quality rules (valid URLs, no duplicate questions, ...) are checked by
`tests/eval/test_golden_set_data.py` in CI, so this module only loads.
"""

from pathlib import Path
from typing import Final

from pydantic import ValidationError

from contracts.evaluation import EvalCase

GOLDEN_SET_PATH: Final = Path(__file__).with_name("golden_set.jsonl")


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[EvalCase]:
    """Read the Golden Set. Blank lines are skipped; a bad line raises ValueError with its number.

    Synchronous on purpose: a small local file read once before an eval run
    (asyncio has no native async file I/O; it would only add a thread hop).
    """
    cases: list[EvalCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path.name}, line {line_no}: {exc}") from exc
    return cases
