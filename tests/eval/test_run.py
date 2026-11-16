"""Tests for the CLI (eval/run.py): retriever loading and the main() wiring."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from eval_factories import KC, make_case, make_result

from contracts.retrieval import RetrievalError
from eval.run import load_retriever, main


def test_load_retriever_builds_fake_retriever() -> None:
    assert callable(load_retriever("kc.retrieval.fake_retriever:FakeRetriever").search)


@pytest.mark.parametrize("spec", ["no_colon", ":Attr", "module:", ""])
def test_load_retriever_rejects_bad_spec(spec: str) -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        load_retriever(spec)


def test_load_retriever_rejects_object_without_search() -> None:
    with pytest.raises(TypeError, match="search"):
        load_retriever("builtins:object")


@pytest.fixture
def golden_set(tmp_path: Path) -> Path:
    path = tmp_path / "golden_set.jsonl"
    path.write_text(make_case("q").model_dump_json(), encoding="utf-8")
    return path


def test_main_prints_report_and_returns_0(
    golden_set: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    retriever = AsyncMock()
    retriever.search.return_value = [make_result(KC + "a")]
    code = main(["--golden-set", str(golden_set), "--k", "2"], lambda _: retriever)
    assert code == 0
    retriever.search.assert_awaited_once_with("q", k=2)
    assert "Retrieval metrics @k=2" in capsys.readouterr().out


def test_main_returns_1_when_a_case_failed(golden_set: Path) -> None:
    retriever = AsyncMock()
    retriever.search.side_effect = RetrievalError("down")
    assert main(["--golden-set", str(golden_set)], lambda _: retriever) == 1


def test_main_rejects_k_below_1() -> None:
    with pytest.raises(SystemExit):
        main(["--k", "0"], lambda _: AsyncMock())


def test_main_runs_on_real_golden_set_with_fake_retriever() -> None:
    # Smoke test of the default wiring. Scores are meaningless (fake corpus has mock URLs).
    assert main([]) == 0
