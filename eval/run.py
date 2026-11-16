"""CLI: run the Golden Set retrieval evaluation and print the report.

Usage (from the repo root):
    uv run python -m eval.run                       # FakeRetriever, k=5
    uv run python -m eval.run --k 10
    uv run python -m eval.run --retriever some.module:SomeRetriever

Dependency injection: the retriever is chosen by a "module:attribute" string loaded at
runtime, so eval depends only on the `Retriever` protocol in `contracts/`. On integration
day, switch to the real retriever by changing `DEFAULT_RETRIEVER` or passing --retriever.

Exit code: 0 = all cases ran, 1 = some cases hit a RetrievalError (so failed cases
cannot silently improve the averages).
"""

import argparse
import asyncio
import importlib
import io
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, cast

from contracts.retrieval import Retriever
from eval.golden_set import GOLDEN_SET_PATH, load_golden_set
from eval.report import format_report
from eval.retrieval_eval import evaluate_retrieval

DEFAULT_RETRIEVER: Final = "kc.retrieval.fake_retriever:FakeRetriever"
DEFAULT_K: Final = 5


def load_retriever(spec: str) -> Retriever:
    """Build a retriever from "package.module:Attribute" (a class or no-argument factory)."""
    module_name, sep, attr_name = spec.partition(":")
    if not (sep and module_name and attr_name):
        raise ValueError(f"expected 'module:attribute', got {spec!r}")
    retriever = getattr(importlib.import_module(module_name), attr_name)()
    if not callable(getattr(retriever, "search", None)):
        raise TypeError(f"{spec} did not produce an object with a search() method")
    return cast(Retriever, retriever)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run", description="Run the Golden Set retrieval evaluation."
    )
    parser.add_argument("--retriever", default=DEFAULT_RETRIEVER, help="'module:attribute'")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="results per question")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    return args


def main(
    argv: Sequence[str] | None = None,
    retriever_loader: Callable[[str], Retriever] = load_retriever,
) -> int:
    args = _parse_args(argv)
    retriever = retriever_loader(args.retriever)
    report = asyncio.run(evaluate_retrieval(retriever, load_golden_set(args.golden_set), args.k))
    print(format_report(report))
    return 1 if report.num_errors else 0


if __name__ == "__main__":
    # Windows defaults piped output to a legacy code page (e.g. cp1252), which crashes on
    # non-Latin questions. Force UTF-8 for the CLI.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
