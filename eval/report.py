"""Human-readable text report for a retrieval evaluation run."""

from eval.retrieval_eval import CaseResult, RetrievalReport


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _case_line(r: CaseResult) -> str:
    if r.error is not None:
        status, detail = "ERROR", r.error
    elif r.scores is None:
        status, detail = "NEG  ", "not scored (negative case)"
    else:
        status = "HIT  " if r.scores.hit else "MISS "
        detail = f"RR={_fmt(r.scores.reciprocal_rank)} recall={_fmt(r.scores.recall)}"
    return f"{status} [{r.case.product or 'any'}] {r.case.question}  ({detail})"


def format_report(report: RetrievalReport) -> str:
    """One line per case, then the summary."""
    summary = [
        "",
        f"Retrieval metrics @k={report.k} (article level)",
        f"  cases:      {len(report.results)} total, {len(report.scores)} scored, "
        f"{report.num_negative} negative (not scored), {report.num_errors} errors",
        f"  Hit Rate:   {_fmt(report.hit_rate)}",
        f"  Recall:     {_fmt(report.mean_recall)}",
        f"  MRR:        {_fmt(report.mrr)}",
    ]
    return "\n".join([*map(_case_line, report.results), *summary])
