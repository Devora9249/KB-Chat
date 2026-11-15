"""Benchmark BgeM3Embedder on THIS machine (CPU) and print a Markdown table.

Run (from the repo root):
    uv sync --extra local-embeddings
    uv run python -m kc.indexing.benchmarks.embedder_benchmark

What it measures, in order:
1. Cold start: `import torch`, `import sentence_transformers`, model load.
2. RAM (resident set) before and after loading.
3. First question right after loading (the cold first call).
4. Single question after warm-up: mean / median / p95 over 20 questions (10 he, 10 en).
5. 20 questions at once (asyncio.gather): time until the first and until the last answer.
   Calls are queued by the embedder's lock, so the last one waits for all others.
6. 100 realistic chunks (300-600 tokens, counted with the model's tokenizer):
   total time, time per chunk, peak RAM.

The model is loaded exactly as in production (BgeM3Embedder's loader): pinned revision and
no background safetensors download, so nothing in the background distorts the timings.
"""

import argparse
import asyncio
import importlib
import os
import platform
import statistics
import sys
import time
from typing import Any

from kc.indexing.bge_m3_embedder import (
    MODEL_NAME,
    BgeM3Embedder,
    SentenceEncoder,
    _load_sentence_transformer,
)

QUESTIONS_HE = [
    "איך מגדירים אימות SAML ב-SeQuence 9?",
    "מה ההבדל בין Cora Orchestration ל-SeQuence?",
    "איך מתקינים את Case Manager על Windows Server?",
    "איך יוצרים טופס חדש ב-UX Studio?",
    "למה ה-workflow נתקע בשלב אישור?",
    "איך מוסיפים משתמש לקבוצת מנהלים?",
    "מה חדש בגרסה 11 של Orchestration?",
    "איך מגדירים חיבור ל-SQL Server?",
    "איך כותבים expression שמחשב תאריך יעד?",
    "איפה רואים את הלוגים של Flowtime?",
]
QUESTIONS_EN = [
    "How do I configure SAML authentication in SeQuence 9?",
    "What is the difference between Cora Orchestration and SeQuence?",
    "How do I install Case Manager on Windows Server?",
    "How do I create a new form in UX Studio?",
    "Why is my workflow stuck at the approval step?",
    "How do I add a user to the administrators group?",
    "What is new in Orchestration version 11?",
    "How do I configure a SQL Server connection?",
    "How do I write an expression that calculates a due date?",
    "Where can I see the Flowtime logs?",
]
_CHUNK_SENTENCES = [
    "To configure SAML authentication, open the Administration console and select Security.",
    "The identity provider metadata must be uploaded before users can sign in.",
    "Run the Set-CoraSeQuenceSiteAuthentication cmdlet with the -Mode parameter set to SAML.",
    "If the certificate has expired, the sign-in page returns an error and the event is logged.",
    "Workflows are designed in App Studio and deployed to the runtime environment.",
    "Each activity can define input and output parameters that are mapped in the designer.",
    "The Job Execution Service processes scheduled tasks and retries failed jobs.",
    "Release notes list new features, fixed issues and known limitations for each version.",
    "Before upgrading, back up the database and verify that all services are stopped.",
    "UX Studio forms bind controls to data sources and support client-side validation.",
]


# --- memory ------------------------------------------------------------------------------------


def rss_mb() -> tuple[float | None, float | None]:
    """(current, peak) resident memory of this process in MB, or None if unavailable."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Declare 64-bit HANDLE types explicitly: ctypes' default int is 32-bit and the call
        # would fail silently on 64-bit Windows.
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        get_memory_info.restype = wintypes.BOOL

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        if get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
        return None, None
    if sys.platform.startswith("linux"):
        values: dict[str, float] = {}
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    values[key] = int(rest.split()[0]) / 1024  # kB -> MB
        return values.get("VmRSS"), values.get("VmHWM")
    return None, None


# --- helpers -----------------------------------------------------------------------------------


def _timed_import(module: str) -> tuple[Any, float]:
    start = time.perf_counter()
    loaded = importlib.import_module(module)
    return loaded, time.perf_counter() - start


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _make_chunks(tokenizer: Any, count: int) -> tuple[list[str], list[int]]:
    """`count` chunks with a token length spread over 300-600 (measured, not estimated)."""
    chunks: list[str] = []
    lengths: list[int] = []
    for i in range(count):
        target = 300 + (i * 300) // max(count - 1, 1)  # 300 ... 600
        sentences: list[str] = []
        j = i
        while True:
            sentences.append(_CHUNK_SENTENCES[j % len(_CHUNK_SENTENCES)])
            j += 1
            text = " ".join(sentences)
            n_tokens = len(tokenizer(text)["input_ids"])
            if n_tokens >= target:
                break
        chunks.append(text)
        lengths.append(n_tokens)
    return chunks, lengths


async def _concurrent(embedder: BgeM3Embedder, questions: list[str]) -> tuple[float, float]:
    start = time.perf_counter()
    finished: list[float] = []

    async def one(question: str) -> None:
        await embedder.embed([question], input_type="query")
        finished.append(time.perf_counter() - start)

    await asyncio.gather(*(one(q) for q in questions))
    return min(finished), max(finished)


# --- main --------------------------------------------------------------------------------------


async def run(batch_size: int, chunk_count: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rss_start, _ = rss_mb()

    torch, t_torch = _timed_import("torch")
    _, t_st = _timed_import("sentence_transformers")
    rows.append(("import torch", f"{t_torch:.1f} s"))
    rows.append(("import sentence_transformers", f"{t_st:.1f} s"))

    start = time.perf_counter()
    encoder: SentenceEncoder = _load_sentence_transformer("cpu")
    t_load = time.perf_counter() - start
    rss_loaded, _ = rss_mb()
    rows.append(("Model load (from local cache)", f"{t_load:.1f} s"))
    rows.append(("**Cold start total** (imports + load)", f"**{t_torch + t_st + t_load:.1f} s**"))
    if rss_start is not None and rss_loaded is not None:
        rows.append(("RAM before -> after load", f"{rss_start:,.0f} MB -> {rss_loaded:,.0f} MB"))

    embedder = BgeM3Embedder(batch_size=batch_size, encoder_factory=lambda: encoder)

    start = time.perf_counter()
    await embedder.embed([QUESTIONS_EN[0]], input_type="query")
    rows.append(
        ("First question right after load", f"{(time.perf_counter() - start) * 1000:,.0f} ms")
    )

    singles: list[float] = []
    for question in QUESTIONS_HE + QUESTIONS_EN:
        start = time.perf_counter()
        await embedder.embed([question], input_type="query")
        singles.append((time.perf_counter() - start) * 1000)
    rows.append(
        (
            "Single question, warm (20: 10 he + 10 en)",
            f"mean {statistics.mean(singles):,.0f} ms; median {statistics.median(singles):,.0f} ms"
            f"; p95 {_p95(singles):,.0f} ms",
        )
    )

    first, last = await _concurrent(embedder, QUESTIONS_HE + QUESTIONS_EN)
    rows.append(
        (
            "**20 questions at once** (gather)",
            f"first answer {first * 1000:,.0f} ms; **last answer {last * 1000:,.0f} ms**",
        )
    )

    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        rows.append((f"{chunk_count} chunks", "skipped: encoder has no tokenizer"))
    else:
        chunks, lengths = _make_chunks(tokenizer, chunk_count)
        start = time.perf_counter()
        await embedder.embed(chunks, input_type="document")
        total = time.perf_counter() - start
        _, rss_peak = rss_mb()
        rows.append(
            (
                f"{chunk_count} chunks ({min(lengths)}-{max(lengths)} tokens, "
                f"mean {statistics.mean(lengths):.0f})",
                f"total {total:.1f} s; **{total / chunk_count * 1000:,.0f} ms per chunk**",
            )
        )
        if rss_peak is not None:
            rows.append(("Peak RAM (whole run)", f"{rss_peak:,.0f} MB"))

    rows.insert(
        0,
        (
            "Machine",
            f"{platform.system()} {platform.release()}; {os.cpu_count()} logical CPUs; "
            f"torch {torch.__version__}, {torch.get_num_threads()} threads; "
            f"batch_size {batch_size}",
        ),
    )
    rows.insert(0, ("Model", MODEL_NAME))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    rows = asyncio.run(run(args.batch_size, args.chunks))

    print()
    print("| Measurement | Result |")
    print("|---|---|")
    for name, value in rows:
        print(f"| {name} | {value} |")
    print()


if __name__ == "__main__":
    main()
