"""Chainlit UI for the Knowledge Center Assistant.

Run from inside ui/ (Chainlit writes .chainlit/ and chainlit.md into the current directory),
on port 8080 so it does not clash with the API on 8000:
    cd ui
    uv run --with chainlit chainlit run app.py --port 8080

Backend is chosen by KC_UI_BACKEND: "api" (default, Leah's /v1/ask) or "fake" (offline).
"""

import os
from collections.abc import Awaitable, Callable
from typing import assert_never

import api_client
import chainlit as cl
import fake_api
from api_client import ApiError

from contracts.generation import Answer

AskFn = Callable[[str], Awaitable[Answer]]

BACKENDS: dict[str, AskFn] = {"api": api_client.ask, "fake": fake_api.ask}

BACKEND = os.environ.get("KC_UI_BACKEND", "api").strip().lower()
if BACKEND not in BACKENDS:
    # Fail at startup with a clear message rather than on the first question.
    raise ValueError(f"KC_UI_BACKEND must be one of {sorted(BACKENDS)}, got {BACKEND!r}")

# Integration day swap, now driven by configuration instead of an import line.
ask: AskFn = BACKENDS[BACKEND]


def render_sources(answer: Answer) -> str:
    """Numbered list of source links, numbered by citation id to match the [n] markers."""
    return "\n".join(
        f"{citation.id}. [{citation.title}]({citation.url})"
        for citation in sorted(answer.citations, key=lambda c: c.id)
    )


def render_answer(answer: Answer) -> str:
    """Format the answer as Markdown according to its status."""
    match answer.status:
        case "answered":
            return f"{answer.text}\n\n**Sources:**\n{render_sources(answer)}"
        case "not_found":
            return answer.text
        case "needs_clarification":
            # The API phrases the text as a question back to the user.
            return answer.text
        case _:
            assert_never(answer.status)


@cl.on_chat_start
async def on_chat_start() -> None:
    note = " (Offline mode: answers are FAKE.)" if BACKEND == "fake" else ""
    await cl.Message(
        content=(
            "### Cora Knowledge Center Assistant\n"
            "Ask a question about Cora Orchestration, Case Manager, SeQuence or OpsManager."
            f"{note}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        answer = await ask(message.content)
    except ApiError as error:
        await cl.Message(content=f"⚠️ {error}").send()
        return
    await cl.Message(content=render_answer(answer)).send()
