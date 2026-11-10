"""Chainlit UI skeleton for the Knowledge Center Assistant.

Run from inside ui/ (Chainlit writes .chainlit/ and chainlit.md into the current directory):
    cd ui
    uv run --with chainlit chainlit run app.py
"""

import chainlit as cl

# The one line to change on integration day: swap the fake for the real API client.
from fake_api import ask
from models import Answer


def render_answer(answer: Answer) -> str:
    """Format the answer as Markdown: answer text, then a numbered list of source links."""
    sources = "\n".join(
        f"{i}. [{citation.title}]({citation.url})"
        for i, citation in enumerate(answer.citations, start=1)
    )
    return f"{answer.text}\n\n**Sources:**\n{sources}" if sources else answer.text


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content="Hi! Ask me a question about Cora products (answers are FAKE for now)."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    answer = await ask(message.content)
    await cl.Message(content=render_answer(answer)).send()
