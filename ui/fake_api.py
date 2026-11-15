"""Fake answer provider for the UI skeleton.

Stands in for Leah's real `/v1/ask` endpoint. It has the same shape the real client
will have (`async def ask(question) -> Answer`), so swapping it is a one-line import
change in `app.py`.
"""

import asyncio

from models import Answer, Citation

KC_BASE_URL = "https://knowledgecenter.gcora.genpact.com/help"


async def ask(question: str) -> Answer:
    """Return a canned answer. The sleep simulates network latency of a real API call."""
    await asyncio.sleep(0.5)
    return Answer(
        text=(
            f"[FAKE] This is a placeholder answer to: '{question}'. "
            "The real answer will be generated from Knowledge Center articles."
        ),
        citations=[
            Citation(
                title="[FAKE] Configuring SAML authentication",
                url=f"{KC_BASE_URL}/configure-saml-authentication",
            ),
            Citation(
                title="[FAKE] Installation overview",
                url=f"{KC_BASE_URL}/installation-overview",
            ),
        ],
    )
