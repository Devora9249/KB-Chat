"""Offline fallback for the UI: canned answers in the real `contracts.generation.Answer` shape.

Used when KC_UI_BACKEND=fake (no API running). Same signature as `api_client.ask`.

The rules mirror the API's FakeAnswerer (Leah), so the UI behaves the same with either
backend and every status can be exercised. They are copied, not imported, because the
UI must not import another domain's implementation. Checked in this order:
- question contains "unknown"      -> not_found
- question has fewer than 3 words  -> needs_clarification
- anything else                    -> answered, with 2 citations
"""

import asyncio

from contracts.generation import Answer, Citation

KC_BASE_URL = "https://knowledgecenter.gcora.genpact.com/help"
MIN_WORDS_FOR_ANSWER = 3


async def ask(question: str) -> Answer:
    """Return a canned answer. The sleep simulates network latency of a real API call."""
    await asyncio.sleep(0.5)
    question = question.strip()

    if "unknown" in question.lower():
        return Answer(
            text="[FAKE] I didn't find information about this in the Knowledge Center.",
            status="not_found",
        )

    if len(question.split()) < MIN_WORDS_FOR_ANSWER:
        return Answer(
            text=(
                f"[FAKE] Could you be more specific about '{question}'? "
                "Which product and version are you using?"
            ),
            status="needs_clarification",
        )

    return Answer(
        text=(
            f"[FAKE] This is a placeholder answer to: '{question}'. "
            "See the SAML configuration guide [1] and the installation overview [2]."
        ),
        citations=[
            Citation(
                id=1,
                url=f"{KC_BASE_URL}/configure-saml-authentication",
                title="[FAKE] Configuring SAML authentication",
                chunk_id="fake-configure-saml-authentication#0",
            ),
            Citation(
                id=2,
                url=f"{KC_BASE_URL}/installation-overview",
                title="[FAKE] Installation overview",
                chunk_id="fake-installation-overview#0",
            ),
        ],
        status="answered",
    )
