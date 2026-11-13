"""Tests for kc.ingestion.ids."""

import pytest

from contracts import Chunk
from kc.ingestion.ids import content_hash, make_chunk_id

URL = "https://knowledgecenter.gcora.genpact.com/help/configure-saml"


def test_content_hash_is_stable_and_sensitive() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
    assert len(content_hash("abc")) == 64


def test_chunk_id_is_deterministic() -> None:
    assert make_chunk_id(URL, ["A", "B"], 0) == make_chunk_id(URL, ["A", "B"], 0)


@pytest.mark.parametrize(
    ("url", "path", "part"),
    [
        (URL + "-other", ["A", "B"], 0),  # other page
        (URL, ["A", "C"], 0),  # other heading
        (URL, ["A", "B"], 1),  # next part of the same section
    ],
)
def test_chunk_id_differs_when_any_input_differs(url: str, path: list[str], part: int) -> None:
    assert make_chunk_id(url, path, part) != make_chunk_id(URL, ["A", "B"], 0)


def test_chunk_id_matches_contract_format() -> None:
    chunk_id = make_chunk_id(URL, ["A"], 0)
    Chunk(
        id=chunk_id,
        document_url=URL,
        text="x",
        heading_path=["A"],
        product="sequence",
        doc_type="how-to",
    )


def test_negative_part_rejected() -> None:
    with pytest.raises(ValueError):
        make_chunk_id(URL, ["A"], -1)
