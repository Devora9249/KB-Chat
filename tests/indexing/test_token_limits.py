"""Tests for kc.indexing.token_limits (chunks longer than a model's max input)."""

import pytest

from kc.indexing.token_limits import (
    BGE_M3,
    E5_LARGE,
    TokenizerSpec,
    chunks_over_limit,
    count_over_limit,
    load_token_counter,
)

SPEC = TokenizerSpec(hub_name="m", revision="r", model_id="m@r", max_tokens=5)


def _words(text: str) -> int:
    """Fake tokenizer: one token per word."""
    return len(text.split())


def test_counts_only_texts_above_the_limit() -> None:
    texts = ["one two", "1 2 3 4 5", "1 2 3 4 5 6", "1 2 3 4 5 6 7 8"]
    report = count_over_limit(texts, SPEC, _words)
    assert (report.total, report.over_limit, report.longest) == (4, 2, 8)
    assert report.over_limit_indices == (2, 3)  # exactly max_tokens (5) is NOT over
    assert report.share_over_limit == pytest.approx(0.5)
    assert (report.model_id, report.max_tokens) == ("m@r", 5)


def test_prefix_is_counted_like_the_model_sees_it() -> None:
    spec = TokenizerSpec(
        hub_name="m", revision="r", model_id="m@r", max_tokens=5, document_prefix="passage: "
    )
    # 5 words alone fit; with "passage:" they are 6 tokens.
    report = count_over_limit(["1 2 3 4 5"], spec, _words)
    assert report.over_limit == 1


def test_empty_input() -> None:
    report = chunks_over_limit([], SPEC, count_tokens=_words)
    assert report.total == report.over_limit == report.longest == 0
    assert report.share_over_limit == 0.0


def test_specs_match_the_embedders() -> None:
    assert (BGE_M3.max_tokens, BGE_M3.document_prefix) == (8192, "")
    assert (E5_LARGE.max_tokens, E5_LARGE.document_prefix) == (512, "passage: ")
    assert E5_LARGE.model_id == "intfloat/multilingual-e5-large@3d7cfbd"


@pytest.mark.real_model
@pytest.mark.parametrize("spec", [BGE_M3, E5_LARGE], ids=["bge_m3", "e5"])
def test_real_tokenizer_counts_special_tokens_and_finds_long_chunks(spec: TokenizerSpec) -> None:
    count = load_token_counter(spec)
    assert count("") >= 2  # <s> and </s>
    long_text = "configure SAML authentication " * 3000  # far above 8192 tokens
    report = chunks_over_limit(["short chunk", long_text], spec, count_tokens=count)
    assert report.over_limit_indices == (1,)
    assert report.longest > spec.max_tokens
