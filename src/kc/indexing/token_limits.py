"""How many chunks are longer than a model's max input (for Tami's quality report).

A model silently TRUNCATES text that is longer than its max input: the end of the chunk is
not embedded at all. This counts the chunks affected, with the model's OWN tokenizer
(token counts differ between models) and on exactly the text the model sees (e5 adds
"passage: "), special tokens included.

    from kc.indexing.token_limits import E5_LARGE, chunks_over_limit
    report = chunks_over_limit([chunk.text for chunk in chunks], E5_LARGE)
    print(report.over_limit, "of", report.total, "chunks are truncated")

Loading a tokenizer downloads only its files (~20 MB), not the model weights. Needs the
optional extra (`uv sync --extra local-embeddings`). The counting itself
(`count_over_limit`) takes any token counter, so it is tested without downloads.
"""

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from kc.indexing import bge_m3_embedder, e5_embedder

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class TokenizerSpec:
    """Which tokenizer to use, the model's max input, and what the model adds to a chunk."""

    hub_name: str
    revision: str
    model_id: str
    max_tokens: int
    document_prefix: str = ""


BGE_M3 = TokenizerSpec(
    hub_name=bge_m3_embedder.MODEL_NAME,
    revision=bge_m3_embedder.MODEL_REVISION,
    model_id=bge_m3_embedder.MODEL_ID,
    max_tokens=bge_m3_embedder.MAX_TOKENS,
)
E5_LARGE = TokenizerSpec(
    hub_name=e5_embedder.MODEL_NAME,
    revision=e5_embedder.MODEL_REVISION,
    model_id=e5_embedder.MODEL_ID,
    max_tokens=e5_embedder.MAX_TOKENS,
    document_prefix=e5_embedder.PASSAGE_PREFIX,
)


@dataclass(frozen=True)
class TokenLimitReport:
    model_id: str
    max_tokens: int
    total: int
    over_limit: int
    longest: int  # tokens in the longest chunk (0 if there are no chunks)
    over_limit_indices: tuple[int, ...]  # positions of the truncated chunks in the input

    @property
    def share_over_limit(self) -> float:
        return self.over_limit / self.total if self.total else 0.0


def count_over_limit(
    texts: Sequence[str], spec: TokenizerSpec, count_tokens: TokenCounter
) -> TokenLimitReport:
    """Count the texts whose token count (prefix included) is above spec.max_tokens."""
    counts = [count_tokens(spec.document_prefix + text) for text in texts]
    over = tuple(i for i, n in enumerate(counts) if n > spec.max_tokens)
    return TokenLimitReport(
        model_id=spec.model_id,
        max_tokens=spec.max_tokens,
        total=len(counts),
        over_limit=len(over),
        longest=max(counts, default=0),
        over_limit_indices=over,
    )


def load_token_counter(spec: TokenizerSpec) -> TokenCounter:
    """The model's own tokenizer at the pinned revision; counts special tokens too."""
    # transformers comes with the optional extra; imported lazily (and untyped, as Any).
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hub_name, revision=spec.revision)

    def count_tokens(text: str) -> int:
        # verbose=False: no "sequence longer than max length" warning per long chunk.
        encoded = tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)
        return len(encoded["input_ids"])

    return count_tokens


def chunks_over_limit(
    texts: Sequence[str], spec: TokenizerSpec, count_tokens: TokenCounter | None = None
) -> TokenLimitReport:
    """Report for `texts` (chunk texts) against one model. Loads its tokenizer if needed."""
    return count_over_limit(texts, spec, count_tokens or load_token_counter(spec))
