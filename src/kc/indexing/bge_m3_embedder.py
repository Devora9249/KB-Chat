"""Local embedder for BAAI/bge-m3 (dense, 1024 dimensions, multilingual incl. Hebrew).

Implements contracts.storage.Embedder. Needs the optional extra:
    uv sync --extra local-embeddings

Lazy loading, worker thread, locks, output check, pinned revision, no background
safetensors conversion and error wrapping are shared with the other local models: see
kc.indexing.sentence_transformer_embedder.

bge-m3 specifics
- input_type is accepted (callers MUST pass it) but IGNORED: per the official model card,
  bge-m3 "no longer requires adding instructions to the queries".
- Pinned model version: MODEL_REVISION is the official BAAI `main` commit. BAAI's `main`
  only has pytorch_model.bin, which is why the loader stops the background safetensors
  conversion (~2.1 GB that would never be used).
- Max input: 8192 tokens (sentence_bert_config.json); longer texts are truncated.
"""

from kc.indexing.sentence_transformer_embedder import (
    DISABLE_CONVERSION_ENV_VAR,
    EncoderFactory,
    SentenceEncoder,
    SentenceTransformerEmbedder,
    load_sentence_transformer,
    model_id,
)

__all__ = [
    "DIMENSION",
    "DISABLE_CONVERSION_ENV_VAR",
    "MAX_TOKENS",
    "MODEL_ID",
    "MODEL_NAME",
    "MODEL_REVISION",
    "BgeM3Embedder",
    "SentenceEncoder",
]

MODEL_NAME = "BAAI/bge-m3"
# Pinned commit of BAAI/bge-m3 `main` on the Hugging Face Hub (official BAAI weights).
# Changing it changes the vectors: it REQUIRES a full re-index (coordinate with Tami).
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
# What `model_name` reports, e.g. "BAAI/bge-m3@5617a9f" (see model_id()).
MODEL_ID = model_id(MODEL_NAME, MODEL_REVISION)
DIMENSION = 1024
# Longest input in tokens, special tokens included (sentence_bert_config.json).
MAX_TOKENS = 8192


def _load_sentence_transformer(device: str) -> SentenceEncoder:
    """Load bge-m3 at the pinned revision, without the background safetensors download."""
    return load_sentence_transformer(MODEL_NAME, MODEL_REVISION, device)


class BgeM3Embedder(SentenceTransformerEmbedder):
    """contracts.storage.Embedder backed by a local BAAI/bge-m3 model."""

    def __init__(
        self,
        *,
        batch_size: int = 16,
        device: str = "cpu",
        encoder_factory: EncoderFactory | None = None,
    ) -> None:
        super().__init__(
            hub_name=MODEL_NAME,
            model_id=MODEL_ID,
            dimension=DIMENSION,
            batch_size=batch_size,
            encoder_factory=encoder_factory or (lambda: _load_sentence_transformer(device)),
        )
