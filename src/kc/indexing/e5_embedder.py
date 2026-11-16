"""Local embedder for intfloat/multilingual-e5-large (dense, 1024 dimensions, multilingual).

Implements contracts.storage.Embedder. Needs the optional extra:
    uv sync --extra local-embeddings

Lazy loading, worker thread, locks, output check, pinned revision and error wrapping are
shared with the other local models: see kc.indexing.sentence_transformer_embedder.

Why "large" and not "base" (official model cards)
- large: 24 layers, 1024 dimensions, ~2.24 GB of weights.
- base:  12 layers,  768 dimensions, ~1.11 GB of weights.
- Both: 94 languages incl. Hebrew ("he"), max 512 tokens.
- large has the SAME dimension as bge-m3 (1024), so the same vector column fits both.

e5 specifics
- Prefixes are REQUIRED (model card FAQ: "this is how the model is trained, otherwise you
  will see a performance degradation"): "query: " for questions and "passage: " for
  documents. They are added here from input_type, so callers pass plain text.
- Max input: 512 tokens, prefix and special tokens included; longer texts are truncated
  (see kc.indexing.token_limits to count how many chunks are affected).
- Pinned model version: MODEL_REVISION is the official intfloat `main` commit. The repo
  has model.safetensors, so no background conversion is triggered.
"""

from contracts.storage import InputType
from kc.indexing.sentence_transformer_embedder import (
    EncoderFactory,
    SentenceEncoder,
    SentenceTransformerEmbedder,
    load_sentence_transformer,
    model_id,
)

MODEL_NAME = "intfloat/multilingual-e5-large"
# Pinned commit of intfloat/multilingual-e5-large `main` on the Hugging Face Hub.
# Changing it changes the vectors: it REQUIRES a full re-index (coordinate with Chaya).
MODEL_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
# What `model_name` reports, e.g. "intfloat/multilingual-e5-large@3d7cfbd".
MODEL_ID = model_id(MODEL_NAME, MODEL_REVISION)
DIMENSION = 1024
# Longest input in tokens, prefix and special tokens included (sentence_bert_config.json).
MAX_TOKENS = 512

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
PREFIXES: dict[InputType, str] = {"query": QUERY_PREFIX, "document": PASSAGE_PREFIX}


def _load_sentence_transformer(device: str) -> SentenceEncoder:
    """Load multilingual-e5-large at the pinned revision."""
    return load_sentence_transformer(MODEL_NAME, MODEL_REVISION, device)


class E5Embedder(SentenceTransformerEmbedder):
    """contracts.storage.Embedder backed by a local intfloat/multilingual-e5-large model."""

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

    def _model_inputs(self, texts: list[str], input_type: InputType) -> list[str]:
        prefix = PREFIXES[input_type]
        return [prefix + text for text in texts]
