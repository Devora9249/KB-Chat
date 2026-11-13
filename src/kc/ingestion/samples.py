"""Load the sample fixtures (fixtures/sample_docs.jsonl, fixtures/sample_chunks.jsonl).

Teammates use these until the real Ingestion pipeline is ready:

    from kc.ingestion.samples import load_sample_chunks, load_sample_documents
"""

from pathlib import Path

from contracts import Chunk, Document

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures"


def load_sample_documents(path: Path | None = None) -> list[Document]:
    path = path or FIXTURES_DIR / "sample_docs.jsonl"
    return [Document.model_validate_json(line) for line in _lines(path)]


def load_sample_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or FIXTURES_DIR / "sample_chunks.jsonl"
    return [Chunk.model_validate_json(line) for line in _lines(path)]


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
