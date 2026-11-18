"""SampleIngestionSource: an `IngestionSource` backed by the fixture files.

Use it wherever an `IngestionSource` is needed and the real crawler is not wanted
(today: everywhere; after integration: in tests, since it is fast and needs no network):

    from kc.ingestion.samples import SampleIngestionSource

    source: IngestionSource = SampleIngestionSource()
    async for item in source.iter_documents():
        ...  # item.document, item.chunks

Swapping to the real source changes only the line that creates the source.
"""

from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from contracts.ingestion import Chunk, Document, IngestedDocument, IngestionSource

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures"


class SampleIngestionSource:
    """Reads fixtures/sample_docs.jsonl and fixtures/sample_chunks.jsonl."""

    def __init__(self, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self._docs_path = fixtures_dir / "sample_docs.jsonl"
        self._chunks_path = fixtures_dir / "sample_chunks.jsonl"

    async def iter_documents(self) -> AsyncIterator[IngestedDocument]:
        # The two files are separate; group chunks by their page so each item is
        # one document with all of its chunks.
        chunks_by_url: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in _read_jsonl(self._chunks_path, Chunk):
            chunks_by_url[chunk.document_url].append(chunk)
        for doc in _read_jsonl(self._docs_path, Document):
            # IngestedDocument validates that the chunks really belong to the document.
            yield IngestedDocument(document=doc, chunks=chunks_by_url.pop(doc.url, []))
        if chunks_by_url:
            raise ValueError(f"chunks without a document: {sorted(chunks_by_url)}")


if TYPE_CHECKING:
    # Static check only: mypy fails here if SampleIngestionSource stops matching the contract.
    _source: IngestionSource = SampleIngestionSource()


def _read_jsonl[M: (Document, Chunk)](path: Path, model: type[M]) -> list[M]:
    """Parse one model per non-empty line; raises if a line breaks the contract."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]
