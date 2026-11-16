"""Read documents and chunks from two JSONL files and group each chunk with its document.

Format: one JSON object per line, in the shape of the PROVISIONAL Document / Chunk
(kc.indexing._provisional_contracts). Empty lines are ignored.

TODO(D11): once Devora's PR #4 is merged (contracts/ingestion.py), replace this module
with her IngestionSource / the contract models; it is the only place the CLI reads files.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from kc.indexing._provisional_contracts import Chunk, Document


class InputFileError(ValueError):
    """An input file is missing, malformed or inconsistent (message names file and line)."""


@dataclass
class LoadedItems:
    """Documents with their chunks, in the order of the documents file."""

    items: list[tuple[Document, list[Chunk]]] = field(default_factory=list)
    # Documents that have no chunk in the chunks file. They are NOT in `items`:
    # indexing them would store a page that search can never find.
    documents_without_chunks: list[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return sum(len(chunks) for _, chunks in self.items)


def read_jsonl_items(docs_path: Path, chunks_path: Path) -> LoadedItems:
    """Read both files and group every chunk with its document.

    Raises InputFileError for: a missing file, a line that is not valid JSON or does not
    match the model, a document URL that appears twice, or a chunk whose document is not
    in the documents file. Nothing is returned partially.
    """
    documents = _read_models(docs_path, Document)
    chunks = _read_models(chunks_path, Chunk)

    seen: set[str] = set()
    for line_no, document in documents:
        if document.url in seen:
            raise InputFileError(f"{docs_path}:{line_no}: duplicate document url {document.url}")
        seen.add(document.url)

    chunks_by_url: dict[str, list[Chunk]] = defaultdict(list)
    for line_no, chunk in chunks:
        if chunk.document_url not in seen:
            raise InputFileError(
                f"{chunks_path}:{line_no}: chunk {chunk.id} belongs to {chunk.document_url}, "
                f"which is not in {docs_path}"
            )
        chunks_by_url[chunk.document_url].append(chunk)

    loaded = LoadedItems()
    for _, document in documents:
        document_chunks = chunks_by_url.get(document.url, [])
        if document_chunks:
            loaded.items.append((document, document_chunks))
        else:
            loaded.documents_without_chunks.append(document.url)
    return loaded


def _read_models[M: BaseModel](path: Path, model: type[M]) -> list[tuple[int, M]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise InputFileError(f"{path}: cannot read file ({e.strerror or e})") from e
    parsed: list[tuple[int, M]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed.append((line_no, model.model_validate_json(line)))
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "line"
            raise InputFileError(
                f"{path}:{line_no}: not a valid {model.__name__} ({where}: {first['msg']})"
            ) from e
    return parsed
