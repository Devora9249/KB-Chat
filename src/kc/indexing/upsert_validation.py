"""Input checks for IndexWriter.upsert_document, shared by every implementation.

Run BEFORE touching any storage, so an invalid call changes nothing (contract D6).
"""

from collections import Counter

from kc.indexing._provisional_contracts import Chunk, Document


def validate_upsert(
    document: Document, chunks: list[Chunk], vectors: list[list[float]], dimension: int
) -> None:
    """Raise ValueError (a bug in the caller) if the input cannot be stored as is."""
    if len(chunks) != len(vectors):
        raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        if len(vector) != dimension:
            raise ValueError(
                f"vector {i} has dimension {len(vector)}, index dimension is {dimension}"
            )
        if chunk.document_url != document.url:
            raise ValueError(
                f"chunk {chunk.id!r} has document_url {chunk.document_url!r}, "
                f"expected {document.url!r}"
            )
    duplicates = sorted(cid for cid, count in Counter(c.id for c in chunks).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"duplicate chunk ids in one upsert of {document.url!r}: {duplicates} "
            "(chunk.id must be unique across the whole index, D12)"
        )
