"""Stable identifiers for documents and chunks.

Both functions are deterministic: the same input always gives the same output,
so re-crawling an unchanged page produces the same hash and the same chunk ids.
"""

import hashlib


def content_hash(content_md: str) -> str:
    """Fingerprint of a page body. Changes if and only if the content changes."""
    return hashlib.sha256(content_md.encode("utf-8")).hexdigest()


def make_chunk_id(document_url: str, heading_path: list[str], part: int) -> str:
    """Id of a chunk, unique across the whole index.

    `part` counts chunks under the same heading_path (0, 1, 2...), for long sections
    that are split in several chunks or for repeated headings. Adding content to one
    section only changes the ids of that section.
    """
    if part < 0:
        raise ValueError("part must be >= 0")
    key = f"{document_url}|{' > '.join(heading_path)}|{part}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
