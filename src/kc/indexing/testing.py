"""Test helpers for loading data into an index (public, for every team's tests).

STABLE API FOR TESTS: `sample_document`, `SampleDocument` and `seed_index`. Other teams
import only from this module and never touch Document or Chunk. Once Document and Chunk
move to contracts/ (D11, Devora's PR #4) only the implementation below changes, not these
signatures (e.g. `version="9"` would be mapped to a versions list internally).

    from kc.indexing.testing import sample_document, seed_index

    saml = sample_document(
        "https://knowledgecenter.gcora.genpact.com/help/configure-saml-sequence",
        "Configure SAML Authentication",
        ["Use SAML to sign in.", "Run Set-CoraSeQuenceSiteAuthentication."],
        product="sequence",
        version="9",
        doc_type="how-to",
    )
    await seed_index(store, embedder, [saml])   # e.g. InMemoryIndexStore, FakeEmbedder
    saml.chunk_ids  # compare with SearchResult.chunk_id

What the helper fills in:
- chunk ids as decided in D12: sha256(f"{url}|{' > '.join(heading_path)}|{part}")[:16]
  (16 hex chars), `part` = 0, 1, 2... in the order of chunk_texts. Deterministic.
- content_hash: sha256 hex of the document content (title + chunk texts).
- the document labels (product, version, doc_type) are copied to every chunk.
"""

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from contracts.common import DocType, Product, Version
from contracts.storage import Embedder
from kc.indexing._provisional_contracts import Chunk, Document, IndexWriter
from kc.indexing.pipeline import index_document

__all__ = ["KC_URL_PATTERN", "SampleDocument", "sample_document", "seed_index"]

# An article URL exactly as the Knowledge Center lists it (same rule as Ingestion, PR #4):
# lowercase letters, digits and hyphens after /help/; no trailing "/", "?query" or "#".
KC_URL_PATTERN = r"^https://knowledgecenter\.gcora\.genpact\.com/help/[a-z0-9][a-z0-9-]*$"
_KC_URL_RE = re.compile(KC_URL_PATTERN)


@dataclass(frozen=True)
class SampleDocument:
    """One document with its chunks, ready for `seed_index`. Built by `sample_document`."""

    url: str
    title: str
    product: Product
    version: Version | None
    doc_type: DocType
    chunk_ids: tuple[str, ...]  # in the order of chunk_texts
    chunk_texts: tuple[str, ...]
    # The contract objects stay private: tests use the fields above only.
    _document: Document = field(repr=False, compare=False)
    _chunks: tuple[Chunk, ...] = field(repr=False, compare=False)


def sample_document(
    url: str,
    title: str,
    chunk_texts: Sequence[str],
    *,
    product: Product,
    version: Version | None = None,
    doc_type: DocType,
    heading_path: Sequence[str] | None = None,
) -> SampleDocument:
    """Build one document with one chunk per text. `heading_path` defaults to [title].

    Raises ValueError if `url` is not a Knowledge Center article URL or `chunk_texts` is
    empty (a document without chunks can never be found by search).
    """
    if not _KC_URL_RE.match(url):
        raise ValueError(
            f"url must look like a Knowledge Center article, "
            f"https://knowledgecenter.gcora.genpact.com/help/<slug> (lowercase letters, "
            f"digits, hyphens; no trailing '/', '?' or '#'), got {url!r}. Ingestion enforces "
            "this format, so tests that use it keep working after D11."
        )
    if isinstance(chunk_texts, str):  # a str is a Sequence[str] of characters
        raise ValueError("chunk_texts must be a list of texts, not a single string")
    texts = tuple(chunk_texts)
    if not texts:
        raise ValueError("chunk_texts must not be empty: a document without chunks is unsearchable")
    headings = list(heading_path) if heading_path is not None else [title]

    content_md = "\n\n".join([f"# {title}", *texts])
    document = Document(
        url=url,
        title=title,
        breadcrumb=[],
        product=product,
        version=version,
        doc_type=doc_type,
        content_md=content_md,
        content_hash=hashlib.sha256(content_md.encode("utf-8")).hexdigest(),
    )
    ids = tuple(_chunk_id(url, headings, part) for part in range(len(texts)))
    chunks = tuple(
        Chunk(
            id=chunk_id,
            document_url=url,
            text=text,
            heading_path=headings,
            product=product,
            version=version,
            doc_type=doc_type,
        )
        for chunk_id, text in zip(ids, texts, strict=True)
    )
    return SampleDocument(
        url=url,
        title=title,
        product=product,
        version=version,
        doc_type=doc_type,
        chunk_ids=ids,
        chunk_texts=texts,
        _document=document,
        _chunks=chunks,
    )


async def seed_index(
    store: IndexWriter, embedder: Embedder, documents: Iterable[SampleDocument]
) -> None:
    """Index `documents` into `store` exactly as production does (index_document).

    Errors are NOT swallowed (e.g. EmbeddingError): a test whose data fails to load must
    fail loudly. Raises ValueError if two documents share a url (the second would silently
    replace the first).
    """
    documents = list(documents)
    urls = [document.url for document in documents]
    duplicates = sorted({url for url in urls if urls.count(url) > 1})
    if duplicates:
        raise ValueError(f"the same url appears more than once: {duplicates}")
    for document in documents:
        await index_document(
            document._document, list(document._chunks), embedder=embedder, store=store
        )


def _chunk_id(url: str, heading_path: list[str], part: int) -> str:
    """D12: deterministic, 16 hex chars, unique across the index (the url is part of it)."""
    key = f"{url}|{' > '.join(heading_path)}|{part}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
