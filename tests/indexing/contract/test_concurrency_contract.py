"""Concurrency contract: ONE IndexStore / Embedder instance serves many concurrent async calls.

The application (HybridRetriever, API) keeps a single instance for its whole lifetime and
uses it from concurrent requests. These tests run many calls at once with asyncio.gather.

Note: InMemoryIndexStore and FakeEmbedder have no `await` inside their methods, so each
call runs to completion without interruption. The tests still matter: they guard against
a future change that adds an `await` mid-operation, and they will exercise real
interleaving once PgIndexStore / BgeM3Embedder are registered in conftest.py.
"""

import asyncio

from index_store_helpers import make_chunk, make_doc

from kc.indexing._provisional_contracts import Chunk, Document, Embedder, IndexStore, SearchResult
from kc.indexing.vector_math import cosine_similarity

QUERY = [1.0, 0.0, 0.0]
# Two versions of the SAME document, with different sizes, so a mix is always detectable.
VERSION_SIZES = {"A": 5, "B": 3}


def _version(tag: str) -> tuple[Document, list[Chunk], list[list[float]]]:
    size = VERSION_SIZES[tag]
    doc = make_doc("u1", title=f"title-{tag}", content_hash=f"hash-{tag}")
    chunks = [make_chunk(f"{tag}-{i}", "u1", "saml") for i in range(size)]
    return doc, chunks, [QUERY] * size


def _assert_single_complete_version(results: list[SearchResult]) -> None:
    tags = {r.chunk_id.split("-")[0] for r in results}
    assert len(tags) == 1, (
        "search must see exactly one complete version, got "
        f"{sorted(r.chunk_id for r in results) or 'no results (document missing mid-update)'}"
    )
    (tag,) = tags
    assert len(results) == VERSION_SIZES[tag], f"search saw a partial version {tag}"
    assert {r.title for r in results} == {f"title-{tag}"}


def test_concurrent_searches_during_upserts_never_see_a_mix(store: IndexStore) -> None:
    async def scenario() -> tuple[list[list[SearchResult]], dict[str, str]]:
        await store.upsert_document(*_version("A"))

        async def write(tag: str) -> None:
            await asyncio.sleep(0)  # yield, so tasks interleave
            await store.upsert_document(*_version(tag))

        async def read(kind: str) -> list[SearchResult]:
            await asyncio.sleep(0)
            if kind == "vector":
                return await store.vector_search(QUERY, k=100)
            return await store.text_search("saml", k=100)

        writes = [write("B" if i % 2 == 0 else "A") for i in range(20)]
        reads = [read("vector" if i % 2 == 0 else "text") for i in range(60)]
        # Interleave: read, read, read, write, ...
        tasks = []
        for i, r in enumerate(reads):
            tasks.append(asyncio.ensure_future(r))
            if i % 3 == 2 and writes:
                tasks.append(asyncio.ensure_future(writes.pop()))
        tasks.extend(asyncio.ensure_future(w) for w in writes)
        done = await asyncio.gather(*tasks)
        searches = [d for d in done if isinstance(d, list)]
        return searches, await store.get_document_hashes()

    searches, hashes = asyncio.run(scenario())

    assert len(searches) == 60
    for results in searches:
        _assert_single_complete_version(results)
    assert hashes in ({"u1": "hash-A"}, {"u1": "hash-B"})


def test_concurrent_embed_calls_get_their_own_vectors_in_order(embedder: Embedder) -> None:
    # 30 calls of different sizes and input types; every text is unique (incl. Hebrew), so a
    # vector returned to the wrong call or at the wrong position is detectable.
    calls = [
        (
            [
                f"call {c} text {t} {'שאלה בעברית' if t % 2 else 'configure SAML'}"
                for t in range(1 + c % 6)
            ],
            "query" if c % 2 else "document",
        )
        for c in range(30)
    ]

    async def scenario() -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        concurrent = await asyncio.gather(
            *(embedder.embed(texts, input_type=kind) for texts, kind in calls)  # type: ignore[arg-type]
        )
        # Reference: each text embedded ALONE, one call at a time.
        one_by_one = [
            [(await embedder.embed([text], input_type=kind))[0] for text in texts]  # type: ignore[arg-type]
            for texts, kind in calls
        ]
        return list(concurrent), one_by_one

    concurrent, one_by_one = asyncio.run(scenario())

    assert len(concurrent) == len(calls)
    for (texts, _), got, expected in zip(calls, concurrent, one_by_one, strict=True):
        assert len(got) == len(texts)
        for got_vector, expected_vector in zip(got, expected, strict=True):
            # Tolerance instead of equality: real models may differ in the last digits
            # depending on how texts are batched.
            assert cosine_similarity(got_vector, expected_vector) > 0.999
