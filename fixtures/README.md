# Fixtures (owner: Devora)

Sample Ingestion output in the real contract shape (`contracts/ingestion.py`),
so every domain can build and test without waiting for the crawler.

| File | Content |
|---|---|
| `sample_docs.jsonl` | 10 `Document`s, one JSON object per line |
| `sample_chunks.jsonl` | 28 `Chunk`s belonging to those documents |
| `build_samples_v0.py` | Script that generates both files |

## How to use

Depend on the `IngestionSource` Protocol from `contracts`, and receive it as a parameter:

```python
from contracts import IngestionSource


async def index_all(source: IngestionSource) -> None:
    async for item in source.iter_documents():
        ...  # item.document: Document, item.chunks: list[Chunk]
```

For now, pass the sample implementation:

```python
from kc.ingestion.samples import SampleIngestionSource

await index_all(SampleIngestionSource())
```

On integration day only that line changes (to the real crawler-based source).
Keep using `SampleIngestionSource` in unit tests: it is fast and needs no network.

Every item is checked by the contract: it has at least one chunk, and every chunk
belongs to its document and carries the same `product`, `versions` and `doc_type`.

## Hard cases included

- Same title in two products: "Configure SAML Authentication", "Expression Samples"
  (`sequence` vs `orchestration`). Use these to test product filtering.
- Release notes (`doc_type="release-notes"`, with `release`). Use these to test down-weighting.
- Exact technical names (`Set-CoraSeQuenceSiteAuthentication`). Use these to test text search.
- A page with several versions (`["7.x", "8.x"]`) and a `general` page with no versions.

## Important

**v0 is synthetic.** The page bodies are short hand-written samples, not copies of the
real site. Don't use them as ground truth for answers or for the Golden Set.
v1 will contain 30-50 real crawled pages, in the same files and with the same schema,
so your code will not need to change.

Regenerate after changing the builder: `uv run python fixtures/build_samples_v0.py`
