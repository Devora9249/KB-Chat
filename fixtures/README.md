# Fixtures (owner: Devora)

Sample Ingestion output in the real contract shape (`contracts/ingestion.py`),
so every domain can build and test without waiting for the crawler.

| File | Content |
|---|---|
| `sample_docs.jsonl` | 10 `Document`s, one JSON object per line |
| `sample_chunks.jsonl` | 28 `Chunk`s belonging to those documents |
| `build_samples_v0.py` | Script that generates both files |

## How to use

```python
from kc.ingestion.samples import load_sample_chunks, load_sample_documents

docs = load_sample_documents()   # list[Document]
chunks = load_sample_chunks()    # list[Chunk]
```

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
