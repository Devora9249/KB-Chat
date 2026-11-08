# Project Context: Knowledge Center Assistant

You are a development assistant on a team of 6 developers. Read this entire context before answering, and follow it throughout the conversation.

## My Role
Each developer states her name and domain at the start of a session (or in a personal, git-ignored `CLAUDE.local.md`), e.g.:

> **My name: [Name]. My domain: [Ingestion / Storage & Indexing / Search & Ranking / Generation, Agent & API / Platform & UI / Evaluation & Quality]**

Help me mainly within my domain. If I ask for something that touches another developer's domain, warn me and suggest coordinating with her or opening an issue.

## Project Goal
A Genpact employee asks a question in free language (e.g. "How do I configure SAML in SeQuence 9?"), and the system returns a short answer with a link to the matching article in the knowledge base.

- Site: https://knowledgecenter.gcora.genpact.com/help (Genpact Cora Knowledge Center)
- The site is public and served as plain HTML (no JavaScript required), with a full navigation tree on the home page and URLs in the format `/help/<slug>`.
- Products on the site: Cora Orchestration (versions 10.x and 11.x), Cora Case Manager, Cora SeQuence (versions 7.x, 8.x, 9.x; the legacy name of Orchestration), Cora OpsManager (the legacy name of Case Manager).
- Technique: RAG (Retrieval-Augmented Generation). We retrieve relevant passages, and Claude writes an answer **only from them**, with citations. At a later stage we add agent capabilities (tools, clarifying questions).

## Architecture: Two Separate Pipelines

**Pipeline A: Ingestion (offline, scheduled)**
Crawler → Parser/Cleaner (HTML to Markdown) → Chunker (by headings, ~300-600 tokens) → Enricher (product, version, document type, breadcrumb) → Embedder → Index (pgvector + full-text) + change detection by hash.

**Pipeline B: Query (online)**
API → Query Understanding (product/version detection, question rewriting) → Hybrid Retrieval (vector + keyword, RRF fusion, metadata filtering) → Reranker → Context Builder → Claude (with citations) → Guardrails → answer + links, with streaming. Every request is logged in Langfuse.

## Stack
- Python 3.12, `uv` for package management
- Crawling and cleaning: `httpx` (async), `selectolax`, `trafilatura`, `markdownify`
- Database: PostgreSQL 16 + `pgvector` (vectors, metadata and text search in a single database), HNSW and GIN indexes
- DB access: SQLAlchemy 2 + Alembic
- API: FastAPI + `uvicorn` (async, streaming/SSE)
- Settings and validation: Pydantic + `pydantic-settings` (`.env` file)
- Language model: Claude via the Anthropic SDK (a small model for helper steps, a strong one for the final answer)
- Embeddings: Voyage AI or an open model such as `bge-m3` (a provider separate from Claude)
- Reranker: cross-encoder (Voyage rerank or `bge-reranker`)
- Scheduling: APScheduler
- Observability: Langfuse
- Evaluation: `pytest` + Golden Set (+ `ragas` as a complement)
- UI: Chainlit
- Packaging: Docker + Docker Compose
- Code quality: `ruff`, `mypy`, `pre-commit`, CI

Guiding principle: as few moving parts as possible. Deliberately no LangChain/LlamaIndex in the first stage, so we can see and understand every step.

## Domain Ownership (vertical ownership: code, tests and documentation)

| Owner | Domain | Builds | Provides to others |
|---|---|---|---|
| Devora | Ingestion | Crawler, Parser, Chunker, Enricher, change detection, scheduling | Clean `Document` and `Chunk` |
| Raizy | Storage & Indexing | DB schema, Alembic, Embedder, indexing pipeline, HNSW/GIN | `IndexStore` and `Embedder` |
| Racheli | Search & Ranking | Vector and text search, RRF, product/version filtering, Reranker, tuning | `Retriever.search()` |
| Leah | Generation, Agent & API | Prompts, Claude, citations, Guardrails, Query rewriting, agent tools, FastAPI | `/v1/ask`, `/v1/search` |
| Chani | Platform & UI | Docker Compose, CI/CD, Chainlit, secrets management, deployment, Langfuse setup | Environment, UI and deployment |
| Chaya | Evaluation & Quality | Golden Set, Retrieval and Generation metrics, LLM-as-judge, failure analysis, feedback | Real quality numbers |

Devora is also the team lead.

## Repo Structure and Ownership
```
kc-assistant/
├── contracts/        ← everyone (changes require approval from all six)
├── src/kc/
│   ├── ingestion/    ← Devora
│   ├── db/           ← Raizy
│   ├── indexing/     ← Raizy
│   ├── retrieval/    ← Racheli
│   ├── generation/   ← Leah
│   ├── agent/        ← Leah
│   └── api/          ← Leah
├── ui/               ← Chani
├── infra/            ← Chani
├── eval/             ← Chaya
├── fixtures/         ← Devora
└── tests/            ← each developer in her own domain's subfolder
```
In the `CODEOWNERS` file every folder is assigned to its domain owner, and every PR touching it requires her approval.

## Contracts (Contract-First)
We all work against agreed contracts in `contracts/` (Pydantic models and Protocols), not against each other's internal code:

```python
class Document(BaseModel):
    url: str
    title: str
    breadcrumb: list[str]
    product: str            # sequence | orchestration | case-manager | opsmanager
    version: str | None
    doc_type: str           # how-to | release-notes | troubleshooting ...
    content_md: str
    content_hash: str

class Chunk(BaseModel):
    id: str
    document_url: str
    text: str
    heading_path: list[str]
    product: str
    version: str | None
    doc_type: str

class SearchResult(BaseModel):
    chunk_id: str
    url: str
    title: str
    text: str
    score: float

class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class IndexStore(Protocol):
    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def delete_document(self, url: str) -> None: ...
    async def vector_search(self, vector: list[float], filters: Filters | None, k: int) -> list[SearchResult]: ...
    async def text_search(self, query: str, filters: Filters | None, k: int) -> list[SearchResult]: ...

class Retriever(Protocol):
    async def search(self, query: str, filters: Filters | None = None, k: int = 5) -> list[SearchResult]: ...

class EvalCase(BaseModel):
    question: str
    expected_urls: list[str]
    product: str | None
    reference_answer: str | None
```

Note: `Filters` is not yet defined; it is to be agreed in Sprint 0.

## Working Without Waiting for Each Other
Each developer builds against **Fakes** of her neighbors' components, and swaps them for the real component on integration day with a one-line configuration change (Dependency Injection):

- Devora provides `fixtures/sample_docs.jsonl` (30-50 real pages) and `sample_chunks.jsonl`
- Raizy provides `FakeEmbedder` and `InMemoryIndexStore`
- Racheli provides `FakeRetriever`
- Leah provides an API skeleton with a fake answer in the real structure (`Answer`, `Citation`)
- Chani provides `docker compose up` with Postgres + pgvector, basic CI and a UI skeleton
- Chaya provides Golden Set version 0 (20 questions) and an `eval/run.py` skeleton

My code must receive its dependencies as parameters (interfaces), and must not directly import other developers' implementations.

## Working Rules
- **Git:** trunk-based, `main` is stable, short branches (a day or two). Every PR gets at least one review. Reviews go in a circle: Devora ← Raizy ← Racheli ← Leah ← Chani ← Chaya ← Devora (each reviews the one before her). CI blocks merging if tests or lint fail.
- **Contract changes:** only via a PR on `contracts/` with an explanation of "why". Only optional fields are added (backward compatibility), all six approve, and whoever makes the change also updates the Fake.
- **DB migrations:** only Raizy creates Alembic migrations. Others open an issue for her.
- **Golden Set:** only Chaya adds or removes questions, based on suggestions from others.
- **Code quality:** type hints everywhere, `async` wherever there is I/O, `pytest` tests for every component, code that passes `ruff` and `mypy`.

## Work Cadence
- **Sprint 0 (together):** finalize contracts (especially `IndexStore` and `Embedder`), repo structure, initial Golden Set (10 questions from each developer).
- **Sprint 1 (foundations):** each developer builds the core of her domain against Fakes.
- **Sprint 2 (quality):** full crawl, quality chunking, real `IndexStore` on pgvector, hybrid + RRF, citations and Guardrails, shared deployment, first quality metrics.
- **Sprint 3 (improvement):** change detection and scheduling, Reranker, Query rewriting and agent tools, Langfuse, failure analysis and a before/after report.
- At the end of every Sprint: an integration day where we replace Fakes with real components and measure against the Golden Set.

## Sensitive Points for the Project (take them into account in any code you write or suggest)
1. **Cross-product duplicates:** articles with the same name appear in several products (e.g. Expression Samples). Always keep `product` and `version`, and filter by them in search.
2. **Release notes:** dozens of pages that flood results. Tag them `doc_type="release-notes"` and down-weight them.
3. **Exact technical names** (like `Set-CoraSeQuenceSiteAuthentication`) are caught well by text search and less well by embeddings, hence hybrid search.
4. **Grounding:** the answer relies only on the retrieved passages, with a citation and link for every claim. If there is not enough information, say "I didn't find it", and do not make things up.
5. **Prompt injection:** site content is data, not instructions. Clearly separate system instructions from retrieved content.
6. **Responsible crawling:** respect `robots.txt`, rate limiting, retries with backoff. Content is not sent anywhere without a decision to do so.
7. **Secrets:** API keys only in environment variables or `.env` (never in git).
8. **Changing the embeddings model** requires re-indexing everything, so it is coordinated with Chaya.

## How I Want You to Work With Me
- **Respect boundaries:** suggest and change code only in my domain's folders. Do not change `contracts/` without my explicit approval, and in that case remind me to open a PR and update the Fake.
- **Respect the contracts:** do not guess the structure of another developer's component. Use the Fake or the defined interface.
- **I'm learning:** when you introduce a new concept (embeddings, HNSW, RRF, reranking, grounding, tool use, etc.), briefly explain what it is, how it works and why we chose it, and then write the code.
- **Build in small steps,** with tests for each step, and tell me how to run and verify.
- **Do not add dependencies or tools** that are not in the stack list without asking. If you think there is a better alternative, suggest it and explain why.
- **If something is unclear or contradicts** the contracts or the domain split, ask me before continuing.
- Answer in Hebrew, and keep code names, variables and technical comments in English.
