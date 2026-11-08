# KB-Chat — Knowledge Center Assistant

RAG assistant that answers questions about Genpact Cora products from the
[Cora Knowledge Center](https://knowledgecenter.gcora.genpact.com/help), with citations and links.

See [CLAUDE.md](CLAUDE.md) for the full project context: architecture, stack, domain ownership,
contracts and working rules.

## Getting started

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # fill in secrets locally
uv run pytest           # run tests
uv run ruff check .     # lint
uv run mypy             # type-check
```
