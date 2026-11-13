"""Build fixtures v0: hand-written sample pages in the real contract shape.

SYNTHETIC CONTENT: the page bodies below are short, hand-written samples that imitate
the Knowledge Center structure. They are NOT copies of the real pages and must not be
used as ground truth for answers (e.g. in the Golden Set). They will be replaced by
real crawled pages (same files, same schema) once the crawler is ready.

Each sample covers one "hard case" from CLAUDE.md (cross-product duplicates, release
notes, exact technical names, multi-version pages, product-independent pages).

Run:  uv run python fixtures/build_samples_v0.py
"""

import json
from pathlib import Path
from typing import Any

from contracts import Chunk, Document
from kc.ingestion.ids import content_hash, make_chunk_id

BASE = "https://knowledgecenter.gcora.genpact.com/help/"
OUT_DIR = Path(__file__).parent

SAMPLES: list[dict[str, Any]] = [
    {  # Exact technical name (PowerShell) + cross-product duplicate title (1/2)
        "slug": "configure-saml-authentication-sequence",
        "title": "Configure SAML Authentication",
        "breadcrumb": ["Cora SeQuence 9.x", "Administration", "Authentication"],
        "product": "sequence",
        "versions": ["9.x"],
        "doc_type": "how-to",
        "body": (
            "Use SAML to let users sign in with your identity provider.\n\n"
            "## Prerequisites\n\n"
            "- An identity provider that supports SAML 2.0.\n"
            "- Administrator access to the SeQuence server.\n\n"
            "## Enable SAML\n\n"
            "Run the following PowerShell command on the web server:\n\n"
            "```powershell\n"
            "Set-CoraSeQuenceSiteAuthentication -SiteName Runtime -AuthenticationType SAML\n"
            "```\n\n"
            "## Verify\n\n"
            "Open the Runtime site and confirm you are redirected to the identity provider."
        ),
    },
    {  # Cross-product duplicate title (2/2)
        "slug": "configure-saml-authentication-orchestration",
        "title": "Configure SAML Authentication",
        "breadcrumb": ["Cora Orchestration 10.x", "Administration", "Authentication"],
        "product": "orchestration",
        "versions": ["10.x"],
        "doc_type": "how-to",
        "body": (
            "In Cora Orchestration, SAML is configured from the Administration site.\n\n"
            "## Enable SAML\n\n"
            "1. Go to **Administration > Authentication**.\n"
            "2. Select **SAML** and upload the identity provider metadata file.\n\n"
            "## Verify\n\n"
            "Sign out and sign in again to confirm the redirect."
        ),
    },
    {  # Cross-product duplicate: Expression Samples (1/2)
        "slug": "expression-samples-sequence",
        "title": "Expression Samples",
        "breadcrumb": ["Cora SeQuence 9.x", "Developer Guide", "Expressions"],
        "product": "sequence",
        "versions": ["9.x"],
        "doc_type": "reference",
        "body": (
            "Common expressions used in SeQuence workflows.\n\n"
            "## String expressions\n\n"
            "`String.Concat(a, b)` joins two strings.\n\n"
            "## Date expressions\n\n"
            "`DateTime.Now.AddDays(1)` returns tomorrow's date."
        ),
    },
    {  # Cross-product duplicate: Expression Samples (2/2)
        "slug": "expression-samples-orchestration",
        "title": "Expression Samples",
        "breadcrumb": ["Cora Orchestration 10.x", "Developer Guide", "Expressions"],
        "product": "orchestration",
        "versions": ["10.x"],
        "doc_type": "reference",
        "body": (
            "Common expressions used in Orchestration workflows.\n\n"
            "## String expressions\n\n"
            '`String.Join(",", items)` joins a list of strings.\n\n'
            "## Date expressions\n\n"
            "`DateTime.UtcNow` returns the current UTC time."
        ),
    },
    {  # Release notes with an exact release
        "slug": "corasequence-992-release-notes",
        "title": "Cora SeQuence 9.9.2 Release Notes",
        "breadcrumb": ["Cora SeQuence 9.x", "Release Notes"],
        "product": "sequence",
        "versions": ["9.x"],
        "release": "9.9.2",
        "doc_type": "release-notes",
        "body": (
            "## New features\n\n"
            "- Improved performance of the Flowtime task list.\n\n"
            "## Resolved issues\n\n"
            "- Fixed an issue where SAML sign-in failed after a session timeout."
        ),
    },
    {  # Release notes, newer product
        "slug": "cora-orchestration-108-release-notes",
        "title": "Cora Orchestration 10.8 Release Notes",
        "breadcrumb": ["Cora Orchestration 10.x", "Release Notes"],
        "product": "orchestration",
        "versions": ["10.x"],
        "release": "10.8",
        "doc_type": "release-notes",
        "body": (
            "## New features\n\n"
            "- Workflows can be exposed as MCP tools.\n\n"
            "## Breaking changes\n\n"
            "- The legacy REST endpoint `/api/v1/tasks` was removed."
        ),
    },
    {  # Troubleshooting
        "slug": "error-handler-activity-fails",
        "title": "Error Handler Activity Fails",
        "breadcrumb": ["Cora Orchestration 10.x", "Troubleshooting"],
        "product": "orchestration",
        "versions": ["10.x"],
        "doc_type": "troubleshooting",
        "body": (
            "## Symptom\n\n"
            "The Error Handler activity ends with status Failed.\n\n"
            "## Cause\n\n"
            "The activity has no connection to a fallback step.\n\n"
            "## Resolution\n\n"
            "Connect the Error Handler output to a fallback step and republish the workflow."
        ),
    },
    {  # Installation page that applies to several versions
        "slug": "install-the-sequence-database-using-scripts",
        "title": "Install the SeQuence Database Using Scripts",
        "breadcrumb": ["Cora SeQuence 8.x", "Installation", "Versions 7.7 - 8.2"],
        "product": "sequence",
        "versions": ["7.x", "8.x"],
        "doc_type": "installation",
        "body": (
            "Install the database manually when the installer cannot reach SQL Server.\n\n"
            "## Run the scripts\n\n"
            "```powershell\n"
            "Install-CoraSeQuenceDatabase -Server sql01 -Database SeQuence\n"
            "```\n\n"
            "## Next steps\n\n"
            "Continue with the web server installation."
        ),
    },
    {  # Case Manager
        "slug": "create-a-case-type",
        "title": "Create a Case Type",
        "breadcrumb": ["Cora Case Manager 4.x", "Configuration"],
        "product": "case-manager",
        "versions": ["4.x"],
        "doc_type": "how-to",
        "body": (
            "A case type defines the fields and stages of a case.\n\n"
            "## Create the case type\n\n"
            "1. Go to **Settings > Case Types**.\n"
            "2. Click **New** and enter a name.\n\n"
            "## Add stages\n\n"
            "Add at least one stage before you publish the case type."
        ),
    },
    {  # Product-independent page, no versions
        "slug": "glossary",
        "title": "Glossary",
        "breadcrumb": ["Glossary"],
        "product": "general",
        "versions": [],
        "doc_type": "concept",
        "body": (
            "## Activity\n\n"
            "A single step in a workflow.\n\n"
            "## Workflow\n\n"
            "A sequence of activities that automates a business process."
        ),
    },
]


def split_by_h2(title: str, body: str) -> list[tuple[list[str], str]]:
    """Very simple v0 chunker: one chunk per '## ' section (text before it = intro)."""
    sections: list[tuple[list[str], list[str]]] = [([title], [])]
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append(([title, line[3:].strip()], []))
        else:
            sections[-1][1].append(line)
    return [(path, "\n".join(lines).strip()) for path, lines in sections if "".join(lines).strip()]


def build() -> tuple[list[Document], list[Chunk]]:
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for s in SAMPLES:
        content_md = f"# {s['title']}\n\n{s['body']}"
        doc = Document(
            url=BASE + s["slug"],
            title=s["title"],
            breadcrumb=s["breadcrumb"],
            product=s["product"],
            versions=s["versions"],
            release=s.get("release"),
            doc_type=s["doc_type"],
            content_md=content_md,
            content_hash=content_hash(content_md),
        )
        documents.append(doc)
        for heading_path, text in split_by_h2(s["title"], s["body"]):
            chunks.append(
                Chunk(
                    id=make_chunk_id(doc.url, heading_path, part=0),
                    document_url=doc.url,
                    text=text,
                    heading_path=heading_path,
                    product=doc.product,
                    versions=doc.versions,
                    doc_type=doc.doc_type,
                )
            )
    return documents, chunks


def write_jsonl(path: Path, rows: list[Document] | list[Chunk]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    docs, chs = build()
    write_jsonl(OUT_DIR / "sample_docs.jsonl", docs)
    write_jsonl(OUT_DIR / "sample_chunks.jsonl", chs)
    print(f"wrote {len(docs)} documents and {len(chs)} chunks to {OUT_DIR}")
