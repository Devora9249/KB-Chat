"""Shared metadata types used across contracts (Document, Chunk, Filters, SearchResult).

Closed value sets, so mypy and Pydantic catch invalid values (e.g. "Sequence", "9.x")
on both the producing side (Enricher) and the consuming side (search filters).
Adding a value is a backward-compatible contract change; removing or renaming one is not.
"""

from typing import Literal

# Product as it appears on the site. Legacy names are NOT mapped to new ones
# (SeQuence -> Orchestration is resolved in Query Understanding, not in the data).
Product = Literal["sequence", "orchestration", "case-manager", "opsmanager"]

# Document type. "release-notes" is stable (down-weighted in search);
# the list may grow after the full crawl.
DocType = Literal[
    "how-to",
    "reference",
    "installation",
    "release-notes",
    "troubleshooting",
    "overview",
    "other",
]

# Major version only, derived from the breadcrumb. Pages not tied to a version use None
# (callers annotate as `Version | None`).
Version = Literal["11", "10", "9", "8", "7", "4"]
