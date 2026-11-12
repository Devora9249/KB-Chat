"""TEMPORARY stand-ins for shared contracts that do not exist yet.

TODO(D11): replace every definition here with imports from `contracts/` once the
module layout (D11) and Filters (D10) are agreed. Then delete this file.

Rules:
- Import ONLY from inside `src/kc/indexing/` and from tests/indexing (Raizy's code).
  Other domains must never import this module.
- All Storage & Indexing code imports contract types from HERE, so the switch to
  `contracts/` is a one-line change per name.
"""

from typing import Literal

# =====================================================================================
# From my storage contract draft (D2), not yet approved by the team
# =====================================================================================

# Required, keyword-only argument of Embedder.embed.
InputType = Literal["query", "document"]
