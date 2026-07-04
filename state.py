"""
state.py — Graph state schema for the PaderBot self-correcting agent.

State flows through the graph as a single dict conforming to `State`.
Each node reads the fields it needs and returns a partial update; LangGraph
merges that update into the running state (plain overwrite — no reducers,
since nothing here needs to accumulate across nodes).
"""

from typing import Optional, TypedDict

# The real dataclass from paderbot-api (wired in as an editable dependency) —
# not redefined here, so this can't drift from what PaderBot.retrieve() actually returns.
from paderbot import RetrievedChunk


class State(TypedDict):
    # --- inputs: set once by the caller, never written by a node ---
    question: str  # the user's original question, verbatim; never rewritten in place
    language: Optional[str]  # retrieval filter — restricts chunk search to "en"/"de"/None (both)
    force_answer_language: Optional[str]  # output filter — forces the answer's language regardless of what the corpus chunks or question are in

    # --- retrieval loop ---
    retrieval_query: str
    chunks: list[RetrievedChunk]
    retrieval_ok: bool
    retry_count: int

    # --- output, finalized progressively ---
    answer: str
    sources: list[dict]
    refused: bool
