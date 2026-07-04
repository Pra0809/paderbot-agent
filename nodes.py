"""
nodes.py — Graph node functions for the PaderBot self-correcting agent.

Each node is `State -> dict`: it reads only the fields it needs and returns
a partial update, which LangGraph merges into the running state.

Retrieval/generation logic is delegated to PaderBot (imported from the
paderbot-api package, wired in here as an editable dependency) rather than
reimplemented — embedding, BM25, RRF fusion, and the confidence/rewrite
heuristics stay single-sourced in paderbot-api instead of drifting across
two copies.
"""

from pathlib import Path

import index
from paderbot import GROQ_MODEL_REWRITE, PaderBot

from state import State

# index.PAGES_PATH is "data/pages.jsonl", resolved relative to the caller's
# cwd — fine inside paderbot-api's own repo/container, but paderbot-agent
# runs from its own directory. Source data is owned by paderbot-api, so
# point at the data folder next to wherever the (editable) package actually
# lives, derived from the installed module's own path — no hardcoded
# absolute path, so this keeps working regardless of where paderbot-api is
# checked out or how it's installed.
index.PAGES_PATH = Path(index.__file__).parent / "data" / "pages.jsonl"

# Built once at import time, same lifecycle as paderbot-api's own FastAPI
# lifespan hook: ensure the Chroma index exists (building it from
# data/pages.jsonl if missing), then construct the shared PaderBot instance
# every node delegates to. The built index itself (./chroma_db) is local to
# wherever this process runs — paderbot-agent gets its own copy, which is
# fine since it's a disposable, idempotently-rebuildable cache.
index.ensure_index()
_bot = PaderBot()


# ============================================================
# Node: rewrite_query
# ============================================================
def rewrite_query(state: State) -> dict:
    """
    Gated query rewrite: only rewrite if the question looks vague, otherwise
    use it verbatim as the retrieval query. Runs once, at graph entry — the
    `requery` node is what reformulates `retrieval_query` on retry attempts.
    """
    retrieval_query = _bot.maybe_rewrite_query(state["question"])
    return {"retrieval_query": retrieval_query}


# ============================================================
# Node: retrieve
# ============================================================
def retrieve(state: State) -> dict:
    """
    Hybrid retrieval: dense (e5) + BM25, fused with RRF. Reads
    `retrieval_query` (not `question` — the rewritten/requeried form) and
    the `language` filter. Overwrites `chunks` on every call, including
    retry attempts from the requery loop.
    """
    chunks = _bot.retrieve(state["retrieval_query"], language=state["language"])
    return {"chunks": chunks}


# ============================================================
# Node: requery
# ============================================================
def requery(state: State) -> dict:
    """
    Reformulate the retrieval query after a failed `grade` or a
    refusal-shaped `generate`. Unlike `rewrite_query`'s gate, this always
    calls the LLM — reaching this node already means the initial attempt
    (rewritten or not) didn't work — and includes the previous
    `retrieval_query` so the model tries genuinely different phrasing
    instead of repeating it. Reads `question`, `retrieval_query`,
    `retry_count`; writes a new `retrieval_query` and increments
    `retry_count` for the conditional edges' MAX_RETRIES check.
    """
    question = state["question"]
    previous_query = state["retrieval_query"]

    prompt = (
        "Your previous search query for this question did not return good results. "
        "Write ONE new, differently-phrased search query (max 12 words) optimized for "
        "semantic search over a Paderborn University knowledge base — use different terms "
        "or broaden it, don't just repeat the previous attempt. "
        "Output ONLY a single line, no commentary, no quotes.\n\n"
        f"QUESTION: {question}\n"
        f"PREVIOUS QUERY (didn't work well): {previous_query}"
    )
    try:
        resp = _bot.groq.chat.completions.create(
            model=GROQ_MODEL_REWRITE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )
        retrieval_query = resp.choices[0].message.content.strip().strip('"').strip("'")
    except Exception:
        # If reformulation fails for any reason, retry the same query rather
        # than crashing the graph — the retry cap still bounds the loop.
        retrieval_query = previous_query

    return {"retrieval_query": retrieval_query, "retry_count": state["retry_count"] + 1}


# ============================================================
# Node: grade
# ============================================================
def grade(state: State) -> dict:
    """
    Retrieval-confidence gate: delegates to PaderBot.should_refuse (refuse if
    no chunks, or best dense distance exceeds MAX_DISTANCE). Reads `chunks`;
    writes `retrieval_ok` for the conditional edge (in agent.py) to route to
    generate / requery / refuse.
    """
    retrieval_ok = not _bot.should_refuse(state["chunks"])
    return {"retrieval_ok": retrieval_ok}


# ============================================================
# Node: generate
# ============================================================
# Not exposed as a PaderBot method — paderbot.py's query() does this inline
# after calling generate_answer(), so it's ported here rather than delegated.
# Small, self-contained heuristic (marker-string list); much lower drift risk
# than the retrieval logic, but a duplication worth knowing about.
_REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "nicht genug informationen",
    "habe keine informationen",
    "i don't know based on",
]


def generate(state: State) -> dict:
    """
    Grounded generation over the current `chunks`. Reads `question`,
    `chunks`, `force_answer_language`; writes `answer`, `sources`, and
    `refused`. Even when retrieval looked confident, the LLM itself may
    still produce a refusal-shaped answer (context didn't actually contain
    the answer) — that's detected here and drives the same
    requery/refuse routing as a failed `grade`.
    """
    question = state["question"]
    chunks = state["chunks"]

    answer = _bot.generate_answer(question, chunks, force_language=state["force_answer_language"])

    answer_low = answer.lower()
    refused = any(m in answer_low for m in _REFUSAL_MARKERS)

    sources = []
    if not refused:
        sources = [
            {"rank": c.rank, "url": c.url, "title": c.title, "language": c.language}
            for c in chunks
        ]

    return {"answer": answer, "sources": sources, "refused": refused}


# ============================================================
# Node: refuse
# ============================================================
# Same as generate's marker list above: this language-selection heuristic
# lives inline in paderbot.py's query(), not behind a method, so it's ported
# rather than delegated.
_GERMAN_MARKERS = [" ist ", " ich ", " sind ", " was ", " wie ", " wo ", " wer ", " welche ", "?"]

_REFUSAL_TEXT = {
    "de": "Ich habe nicht genug Informationen, um diese Frage anhand der verfügbaren Quellen zu beantworten.",
    "en": "I don't have enough information to answer that based on the available sources.",
}


def refuse(state: State) -> dict:
    """
    Terminal refusal: reached when retries are exhausted, whether from a
    failed `grade` or a refusal-shaped `generate`. Reads `question` and
    `force_answer_language` to pick the refusal's language; writes a clean
    templated `answer` (overwriting any refusal-shaped LLM output),
    `sources: []`, and `refused: True`.
    """
    force_answer_language = state["force_answer_language"]

    if force_answer_language in ("de", "en"):
        lang = force_answer_language
    else:
        q = f" {state['question'].lower()} "
        de_hits = sum(1 for m in _GERMAN_MARKERS if m in q)
        lang = "de" if de_hits >= 2 else "en"

    return {"answer": _REFUSAL_TEXT[lang], "sources": [], "refused": True}
