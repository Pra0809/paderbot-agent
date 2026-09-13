# PaderBot Agent — self-correcting retrieval with LangGraph

A LangGraph rewrite of [PaderBot](https://github.com/Pra0809/Paderbot)'s query path, adding a retry loop when retrieval or generation fails.

**Core system:** [Paderbot](https://github.com/Pra0809/Paderbot) (retrieval, generation, Streamlit demo) · **MCP server:** [paderbot-mcp](https://github.com/Pra0809/paderbot-mcp)

---

## Why this exists

The original PaderBot runs one pass: rewrite the question, retrieve, check confidence, answer or refuse. If retrieval came back weak, that was it — the user got a refusal and no second attempt, even when a differently worded search would have found the page.

This version tries again. When retrieval looks poor, or when the model refuses despite retrieval looking fine, the agent reformulates the search query and re-runs it. It only gives up once the retry cap is hit.

---

## How it works

```
START → rewrite_query → retrieve → grade
                           ^          │
                           │          ├─ retrieval ok              → generate
                           │          ├─ not ok, retries left      → requery ─┐
                           │          └─ not ok, retries used up   → refuse   │
                           │                                                  │
                           └──────────────────────────────────────────────────┘

generate → ├─ answered                        → END
           ├─ refused, retries left           → requery (back to retrieve)
           └─ refused, retries used up        → refuse

refuse → END
```

Six nodes, two conditional edges, one retry counter shared between them.

| Node | What it does |
| --- | --- |
| `rewrite_query` | Runs once at entry, and only if the question looks vague. Short specific questions go through untouched, because an ablation showed rewriting hurt exact-term lookups. |
| `retrieve` | Dense e5 embeddings plus BM25, fused with Reciprocal Rank Fusion. Reads `retrieval_query` rather than `question`, so a retry actually searches something new. |
| `grade` | Confidence gate. Fails if nothing came back, or if the best dense distance is above threshold. Writes `retrieval_ok` for the router. |
| `requery` | Reformulates after a failure. Always calls the LLM, and feeds it the query that just failed so it tries different wording instead of repeating itself. Bumps `retry_count`. |
| `generate` | Grounded generation over the current chunks. Also checks its own output for refusal phrasing, in English and German. |
| `refuse` | Terminal. Writes templated refusal text in the right language and clears `sources`. |

**State.** One `TypedDict`. Each node reads what it needs and returns a partial update, which LangGraph merges by plain overwrite — no reducers, since nothing here accumulates. Inputs are set once by the caller and never written by a node: `question` stays verbatim, `language` restricts retrieval to `en` / `de` / both, `force_answer_language` pins the output language. The loop owns `retrieval_query`, `chunks`, `retrieval_ok`, `retry_count`. Output is `answer`, `sources`, `refused`.

`RetrievedChunk` is imported from `paderbot-api` rather than redefined, so the schema can't drift from what retrieval actually returns.

---

## Decisions

- **Two failure signals, one loop.** Retrieval confidence and answer quality fail independently. `grade` catches the obvious case — nothing came back, or nothing close enough. But retrieval can look confident and the answer still be a refusal, because the chunks didn't contain what was asked. `generate` catches that by checking its own output against a marker list. Both paths lead into the same `requery` node and share one `retry_count`.
- **Refusals are templated, not generated.** `refuse` overwrites whatever the model wrote with fixed text and empties `sources`, so a refusal can't cite pages it never used.
- **Retry cap of 2.** One retry can't tell a bad original query apart from a bad reformulation. If two separately reformulated queries both come back empty, more attempts mostly buy latency and tokens. Worst case is three retrieval passes and three generation calls. Both conditional edges check the same `MAX_RETRIES`, so the loop can't be re-entered through the other branch.
- **Retrieval logic stays in `paderbot-api`.** Embedding, BM25, RRF fusion, the confidence threshold and the rewrite heuristic are all wired in as an editable dependency. The nodes are thin wrappers — two copies of retrieval logic would drift apart.
- **Two heuristics are ported, not delegated.** The refusal-marker list and the German/English detection that picks the refusal's language live inline in `paderbot.py`'s `query()` and aren't exposed as methods. Short and unlikely to change, but it's duplication worth knowing about.
- **`index.PAGES_PATH` is repointed at import.** It resolves relative to the caller's working directory, which is fine inside `paderbot-api` but not here, so it's derived from the installed module's own path instead of hardcoded.

---

## Running it locally

```bash
uv sync
export GROQ_API_KEY="your_key"

uv run python main.py "What is BAföG?"
```

Prints the answer, then ranked sources with URLs, or `Sources: none` on a refusal.

The Chroma index is built at import if missing, from `paderbot-api`'s `data/pages.jsonl`. This repo keeps its own copy — a disposable cache that rebuilds idempotently.

---

## Project layout

| File | Purpose |
| --- | --- |
| `agent.py` | Graph construction, conditional edges, `MAX_RETRIES` |
| `nodes.py` | The six node functions |
| `state.py` | `State` schema |
| `main.py` | CLI entrypoint |

---

## Limitations & future work

- **Not benchmarked against the single-pass version.** The 30-question evaluation in the PaderBot repo scores the original pipeline. Whether retrying improves faithfulness, or mainly turns refusals into weakly grounded answers, is unmeasured. That comparison is the next thing to run.
- **Refusal detection is string matching.** A refusal worded outside the marker list passes as a real answer.
- **Retries cost latency.** A question that ends in a refusal anyway can triple the LLM calls getting there.
- **Single-turn**, inherited from PaderBot. No conversational memory.
- **CLI only.** No API or UI layer here — the Streamlit and FastAPI surfaces live in the other repos.

---

## Stack

Python 3.12 · LangGraph · langchain-groq · ChromaDB · rank-bm25 · sentence-transformers (multilingual-e5-base) · Groq (Llama 3.1 8B / 3.3 70B)
