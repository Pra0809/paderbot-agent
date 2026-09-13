# PaderBot Agent

A LangGraph rewrite of [PaderBot](https://github.com/Pra0809/Paderbot)'s query path.

The original runs one pass: rewrite the question, retrieve, check confidence, answer or refuse. If retrieval came back weak, that was it. The user got a refusal and no second attempt.

This version tries again. When retrieval looks poor, or when the model refuses even though retrieval looked fine, the agent reformulates the search query and re-runs. It only gives up once the retry cap is hit.

Related repos: [Paderbot](https://github.com/Pra0809/Paderbot) holds the retrieval and generation core plus the Streamlit demo. [paderbot-mcp](https://github.com/Pra0809/paderbot-mcp) exposes the same system as an MCP tool.

## The graph

```
START → rewrite_query → retrieve → grade
                           ^          │
                           │          ├─ retrieval ok                 → generate
                           │          ├─ not ok, retries left         → requery ─┐
                           │          └─ not ok, retries used up      → refuse   │
                           │                                                     │
                           └─────────────────────────────────────────────────────┘

generate → ├─ answered                            → END
           ├─ refused, retries left               → requery (back to retrieve)
           └─ refused, retries used up            → refuse

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

## Why two failure signals

Retrieval confidence and answer quality fail independently, so the graph watches both.

`grade` catches the obvious case: nothing came back, or nothing close enough. But retrieval can look confident and the answer still be a refusal, because the chunks simply didn't contain what was asked. `generate` catches that by checking what it produced against a list of refusal markers.

Both paths lead into the same `requery` node and share one `retry_count`. A weak retrieval and a refusal-shaped answer get the same corrective attempt.

`refuse` then overwrites whatever the model wrote with fixed template text and empties `sources`. A refusal can't cite pages it never used.

## Why the cap is 2

One retry can't tell a bad original query apart from a bad reformulation. If two separately reformulated queries both come back empty, more attempts mostly buy latency and tokens.

Worst case is three retrieval passes and three generation calls.

Both conditional edges check the same `MAX_RETRIES`, so the loop can't be re-entered through the other branch.

## State

One `TypedDict`. Each node reads what it needs and returns a partial update, which LangGraph merges by plain overwrite. No reducers, because nothing here accumulates.

Inputs are set once by the caller and never written by a node: `question` stays verbatim and is never rewritten in place, `language` restricts retrieval to `en`, `de` or both, and `force_answer_language` pins the output language regardless of what the corpus or the question are in.

The retrieval loop owns `retrieval_query`, `chunks`, `retrieval_ok` and `retry_count`. Output is `answer`, `sources` and `refused`, filled in as the graph progresses.

`RetrievedChunk` is imported from `paderbot-api` instead of being redefined here, so the schema can't drift from what retrieval actually returns.

## What lives where

Embedding, BM25, RRF fusion, the confidence threshold and the rewrite heuristic all stay in `paderbot-api`, wired in as an editable dependency. The nodes are thin wrappers. Two copies of retrieval logic would drift apart.

Two small heuristics are ported rather than delegated: the refusal-marker list and the German/English detection that picks the refusal's language. Both live inline in `paderbot.py`'s `query()` and aren't exposed as methods. They're short and unlikely to change, but it's duplication worth knowing about.

`index.PAGES_PATH` gets repointed at import. It resolves relative to the caller's working directory, which is fine inside `paderbot-api` but not here, so it's derived from the installed module's own path instead of hardcoded.

## Running it

```bash
uv sync
export GROQ_API_KEY="your_key"

uv run python main.py "What is BAföG?"
```

Prints the answer, then ranked sources with URLs, or `Sources: none` on a refusal.

The Chroma index is built at import if it's missing, from `paderbot-api`'s `data/pages.jsonl`. This repo keeps its own copy. It's a disposable cache and rebuilds idempotently.

## Limitations

The retry loop has not been benchmarked against the single-pass version. The 30-question evaluation in the PaderBot repo scores the original pipeline. Whether retrying improves faithfulness, or mainly turns refusals into weakly grounded answers, is still unmeasured. That comparison is the next thing to run.

Refusal detection is string matching on known phrasings. A refusal worded outside that list will pass as a real answer.

Retries cost latency. A question that ends in a refusal anyway can triple the LLM calls getting there.

Single-turn only, inherited from PaderBot. No conversational memory, and no API or UI layer here. The Streamlit and FastAPI surfaces live in the other repos.

## Stack

Python 3.12, LangGraph, langchain-groq, ChromaDB, rank-bm25, sentence-transformers with multilingual-e5-base, and Groq for inference on Llama 3.1 8B and 3.3 70B.
