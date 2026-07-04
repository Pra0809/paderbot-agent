"""
agent.py — Builds and compiles the PaderBot self-correcting LangGraph agent.

Graph:

    START -> rewrite_query -> retrieve -> grade
                                  ^          |
                                  |          |-- retrieval_ok=True             --> generate
                                  |          |-- retrieval_ok=False,
                                  |          |   retry_count<MAX_RETRIES       --> requery --+
                                  |          |-- retrieval_ok=False,                         |
                                  |          |   retry_count>=MAX_RETRIES      --> refuse    |
                                  |                                                          |
                                  +----------------------------------------------------------+

    generate -- refused=False                     --> END
             -- refused=True, retry_count<MAX_RETRIES  --> requery (loops back to retrieve)
             -- refused=True, retry_count>=MAX_RETRIES  --> refuse

    refuse -> END

Both conditional edges enforce the same MAX_RETRIES cap via `retry_count`, so
a bad retrieval and a refusal-shaped generation share one corrective loop
instead of two separate dead ends.
"""

from langgraph.graph import END, START, StateGraph

from nodes import generate, grade, refuse, requery, retrieve, rewrite_query
from state import State

# 2, not 1 or 5: one retry can't distinguish a bad query from a bad
# reformulation, but if two independently-reformulated queries both fail,
# further attempts are unlikely to change the outcome — they mostly just add
# latency/cost. Caps worst case at 3 retrieval passes, 3 generation calls.
MAX_RETRIES = 2


def route_after_grade(state: State) -> str:
    """Reads `retrieval_ok`, `retry_count`. Returns 'generate' | 'requery' | 'refuse'."""
    if state["retrieval_ok"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "requery"
    return "refuse"


def route_after_generate(state: State) -> str:
    """Reads `refused`, `retry_count`. Returns END | 'requery' | 'refuse'."""
    if not state["refused"]:
        return END
    if state["retry_count"] < MAX_RETRIES:
        return "requery"
    return "refuse"


def build_graph():
    graph = StateGraph(State)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("requery", requery)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("requery", "retrieve")
    graph.add_edge("refuse", END)

    # Explicit path maps: without one, LangGraph can't infer a conditional
    # edge's destinations ahead of time, so graph.get_graph().draw_mermaid()
    # silently draws a placeholder edge to __end__ instead of the real
    # fan-out. Runtime routing doesn't need this, but the diagram does.
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "requery": "requery", "refuse": "refuse"},
    )
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {END: END, "requery": "requery", "refuse": "refuse"},
    )

    return graph.compile()
