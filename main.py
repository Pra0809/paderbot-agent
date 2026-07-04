"""
main.py — CLI entrypoint for the PaderBot self-correcting agent.

Usage:
    uv run python main.py "What is BAföG?"
"""

import os
import sys


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 2:
        print('Usage: uv run python main.py "<question>"', file=sys.stderr)
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    # Deferred: importing agent builds the Chroma index and constructs
    # PaderBot, which is slow and itself requires GROQ_API_KEY — the check
    # above gives a clean error instead of a raw import-time traceback.
    from agent import build_graph

    graph = build_graph()

    initial_state = {
        "question": question,
        "language": None,
        "force_answer_language": None,
        "retry_count": 0,
    }

    result = graph.invoke(initial_state)

    print(result["answer"])
    print()
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['rank']}] {s['title']} ({s['language']}) — {s['url']}")
    else:
        print("Sources: none")


if __name__ == "__main__":
    main()
