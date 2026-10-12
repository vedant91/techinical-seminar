"""Scripted seminar demo for the Agentic RAG pipeline.

Run with no arguments to ingest the sample documents and ask the demo
questions; or pass --ingest / --ask to drive it manually:

    python demo/run_demo.py
    python demo/run_demo.py --ingest docs/my_doc.md --ask "your question"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_rag import AgenticRAG  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent / "sample_docs"

# (kind, question) - each kind exercises a different pipeline strength.
DEMO_QUESTIONS = [
    ("vector", "What encryption standards does the platform use to protect data at rest?"),
    ("structural", "What does section 4 of the product spec say about pricing?"),
    ("multi-hop", "How do the security audit findings compare with the encryption promises in the product spec?"),
    ("honest-failure", "What warranty does Acme provide for its on-premise appliance hardware?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG demo")
    parser.add_argument("--ingest", nargs="*", default=None, help="document files to ingest")
    parser.add_argument("--ask", action="append", default=None, help="question to ask")
    parser.add_argument("--verbose", action="store_true", help="print retrieved chunks")
    args = parser.parse_args()

    rag = AgenticRAG()

    if args.ingest is not None or args.ask is not None:
        for doc in args.ingest or []:
            AgenticRAG.print_result(rag.ingest_document(doc))
        for question in args.ask or []:
            AgenticRAG.print_result(rag.ask(question), verbose=args.verbose)
        return

    print("=" * 76)
    print("AGENTIC RAG - SEMINAR DEMO")
    print("=" * 76)

    docs = sorted(DOCS_DIR.glob("*.md"))
    print(f"\n[1/2] Ingesting {len(docs)} sample documents "
          "(ingestion -> chunking -> chunk validation)...")
    for doc in docs:
        AgenticRAG.print_result(rag.ingest_document(doc))

    print("\n[2/2] Querying (hybrid retrieval -> retrieval validation -> answer)...")
    for kind, question in DEMO_QUESTIONS:
        print("\n" + "=" * 76)
        print(f"QUERY [{kind}]: {question}")
        print("=" * 76)
        AgenticRAG.print_result(rag.ask(question), verbose=args.verbose)


if __name__ == "__main__":
    main()
