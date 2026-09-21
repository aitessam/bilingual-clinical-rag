#!/usr/bin/env python
"""CLI to ask the bilingual clinical RAG agent a question and see the
grounded answer with citations.

Usage:
    python scripts/ask.py "question in English or Arabic"
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.agent.graph import ask  # noqa: E402
from src.agent.nodes import DEFAULT_GROQ_MODEL  # noqa: E402
from src.retrieval.hybrid import (  # noqa: E402
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANK_N,
    DEFAULT_RETRIEVE_K,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the bilingual clinical RAG agent a question.")
    parser.add_argument("query", help="Question, in English or Arabic")
    parser.add_argument("-k", "--retrieve-k", type=int, default=DEFAULT_RETRIEVE_K)
    parser.add_argument("-n", "--rerank-n", type=int, default=DEFAULT_RERANK_N)
    parser.add_argument("--chunks-path", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cross-encoder", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_GROQ_MODEL)
    args = parser.parse_args()

    if not Path(args.chunks_path).exists():
        print(
            f"No chunks file found at '{args.chunks_path}'. "
            "Run the ingestion pipeline first, e.g.:\n"
            "  python -c \"from src.ingestion.chunking import ingest; "
            f"ingest('data/raw', '{args.chunks_path}')\""
        )
        raise SystemExit(1)

    result = ask(
        args.query,
        retrieve_k=args.retrieve_k,
        rerank_n=args.rerank_n,
        chunks_path=args.chunks_path,
        embedding_model=args.embedding_model,
        cross_encoder_model=args.cross_encoder,
        llm_model=args.llm_model,
    )

    print(f"Query: {args.query!r}")
    print(f"Detected language: {result.get('language')}\n")
    print(result.get("final_response", "(no response)"))


if __name__ == "__main__":
    main()
