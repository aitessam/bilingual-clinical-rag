#!/usr/bin/env python
"""CLI to run hybrid (BM25 + FAISS) search, then cross-encoder reranking,
over the ingested chunk corpus.

Usage:
    python scripts/query.py "some question" [-k 20] [-n 5]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.retrieval.hybrid import (  # noqa: E402
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANK_N,
    DEFAULT_RETRIEVE_K,
    hybrid_search_with_rerank,
)


def _preview(text: str, length: int = 220) -> str:
    return " ".join(text.split())[:length]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the bilingual clinical hybrid retriever.")
    parser.add_argument("query", help="Query text, in English or Arabic")
    parser.add_argument(
        "-k", "--retrieve-k", type=int, default=DEFAULT_RETRIEVE_K,
        help="Number of fused candidates to retrieve before reranking",
    )
    parser.add_argument(
        "-n", "--rerank-n", type=int, default=DEFAULT_RERANK_N,
        help="Number of top candidates to show after reranking",
    )
    parser.add_argument("--chunks-path", default=DEFAULT_CHUNKS_PATH, help="Path to persisted chunks JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence-transformers model name")
    parser.add_argument("--cross-encoder", default=DEFAULT_CROSS_ENCODER_MODEL, help="Cross-encoder model name")
    args = parser.parse_args()

    if not Path(args.chunks_path).exists():
        print(
            f"No chunks file found at '{args.chunks_path}'. "
            "Run the ingestion pipeline first, e.g.:\n"
            "  python -c \"from src.ingestion.chunking import ingest; "
            f"ingest('data/raw', '{args.chunks_path}')\""
        )
        raise SystemExit(1)

    result = hybrid_search_with_rerank(
        args.query,
        retrieve_k=args.retrieve_k,
        rerank_n=args.rerank_n,
        chunks_path=args.chunks_path,
        model_name=args.model,
        cross_encoder_model=args.cross_encoder,
    )

    print(f"Query: {args.query!r}\n")

    if not result.retrieved:
        print("No results found.")
        return

    print(f"--- Pre-rerank: top {len(result.retrieved)} hybrid (RRF) candidates ---\n")
    for rank, hybrid_result in enumerate(result.retrieved, start=1):
        chunk = hybrid_result.chunk
        print(
            f"[{rank}] rrf_score={hybrid_result.score:.4f}  source={chunk.source}  "
            f"lang={chunk.language}  chunk_index={chunk.chunk_index}  "
            f"(dense_rank={hybrid_result.dense_rank}, sparse_rank={hybrid_result.sparse_rank})"
        )
        print(f"    {_preview(chunk.text)}...\n")

    print(f"--- Post-rerank: top {len(result.reranked)} after cross-encoder ---\n")
    for rank, reranked_result in enumerate(result.reranked, start=1):
        chunk = reranked_result.chunk
        print(
            f"[{rank}] rerank_score={reranked_result.rerank_score:.4f}  source={chunk.source}  "
            f"lang={chunk.language}  chunk_index={chunk.chunk_index}  "
            f"(was pre-rerank rank {reranked_result.previous_rank})"
        )
        print(f"    {_preview(chunk.text)}...\n")


if __name__ == "__main__":
    main()
