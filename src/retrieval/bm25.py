"""
Sparse lexical retrieval using BM25.

Builds and queries a BM25 index (via rank-bm25) over the chunked clinical
corpus. Tokenisation is a simple Unicode word-boundary split, which works
for both English and Arabic text without requiring language-specific
stemming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingestion.chunking import Chunk

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split `text` into word tokens (Unicode-aware)."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class SparseIndex:
    """A BM25 index over chunk texts."""

    chunks: list[Chunk]
    bm25: BM25Okapi

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return up to `k` chunks with the highest BM25 score for `query`."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]


def build_sparse_index(chunks: list[Chunk]) -> SparseIndex:
    """Tokenise all chunk texts and build a BM25Okapi index over them."""
    tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return SparseIndex(chunks=chunks, bm25=bm25)
