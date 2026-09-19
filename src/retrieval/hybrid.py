"""
Hybrid retrieval fusion.

Combines BM25 (sparse) and FAISS (dense) retrieval results using
reciprocal rank fusion (RRF), producing a single ranked list of
candidates that benefits from both lexical and semantic matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.chunking import Chunk, load_chunks
from src.retrieval.bm25 import SparseIndex, build_sparse_index
from src.retrieval.faiss_index import DEFAULT_MODEL_NAME, DenseIndex, build_dense_index
from src.retrieval.reranker import DEFAULT_CROSS_ENCODER_MODEL, RerankedResult, get_reranker

DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"
DEFAULT_FETCH_K = 20
DEFAULT_RRF_K = 60
DEFAULT_RETRIEVE_K = 20
DEFAULT_RERANK_N = 5


@dataclass
class HybridResult:
    """A single fused retrieval result."""

    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None


def _chunk_key(chunk: Chunk) -> tuple[str, str, int]:
    return (chunk.source, chunk.language, chunk.chunk_index)


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
    k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridResult]:
    """Fuse dense and sparse ranked result lists via reciprocal rank fusion.

    Each chunk's fused score is the sum of `1 / (rrf_k + rank)` over every
    list in which it appears (rank is 1-based); chunks appearing in both
    lists are combined into a single result.
    """
    fused_scores: dict[tuple, float] = {}
    chunk_by_key: dict[tuple, Chunk] = {}
    dense_ranks: dict[tuple, int] = {}
    sparse_ranks: dict[tuple, int] = {}

    for rank, (chunk, _score) in enumerate(dense_results, start=1):
        key = _chunk_key(chunk)
        chunk_by_key[key] = chunk
        dense_ranks[key] = rank
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (chunk, _score) in enumerate(sparse_results, start=1):
        key = _chunk_key(chunk)
        chunk_by_key[key] = chunk
        sparse_ranks[key] = rank
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)[:k]

    return [
        HybridResult(
            chunk=chunk_by_key[key],
            score=fused_scores[key],
            dense_rank=dense_ranks.get(key),
            sparse_rank=sparse_ranks.get(key),
        )
        for key in ranked_keys
    ]


class HybridRetriever:
    """Owns a dense (FAISS) and sparse (BM25) index over the same chunks."""

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL_NAME):
        if not chunks:
            raise ValueError("Cannot build a retriever over an empty chunk list")
        self.chunks = chunks
        self.dense_index: DenseIndex = build_dense_index(chunks, model_name=model_name)
        self.sparse_index: SparseIndex = build_sparse_index(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = DEFAULT_FETCH_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[HybridResult]:
        """Run both retrievers and fuse their top `fetch_k` results into the top `k`."""
        dense_results = self.dense_index.search(query, fetch_k)
        sparse_results = self.sparse_index.search(query, fetch_k)
        return reciprocal_rank_fusion(dense_results, sparse_results, k=k, rrf_k=rrf_k)


_retriever_cache: dict[tuple[str, str], HybridRetriever] = {}


def get_retriever(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
) -> HybridRetriever:
    """Return a cached `HybridRetriever` for the given chunks file and model.

    Building the dense embedding model and index is expensive, so retrievers
    are cached per (chunks_path, model_name) pair for the lifetime of the process.
    """
    cache_key = (str(chunks_path), model_name)
    if cache_key not in _retriever_cache:
        chunks = load_chunks(chunks_path)
        _retriever_cache[cache_key] = HybridRetriever(chunks, model_name=model_name)
    return _retriever_cache[cache_key]


def hybrid_search(
    query: str,
    k: int = 5,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[HybridResult]:
    """Run hybrid (BM25 + FAISS, RRF-fused) search over the persisted chunks.

    Args:
        query: Free-text query, in English or Arabic.
        k: Number of fused results to return.
        chunks_path: Path to the JSON chunks file produced by the ingestion pipeline.
        model_name: Multilingual sentence-transformers model for dense retrieval.

    Returns:
        Up to `k` `HybridResult` objects, ranked by fused RRF score.
    """
    retriever = get_retriever(chunks_path=chunks_path, model_name=model_name)
    return retriever.search(query, k=k)


@dataclass
class RerankedSearchResult:
    """Hybrid retrieval output, before and after cross-encoder reranking."""

    retrieved: list[HybridResult]
    """Top `retrieve_k` fused candidates, in RRF order (pre-rerank)."""

    reranked: list[RerankedResult]
    """Top `rerank_n` candidates after cross-encoder scoring (post-rerank)."""


def hybrid_search_with_rerank(
    query: str,
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    rerank_n: int = DEFAULT_RERANK_N,
    fetch_k: int = DEFAULT_FETCH_K,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    rrf_k: int = DEFAULT_RRF_K,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> RerankedSearchResult:
    """Run hybrid_search for `retrieve_k` candidates, then rerank the top `rerank_n`.

    This is the two-stage pattern: hybrid (BM25 + FAISS, RRF-fused) search
    is cheap and casts a wide net over `retrieve_k` candidates (e.g. 20),
    then a cross-encoder jointly scores each (query, chunk) pair from that
    shortlist and returns the best `rerank_n` (e.g. 5) reranked. The
    cross-encoder is far more accurate than the initial fused scores but
    too slow to run over the whole corpus, hence the two stages.

    Args:
        query: Free-text query, in English or Arabic.
        retrieve_k: Number of fused candidates to retrieve before reranking.
        rerank_n: Number of top candidates to return after reranking.
        fetch_k: Number of candidates each of BM25/FAISS contributes to fusion
            (must be at least `retrieve_k` to fill the shortlist).
        chunks_path: Path to the JSON chunks file produced by the ingestion pipeline.
        model_name: Multilingual sentence-transformers model for dense retrieval.
        rrf_k: Reciprocal rank fusion smoothing constant.
        cross_encoder_model: Cross-encoder model name for reranking.

    Returns:
        A `RerankedSearchResult` with both the pre-rerank (`retrieved`) and
        post-rerank (`reranked`) orderings, so callers can compare them.
    """
    retriever = get_retriever(chunks_path=chunks_path, model_name=model_name)
    retrieved = retriever.search(query, k=retrieve_k, fetch_k=max(fetch_k, retrieve_k), rrf_k=rrf_k)

    reranker = get_reranker(cross_encoder_model)
    reranked = reranker.rerank(query, [result.chunk for result in retrieved], top_n=rerank_n)

    return RerankedSearchResult(retrieved=retrieved, reranked=reranked)
