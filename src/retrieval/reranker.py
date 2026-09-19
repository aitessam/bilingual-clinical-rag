"""
Reranking of retrieved candidates.

Applies a cross-encoder to reorder hybrid-retrieval candidates by relevance
before they are passed to the agent. A cross-encoder scores each
(query, chunk) pair jointly, which is more accurate than the independent
bi-encoder/BM25 scores used for initial retrieval, at the cost of being
too slow to run over the whole corpus (hence it only reranks the small
shortlist that hybrid retrieval already narrowed down).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.chunking import Chunk

# mmarco-mMiniLMv2-L12-H384-v1 is a cross-encoder fine-tuned on mMARCO, the
# multilingual version of MS MARCO covering ~14 languages including Arabic,
# so it handles English and Arabic passages reasonably well in one model
# for this demo. It is a general-purpose passage-relevance model, not a
# clinical or Arabic-specific one: a production system over Arabic clinical
# text would likely do better with a cross-encoder fine-tuned on in-domain
# Arabic (or bilingual) medical text, since general multilingual rerankers
# tend to be weaker on Arabic than on English and can miss domain-specific
# terminology.
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class RerankedResult:
    """A single candidate after cross-encoder reranking."""

    chunk: Chunk
    rerank_score: float
    previous_rank: int


class CrossEncoderReranker:
    """Wraps a sentence-transformers `CrossEncoder` for query/chunk scoring."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[RerankedResult]:
        """Score each chunk against `query` and return the top `top_n`, best first.

        `previous_rank` (1-based) records each chunk's position in the
        `chunks` list as passed in, so callers can compare the reranked
        order against whatever ordering the chunks arrived in (e.g. the
        hybrid RRF ranking).
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [
            RerankedResult(chunk=chunks[i], rerank_score=float(scores[i]), previous_rank=i + 1)
            for i in order[:top_n]
        ]


_reranker_cache: dict[str, CrossEncoderReranker] = {}


def get_reranker(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> CrossEncoderReranker:
    """Return a cached `CrossEncoderReranker` for `model_name`.

    Loading the cross-encoder is expensive, so instances are cached per
    model name for the lifetime of the process.
    """
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoderReranker(model_name)
    return _reranker_cache[model_name]
