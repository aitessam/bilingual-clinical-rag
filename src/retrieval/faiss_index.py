"""
Dense vector retrieval using FAISS.

Builds and queries a FAISS index over sentence-transformer embeddings of
the chunked clinical corpus, using a multilingual model so English and
Arabic chunks share the same embedding space (supporting cross-lingual
semantic search).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ingestion.chunking import Chunk

DEFAULT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


@dataclass
class DenseIndex:
    """A FAISS index over chunk embeddings, paired with its embedding model."""

    chunks: list[Chunk]
    model_name: str
    index: object
    model: object

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return up to `k` chunks most similar to `query`, with cosine-style scores."""
        query_vector = np.asarray(
            self.model.encode([query], normalize_embeddings=True), dtype="float32"
        )
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def build_dense_index(chunks: list[Chunk], model_name: str = DEFAULT_MODEL_NAME) -> DenseIndex:
    """Embed all chunks with a multilingual sentence-transformer and index them in FAISS.

    Embeddings are L2-normalised and indexed with inner product, which is
    equivalent to cosine similarity.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype="float32",
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return DenseIndex(chunks=chunks, model_name=model_name, index=index, model=model)
