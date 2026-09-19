"""
Chunking strategies for clinical documents.

Splits normalised documents into overlapping, retrieval-sized chunks
(roughly 200-300 tokens with a 50-token overlap), attaching source,
language, and chunk-index metadata to each chunk. "Tokens" here are
whitespace-delimited words, which keeps the implementation dependency-free
and works reasonably for both English and Arabic text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ingestion.loaders import Document, load_documents

DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class Chunk:
    """A single retrieval-sized passage with its provenance metadata."""

    text: str
    source: str
    language: str
    chunk_index: int


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split `text` into overlapping word-based chunks.

    Args:
        text: The document text to split.
        chunk_size: Target number of tokens (words) per chunk.
        chunk_overlap: Number of tokens shared between consecutive chunks.

    Returns:
        List of chunk strings, in order.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = text.split()
    if not tokens:
        return []

    stride = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a single `Document`, attaching source/language/index metadata."""
    texts = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        Chunk(
            text=chunk_text_value,
            source=document.source,
            language=document.language,
            chunk_index=index,
        )
        for index, chunk_text_value in enumerate(texts)
    ]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a list of documents into a flat list of `Chunk` objects."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return chunks


def save_chunks(chunks: list[Chunk], output_path: Path | str) -> None:
    """Persist chunks to disk as a JSON list of objects, for reuse by indexing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(chunk) for chunk in chunks]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(input_path: Path | str) -> list[Chunk]:
    """Load previously persisted chunks from a JSON file."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in payload]


def ingest(
    raw_dir: Path | str,
    output_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Load raw documents, chunk them, persist to `output_path`, and return them."""
    documents = load_documents(raw_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_chunks(chunks, output_path)
    return chunks
