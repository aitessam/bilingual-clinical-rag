"""Tests for the ingestion pipeline: loading, chunking, and persistence."""

import json
import math

from src.ingestion.chunking import (
    chunk_documents,
    chunk_text,
    load_chunks,
    save_chunks,
)
from src.ingestion.loaders import Document, load_documents


def _make_raw_tree(tmp_path, en_word_counts, ar_word_counts):
    raw_dir = tmp_path / "raw"
    (raw_dir / "en").mkdir(parents=True)
    (raw_dir / "ar").mkdir(parents=True)

    for index, word_count in enumerate(en_word_counts, start=1):
        text = " ".join(f"word{i}" for i in range(word_count))
        (raw_dir / "en" / f"note_{index:03d}.txt").write_text(text, encoding="utf-8")

    for index, word_count in enumerate(ar_word_counts, start=1):
        text = " ".join(f"كلمة{i}" for i in range(word_count))
        (raw_dir / "ar" / f"note_{index:03d}.txt").write_text(text, encoding="utf-8")

    return raw_dir


def _expected_chunk_count(word_count, chunk_size, chunk_overlap):
    if word_count == 0:
        return 0
    stride = chunk_size - chunk_overlap
    return max(1, math.ceil(max(0, word_count - chunk_size) / stride) + 1)


def test_load_documents_reads_both_languages(tmp_path):
    raw_dir = _make_raw_tree(tmp_path, en_word_counts=[10, 20], ar_word_counts=[15])

    documents = load_documents(raw_dir)

    assert len(documents) == 3
    assert {d.language for d in documents} == {"en", "ar"}
    assert all(isinstance(d, Document) for d in documents)


def test_chunk_text_produces_expected_chunk_count():
    chunk_size, chunk_overlap = 250, 50
    word_count = 620
    text = " ".join(f"word{i}" for i in range(word_count))

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) == _expected_chunk_count(word_count, chunk_size, chunk_overlap)
    # Overlap check: last `chunk_overlap` tokens of chunk N are the first
    # `chunk_overlap` tokens of chunk N+1.
    for first, second in zip(chunks, chunks[1:]):
        assert first.split()[-chunk_overlap:] == second.split()[:chunk_overlap]


def test_chunk_documents_attaches_metadata_and_expected_count(tmp_path):
    chunk_size, chunk_overlap = 250, 50
    en_word_counts = [600, 100]
    ar_word_counts = [400]
    raw_dir = _make_raw_tree(tmp_path, en_word_counts, ar_word_counts)

    documents = load_documents(raw_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    expected_total = sum(
        _expected_chunk_count(wc, chunk_size, chunk_overlap)
        for wc in en_word_counts + ar_word_counts
    )
    assert len(chunks) == expected_total

    for chunk in chunks:
        assert chunk.source
        assert chunk.language in ("en", "ar")
        assert isinstance(chunk.chunk_index, int)

    # Chunk indices restart at 0 for each source document.
    by_source = {}
    for chunk in chunks:
        by_source.setdefault((chunk.source, chunk.language), []).append(chunk.chunk_index)
    for indices in by_source.values():
        assert indices == list(range(len(indices)))


def test_save_and_load_chunks_round_trip(tmp_path):
    raw_dir = _make_raw_tree(tmp_path, en_word_counts=[300], ar_word_counts=[])
    documents = load_documents(raw_dir)
    chunks = chunk_documents(documents)

    output_path = tmp_path / "processed" / "chunks.json"
    save_chunks(chunks, output_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == len(chunks)
    assert payload[0]["source"] == chunks[0].source

    reloaded = load_chunks(output_path)
    assert reloaded == chunks
