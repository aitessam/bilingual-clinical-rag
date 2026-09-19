"""
Document loaders for raw clinical source material.

Reads plain-text clinical documents from `data/raw/<lang>/` (English and
Arabic subfolders), and converts them into a common in-memory document
representation for downstream chunking and indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_LANGUAGES = ("en", "ar")


@dataclass(frozen=True)
class Document:
    """A single raw document loaded from disk."""

    source: str
    language: str
    text: str


def load_documents(raw_dir: Path | str, languages: tuple[str, ...] = SUPPORTED_LANGUAGES) -> list[Document]:
    """Load all `.txt` files from `raw_dir/<language>/` for each language.

    Args:
        raw_dir: Path to the root raw-data directory (e.g. `data/raw`).
        languages: Language subfolder names to load (default: en, ar).

    Returns:
        List of `Document` objects, one per `.txt` file, sorted by
        language then filename for deterministic ordering.
    """
    raw_dir = Path(raw_dir)
    documents: list[Document] = []

    for language in languages:
        lang_dir = raw_dir / language
        if not lang_dir.is_dir():
            continue
        for file_path in sorted(lang_dir.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(source=file_path.name, language=language, text=text)
            )

    return documents
