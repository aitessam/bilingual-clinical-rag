"""
Document loaders for raw clinical source material.

This module will be responsible for reading clinical documents (e.g. PDF,
DOCX, plain text) from `data/raw/`, detecting language (English/Arabic),
and converting them into a common in-memory document representation for
downstream chunking and indexing. No loading logic is implemented yet.
"""
