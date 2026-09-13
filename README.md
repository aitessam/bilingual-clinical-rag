# Clinical RAG Bilingual

A bilingual (English/Arabic) agentic RAG assistant over clinical documents, built with LangChain, LangGraph, hybrid BM25/FAISS retrieval, and evaluated with RAGAS.

Part of the [BlockMed Pro](https://www.blockmedpro.com) platform.

## Status

This is a scaffold only. Folders and modules are in place with docstrings describing their intended purpose, but no ingestion, retrieval, agent, or evaluation logic has been implemented yet.

## Project structure

```
src/
  ingestion/   # Loading and chunking of raw clinical documents
  retrieval/   # BM25, FAISS, hybrid fusion and reranking
  agent/       # LangGraph nodes and graph definition
  eval/        # RAGAS evaluation harness
app/
  streamlit_app.py   # Streamlit UI entry point
data/
  raw/         # Unprocessed source documents
  processed/   # Chunked / indexed documents
tests/         # Test suite
```

## Setup

1. Create and activate a virtual environment (Python 3.11+).
2. Install the project in editable mode with dev dependencies:

   ```
   pip install -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and fill in your Groq API key:

   ```
   cp .env.example .env
   ```

## Running the UI (once implemented)

```
streamlit run app/streamlit_app.py
```

## Running tests

```
pytest
```
