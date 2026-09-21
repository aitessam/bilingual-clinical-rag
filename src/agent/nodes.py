"""
LangGraph node functions.

Defines the individual node functions (language detection, hybrid
retrieval + reranking, grounded generation, and citation formatting)
that make up the agentic RAG workflow. Nodes operate on a shared
`AgentState` dict and return partial state updates, per LangGraph
convention.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypedDict

from src.retrieval.hybrid import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANK_N,
    DEFAULT_RETRIEVE_K,
    hybrid_search_with_rerank,
)
from src.retrieval.reranker import RerankedResult

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

_ARABIC_CHAR_RE = re.compile(r"[؀-ۿ]")

_LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a clinical documentation assistant. Answer the user's question "
    "using ONLY the information in the provided context chunks below - do not "
    "use any outside or prior medical knowledge. "
    "Respond in {language_name}, regardless of the language the context is written in. "
    "If the context does not contain enough information to answer the question, "
    "say so explicitly instead of guessing or inventing an answer. "
    "Be concise and factual."
)


class Citation(TypedDict):
    """A single source reference attached to the final answer."""

    source: str
    language: str
    chunk_index: int
    rerank_score: float


class AgentState(TypedDict, total=False):
    """Shared state threaded through the LangGraph nodes."""

    query: str
    language: str
    reranked: list[RerankedResult]
    answer: str
    citations: list[Citation]
    final_response: str


def detect_language(text: str) -> str:
    """Classify `text` as "ar" or "en" by the proportion of Arabic-script characters.

    This is a lightweight heuristic (no external language-ID dependency):
    text is treated as Arabic once more than 15% of its characters fall in
    the Arabic Unicode block, which is robust to short queries containing a
    few digits, punctuation, or Latin medical abbreviations.
    """
    if not text:
        return "en"
    arabic_chars = len(_ARABIC_CHAR_RE.findall(text))
    return "ar" if arabic_chars / len(text) > 0.15 else "en"


def detect_language_node(state: AgentState) -> dict[str, Any]:
    """Node 1: detect whether the incoming query is English or Arabic."""
    return {"language": detect_language(state["query"])}


def make_retrieve_node(
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    rerank_n: int = DEFAULT_RERANK_N,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embedding_model: str = DEFAULT_MODEL_NAME,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
):
    """Build node 2: hybrid_search + cross-encoder rerank, with fixed retrieval settings."""

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        result = hybrid_search_with_rerank(
            state["query"],
            retrieve_k=retrieve_k,
            rerank_n=rerank_n,
            chunks_path=chunks_path,
            model_name=embedding_model,
            cross_encoder_model=cross_encoder_model,
        )
        return {"reranked": result.reranked}

    return retrieve_node


def _build_context(reranked: list[RerankedResult]) -> str:
    blocks = []
    for i, result in enumerate(reranked, start=1):
        chunk = result.chunk
        blocks.append(f"[Chunk {i} | source={chunk.source} | language={chunk.language}]\n{chunk.text}")
    return "\n\n".join(blocks)


def make_generate_node(model_name: str = DEFAULT_GROQ_MODEL, temperature: float = 0.0):
    """Build node 3: answer the query from retrieved context only, via a Groq LLM."""

    def generate_node(state: AgentState) -> dict[str, Any]:
        reranked = state.get("reranked", [])
        language = state.get("language", "en")
        language_name = _LANGUAGE_NAMES.get(language, "English")

        if not reranked:
            no_context_answer = {
                "en": "The provided context does not contain information to answer this question.",
                "ar": "لا يحتوي السياق المتوفر على معلومات كافية للإجابة على هذا السؤال.",
            }
            return {"answer": no_context_answer.get(language, no_context_answer["en"])}

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example) "
                "before running the generate step."
            )

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model_name, temperature=temperature)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(language_name=language_name)
        context = _build_context(reranked)
        human_prompt = f"Context:\n{context}\n\nQuestion: {state['query']}"

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        return {"answer": response.content}

    return generate_node


def format_response_node(state: AgentState) -> dict[str, Any]:
    """Node 4: attach source/chunk-index citations to the generated answer."""
    reranked = state.get("reranked", [])
    answer = state.get("answer", "")

    citations: list[Citation] = [
        {
            "source": result.chunk.source,
            "language": result.chunk.language,
            "chunk_index": result.chunk.chunk_index,
            "rerank_score": result.rerank_score,
        }
        for result in reranked
    ]

    if citations:
        citation_lines = "\n".join(
            f"  [{i}] {c['source']} (chunk {c['chunk_index']}, {c['language']})"
            for i, c in enumerate(citations, start=1)
        )
        final_response = f"{answer}\n\nSources:\n{citation_lines}"
    else:
        final_response = answer

    return {"citations": citations, "final_response": final_response}
