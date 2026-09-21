"""
LangGraph graph definition.

Assembles the node functions from `nodes.py` into a compiled LangGraph
state graph:

    START -> detect_language -> retrieve -> generate -> format_response -> END

`detect_language` identifies the query's language, `retrieve` runs hybrid
(BM25 + FAISS) search followed by cross-encoder reranking, `generate`
answers the query using only the retrieved chunks as context, and
`format_response` attaches source/chunk-index citations to the final
answer.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    DEFAULT_GROQ_MODEL,
    AgentState,
    detect_language_node,
    format_response_node,
    make_generate_node,
    make_retrieve_node,
)
from src.retrieval.hybrid import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANK_N,
    DEFAULT_RETRIEVE_K,
)


def build_graph(
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    rerank_n: int = DEFAULT_RERANK_N,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embedding_model: str = DEFAULT_MODEL_NAME,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    llm_model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
):
    """Build and compile the bilingual clinical RAG agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("detect_language", detect_language_node)
    graph.add_node(
        "retrieve",
        make_retrieve_node(
            retrieve_k=retrieve_k,
            rerank_n=rerank_n,
            chunks_path=chunks_path,
            embedding_model=embedding_model,
            cross_encoder_model=cross_encoder_model,
        ),
    )
    graph.add_node("generate", make_generate_node(model_name=llm_model, temperature=temperature))
    graph.add_node("format_response", format_response_node)

    graph.add_edge(START, "detect_language")
    graph.add_edge("detect_language", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


_graph_cache: dict[tuple, object] = {}


def get_graph(
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    rerank_n: int = DEFAULT_RERANK_N,
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    embedding_model: str = DEFAULT_MODEL_NAME,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    llm_model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
):
    """Return a cached compiled graph for the given configuration."""
    cache_key = (
        retrieve_k,
        rerank_n,
        str(chunks_path),
        embedding_model,
        cross_encoder_model,
        llm_model,
        temperature,
    )
    if cache_key not in _graph_cache:
        _graph_cache[cache_key] = build_graph(
            retrieve_k=retrieve_k,
            rerank_n=rerank_n,
            chunks_path=chunks_path,
            embedding_model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            llm_model=llm_model,
            temperature=temperature,
        )
    return _graph_cache[cache_key]


def ask(query: str, **kwargs) -> AgentState:
    """Run the full agent graph on `query` and return the final state.

    Any keyword arguments are forwarded to `get_graph` to configure
    retrieval/reranking/generation (e.g. `retrieve_k`, `rerank_n`,
    `chunks_path`, `llm_model`).
    """
    graph = get_graph(**kwargs)
    return graph.invoke({"query": query})
