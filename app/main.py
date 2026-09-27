"""
Streamlit UI for the bilingual clinical RAG assistant.

A simple chat interface: type a question in English or Arabic, see the
detected language and grounded answer, expand to see the source
citations, and check the sidebar for the latest RAGAS evaluation scores.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

RESULTS_PATH = Path("eval/results.csv")
METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LANGUAGE_LABELS = {"en": "English", "ar": "Arabic"}


@st.cache_resource(show_spinner=False)
def _get_compiled_graph():
    from src.agent.graph import get_graph

    return get_graph()


def _language_label(language: str | None) -> str:
    if not language:
        return "unknown"
    return LANGUAGE_LABELS.get(language, language)


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for i, citation in enumerate(citations, start=1):
            st.markdown(
                f"**[{i}]** `{citation['source']}` &nbsp;·&nbsp; chunk {citation['chunk_index']} "
                f"&nbsp;·&nbsp; {_language_label(citation['language'])} "
                f"&nbsp;·&nbsp; rerank score {citation['rerank_score']:.3f}"
            )


def _render_sidebar() -> None:
    st.sidebar.header("Evaluation (RAGAS)")

    if not RESULTS_PATH.exists():
        st.sidebar.info(
            "No eval/results.csv yet.\n\nRun:\n\n`python -m src.eval.run_eval`\n\nto generate scores."
        )
        return

    results_df = pd.read_csv(RESULTS_PATH)
    available_metrics = [col for col in METRIC_COLUMNS if col in results_df.columns]
    means = results_df[available_metrics].mean(numeric_only=True) if available_metrics else pd.Series(dtype=float)

    st.sidebar.caption(f"{len(results_df)} evaluation questions")
    for metric in available_metrics:
        value = means.get(metric)
        label = metric.replace("_", " ").title()
        st.sidebar.metric(label, f"{value:.2f}" if pd.notna(value) else "n/a")

    with st.sidebar.expander("Per-question scores"):
        st.dataframe(results_df, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Bilingual Clinical RAG Assistant", layout="wide")

    st.title("Bilingual Clinical RAG Assistant")
    st.caption(
        "Ask a clinical question in English or Arabic. Answers are grounded only in the "
        "ingested notes under data/raw/, with citations back to the source chunks."
    )

    _render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("language"):
                st.caption(f"Detected language: {_language_label(message['language'])}")
            st.write(message["content"])
            _render_citations(message.get("citations", []))

    query = st.chat_input("Ask a question in English or Arabic...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query, "language": None, "citations": []})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                graph = _get_compiled_graph()
                state = graph.invoke({"query": query})
                answer = state.get("answer", "(no answer produced)")
                language = state.get("language")
                citations = state.get("citations", [])
            except RuntimeError as exc:
                answer = f"Could not generate an answer: {exc}"
                language = None
                citations = []

        if language:
            st.caption(f"Detected language: {_language_label(language)}")
        st.write(answer)
        _render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "language": language, "citations": citations}
    )


if __name__ == "__main__":
    main()
