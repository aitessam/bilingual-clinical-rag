"""
RAGAS evaluation harness.

Runs RAGAS metrics (faithfulness, answer relevancy, context precision,
context recall) against the bilingual clinical RAG agent's outputs on a
small hand-written evaluation dataset (`eval/testset.json`), and reports
the results.

The Groq chat model already used by the agent (`src.agent.nodes`) doubles
as the RAGAS "judge" LLM, and the multilingual sentence-transformers model
already used for dense retrieval doubles as the RAGAS embedding model -
this keeps the evaluation stack consistent with the agent it is grading,
without introducing a second LLM provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.graph import get_graph
from src.agent.nodes import DEFAULT_GROQ_MODEL
from src.retrieval.faiss_index import DEFAULT_MODEL_NAME

DEFAULT_TESTSET_PATH = "eval/testset.json"
DEFAULT_RESULTS_PATH = "eval/results.csv"

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_testset(path: Path | str = DEFAULT_TESTSET_PATH) -> list[dict[str, Any]]:
    """Load the hand-written bilingual QA evaluation set."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_agent_on_testset(
    testset: list[dict[str, Any]],
    **agent_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run the LangGraph agent on every testset question and collect its outputs.

    Returns one record per question with the fields RAGAS needs
    (`user_input`, `retrieved_contexts`, `response`, `reference`), plus the
    original `id`/`language`/`source` metadata for reporting.
    """
    graph = get_graph(**agent_kwargs)
    records = []
    for item in testset:
        state = graph.invoke({"query": item["question"]})
        reranked = state.get("reranked", [])
        records.append(
            {
                "id": item["id"],
                "language": item["language"],
                "source": item["source"],
                "detected_language": state.get("language"),
                "user_input": item["question"],
                "retrieved_contexts": [result.chunk.text for result in reranked],
                "response": state.get("answer", ""),
                "reference": item["reference"],
            }
        )
    return records


def build_ragas_dataset(records: list[dict[str, Any]]):
    """Build a RAGAS `EvaluationDataset` from agent-output records."""
    from ragas import EvaluationDataset

    columns = ["user_input", "retrieved_contexts", "response", "reference"]
    return EvaluationDataset.from_list([{col: record[col] for col in columns} for record in records])


def get_ragas_llm_and_embeddings(
    llm_model: str = DEFAULT_GROQ_MODEL,
    embedding_model: str = DEFAULT_MODEL_NAME,
):
    """Build the RAGAS-wrapped judge LLM and embedding model.

    Reuses the same Groq chat model as the agent's `generate` node, and the
    same multilingual sentence-transformers model as dense retrieval, so
    the evaluator isn't introducing an unrelated third model.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatGroq(model=llm_model, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_model))
    return llm, embeddings


def get_metrics():
    """Return the four RAGAS metrics this harness scores.

    `answer_relevancy` defaults to `strictness=3` (three sampled candidate
    questions per answer, via the LLM's `n` parameter), but Groq's API
    rejects `n > 1`, so it's pinned to 1 here.
    """
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    answer_relevancy.strictness = 1
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def run_evaluation(
    records: list[dict[str, Any]],
    llm_model: str = DEFAULT_GROQ_MODEL,
    embedding_model: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """Score `records` with RAGAS and return a per-question results DataFrame.

    The returned DataFrame has one row per question, columns for the four
    metrics, plus the `id`/`language`/`source` metadata carried over from
    `records` for readability.
    """
    from ragas import evaluate
    from ragas.run_config import RunConfig

    dataset = build_ragas_dataset(records)
    llm, embeddings = get_ragas_llm_and_embeddings(llm_model=llm_model, embedding_model=embedding_model)

    # Groq's free-tier rate limits make the default high-concurrency RunConfig
    # (max_workers=16) time out most jobs; a small worker pool with a longer
    # per-call timeout runs reliably instead.
    run_config = RunConfig(max_workers=2, timeout=180)

    result = evaluate(dataset, metrics=get_metrics(), llm=llm, embeddings=embeddings, run_config=run_config)
    scores_df = result.to_pandas()

    meta_df = pd.DataFrame(
        [{"id": r["id"], "language": r["language"], "source": r["source"]} for r in records]
    )
    return pd.concat([meta_df, scores_df[METRIC_COLUMNS]], axis=1)


def summarize(results_df: pd.DataFrame) -> str:
    """Produce a one-paragraph summary of strengths/weaknesses from `results_df`.

    Purely computed from the scored data (overall and per-language metric
    averages) - not a canned template - so it reflects whatever the actual
    run produced.
    """
    overall_means = results_df[METRIC_COLUMNS].mean().sort_values(ascending=False)
    strongest_metric, strongest_score = overall_means.index[0], overall_means.iloc[0]
    weakest_metric, weakest_score = overall_means.index[-1], overall_means.iloc[-1]

    by_language = results_df.groupby("language")[METRIC_COLUMNS].mean()
    language_summary = ", ".join(
        f"{lang} overall {by_language.loc[lang].mean():.2f}" for lang in sorted(by_language.index)
    )

    return (
        f"Across {len(results_df)} questions, the system scored highest on "
        f"{strongest_metric.replace('_', ' ')} ({strongest_score:.2f}) and lowest on "
        f"{weakest_metric.replace('_', ' ')} ({weakest_score:.2f}), out of 1.0 for each RAGAS metric. "
        f"By language, {language_summary}. "
        f"{'A gap between languages suggests the multilingual embedding/reranking or generation stage handles one language less reliably than the other and is worth targeted review. ' if abs(by_language.mean(axis=1).diff().iloc[-1]) > 0.1 else 'Performance is broadly consistent across English and Arabic. '}"
        f"Low {weakest_metric.replace('_', ' ')} in particular points to where the retrieval-to-generation pipeline "
        "most needs attention next (e.g. tightening the retrieved context set if precision/recall are weak, "
        "or tightening the generation prompt's grounding instructions if faithfulness or relevancy are weak)."
    )
