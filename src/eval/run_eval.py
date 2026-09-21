#!/usr/bin/env python
"""Run the bilingual clinical RAG agent through a RAGAS evaluation.

1. Loads the hand-written bilingual test set (`eval/testset.json`, ~half
   English / half Arabic, grounded in `data/raw`).
2. Runs the LangGraph agent (`src/agent/graph.py`) on every question.
3. Scores faithfulness, answer relevancy, context precision, and context
   recall with RAGAS.
4. Prints a results table, saves it to `eval/results.csv`, and prints a
   one-paragraph summary of where the system is strongest/weakest.

Usage:
    python -m src.eval.run_eval
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import pandas as pd  # noqa: E402

from src.eval.ragas_harness import (  # noqa: E402
    DEFAULT_RESULTS_PATH,
    DEFAULT_TESTSET_PATH,
    load_testset,
    run_agent_on_testset,
    run_evaluation,
    summarize,
)


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 40)

    testset = load_testset(DEFAULT_TESTSET_PATH)
    print(f"Loaded {len(testset)} evaluation questions from {DEFAULT_TESTSET_PATH}")

    print("Running agent over the test set...")
    records = run_agent_on_testset(testset)

    print("Scoring with RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)...")
    results_df = run_evaluation(records)

    Path(DEFAULT_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DEFAULT_RESULTS_PATH, index=False)

    print(f"\nResults (also saved to {DEFAULT_RESULTS_PATH}):\n")
    print(results_df.to_string(index=False))

    print("\nSummary:\n")
    print(summarize(results_df))


if __name__ == "__main__":
    main()
