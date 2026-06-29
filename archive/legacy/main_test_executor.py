# src/main_test_executor.py

"""
Test full evaluator execution.

This demonstrates:

Dataset Profile
→ Evaluation Manager
→ Selected Evaluators
→ Evaluator Executor
→ Evaluation Report
"""

import json

from src.core.evaluation_manager import build_evaluation_plan
from src.core.evaluator_executor import run_evaluation_plan
from src.llm.client import LLMClient


def main():
    llm = LLMClient()

    # Example profile from HotpotQA-like dataset
    profile = {
        "dataset_name": "HotpotQA",
        "question_type": "short_answer",
        "has_context": True,
        "answer_type": "free_text",
        "reasoning_required": True,
    }

    # Example unified quiz sample
    sample = {
        "dataset": "HotpotQA",
        "question_type": "short_answer",
        "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "reference_answer": "Delhi",
        "candidate_answer": "The Oberoi family is part of a hotel company that has its head office in Delhi.",
        "context": "The Oberoi Group is a hotel company with its head office in Delhi.",
        "options": None,
        "metadata": {}
    }

    print("=" * 100)
    print("Dataset Profile:")
    print(json.dumps(profile, indent=2, ensure_ascii=False))

    # Step 1: build evaluation plan
    evaluation_plan = build_evaluation_plan(profile)

    print("\nEvaluation Plan:")
    print(json.dumps(evaluation_plan, indent=2, ensure_ascii=False))

    # Step 2: run selected evaluators
    report = run_evaluation_plan(
        sample=sample,
        evaluation_plan=evaluation_plan,
        llm_client=llm,
    )

    print("\nEvaluation Report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()