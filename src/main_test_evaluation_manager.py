# src/main_test_evaluation_manager.py

"""
Test Evaluation Manager with example dataset profiles.

This demonstrates:

Dataset Profile
→ Evaluation Manager
→ Selected Evaluators
→ Evaluation Report Structure
"""

import json

from src.core.evaluation_manager import (
    build_evaluation_plan,
    build_empty_report,
)


def main():
    profiles = [
        {
            "dataset_name": "HotpotQA",
            "question_type": "short_answer",
            "has_context": True,
            "answer_type": "free_text",
            "reasoning_required": True,
        },
        {
            "dataset_name": "StrategyQA",
            "question_type": "true_false",
            "has_context": False,
            "answer_type": "boolean",
            "reasoning_required": True,
        },
        {
            "dataset_name": "MMLU",
            "question_type": "mcq",
            "has_context": False,
            "has_options": True,
            "answer_type": "multiple_choice",
            "reasoning_required": True,
        },
        {
            "dataset_name": "Natural Questions",
            "question_type": "short_answer",
            "has_context": False,
            "answer_type": "free_text",
            "reasoning_required": False,
        },
    ]

    for profile in profiles:
        print("=" * 100)

        print("Dataset Profile:")
        print(json.dumps(profile, indent=2, ensure_ascii=False))

        plan = build_evaluation_plan(profile)

        print("\nEvaluation Plan:")
        print(json.dumps(plan, indent=2, ensure_ascii=False))

        report = build_empty_report(profile, plan)

        print("\nReport Structure:")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()