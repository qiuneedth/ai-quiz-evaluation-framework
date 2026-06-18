# src/main_test_manager.py

"""
Test the Evaluation Manager.

This file does not call OpenAI.
It only tests whether profiles are mapped to correct evaluators.
"""

import json

from src.core.evaluation_manager import build_evaluation_plan


def main():
    profiles = [
        {
            "dataset_name": "HotpotQA",
            "question_type": "short_answer",
            "has_context": True,
            "answer_type": "free_text",
            "reasoning_required": True
        },
        {
            "dataset_name": "StrategyQA",
            "question_type": "true_false",
            "has_context": False,
            "answer_type": "boolean",
            "reasoning_required": True
        },
        {
            "dataset_name": "MMLU",
            "question_type": "mcq",
            "has_context": False,
            "has_options": True,
            "answer_type": "multiple_choice",
            "reasoning_required": True
        },
        {
            "dataset_name": "Natural Questions",
            "question_type": "short_answer",
            "has_context": False,
            "answer_type": "free_text",
            "reasoning_required": False
        }
    ]

    for profile in profiles:
        print("=" * 80)
        print("Input Profile:")
        print(json.dumps(profile, indent=2, ensure_ascii=False))

        plan = build_evaluation_plan(profile)

        print("\nEvaluation Plan:")
        print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()