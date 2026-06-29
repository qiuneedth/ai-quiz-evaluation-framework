# src/main_test_new_structure.py

"""
Test script for the new framework structure.

This script tests the core pipeline:

Dataset samples
    ↓
Dataset Analyzer
    ↓
Dataset Profile
    ↓
Evaluation Manager
    ↓
Selected Evaluators
    ↓
Evaluation Plan Generator
    ↓
Evaluation Plan
    ↓
Evaluator Planner
    ↓
Selected Questions
"""


import json

from src.core.dataset_analyzer import analyze_dataset_to_profile
from src.core.evaluation_manager import EvaluationManager
from src.core.evaluation_plan_generator import EvaluationPlanGenerator
from src.core.evaluator_planner import EvaluatorPlanner


def main():
    """
    Run a small test with fake dataset samples.
    """

    samples = [
        {
            "question_id": "q1",
            "question": "Which city is the Eiffel Tower located in?",
            "context": "The Eiffel Tower is located in Paris, France.",
            "reference_answer": "Paris",
        },
        {
            "question_id": "q2",
            "question": "What country is Paris in?",
            "context": "Paris is the capital city of France.",
            "reference_answer": "France",
        },
    ]

    # Step 1: Generate Dataset Profile.
    profile = analyze_dataset_to_profile(
        dataset_name="demo_context_qa",
        samples=samples,
        llm_client=None,
        dataset_description="A small context-grounded QA dataset.",
        dataset_link=None,
        source="local_demo",
        split="test",
        location="data/raw/demo.jsonl",
    )

    print("\n===== DATASET PROFILE =====")
    print(json.dumps(profile, indent=2, ensure_ascii=False))

    # Step 2: Select evaluators.
    manager = EvaluationManager()
    selected_evaluators = manager.select_evaluators(profile)

    print("\n===== SELECTED EVALUATORS =====")
    print(json.dumps(selected_evaluators, indent=2, ensure_ascii=False))

    # Step 3: Generate Evaluation Plan.
    plan_generator = EvaluationPlanGenerator()
    evaluation_plan = plan_generator.generate_plan(
        profile=profile,
        selected_evaluators=selected_evaluators,
    )

    print("\n===== EVALUATION PLAN =====")
    print(json.dumps(evaluation_plan, indent=2, ensure_ascii=False))

    # Step 4: Select questions using Evaluator Planner.
    planner = EvaluatorPlanner()

    planner_config = planner.build_planner_config(
        collection_strategy="one_by_one_ordered",
        question_selection_strategy="unseen_only",
        progress_strategy="partial_eval",
        max_questions=1,
    )

    selected_questions = planner.select_questions(
        questions=profile["sample_profile"]["sample_questions"],
        answered_question_ids=set(),
        planner_config=planner_config,
    )

    print("\n===== PLANNER CONFIG =====")
    print(json.dumps(planner_config, indent=2, ensure_ascii=False))

    print("\n===== SELECTED QUESTIONS =====")
    print(json.dumps(selected_questions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()