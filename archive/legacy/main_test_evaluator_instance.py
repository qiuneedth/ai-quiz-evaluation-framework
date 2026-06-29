# src/main_test_evaluator_instance.py

"""
Test script for Evaluator Instance.

This script tests:

Dataset Profile
    ↓
Evaluation Manager
    ↓
Evaluation Plan Generator
    ↓
Evaluator Planner
    ↓
Evaluator Instance
    ↓
User Activity Logger

This test does not require real LLM evaluation yet.
"""


import json
import uuid

from src.core.dataset_analyzer import analyze_dataset_to_profile
from src.core.evaluation_manager import EvaluationManager
from src.core.evaluation_plan_generator import EvaluationPlanGenerator
from src.core.evaluator_planner import EvaluatorPlanner
from src.core.evaluator_instance import EvaluatorInstance
from src.storage.user_activity_logger import UserActivityLogger


def main():
    """
    Run a test evaluation instance.
    """

    # ------------------------------------------------------------
    # Step 1: Prepare demo dataset samples.
    # ------------------------------------------------------------
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
        {
            "question_id": "q3",
            "question": "Which landmark is located in Paris?",
            "context": "The Eiffel Tower is located in Paris, France.",
            "reference_answer": "Eiffel Tower",
        },
    ]

    # ------------------------------------------------------------
    # Step 2: Generate Dataset Profile.
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Step 3: Select evaluators.
    # ------------------------------------------------------------
    manager = EvaluationManager()
    selected_evaluators = manager.select_evaluators(profile)

    # ------------------------------------------------------------
    # Step 4: Generate Evaluation Plan.
    # ------------------------------------------------------------
    plan_generator = EvaluationPlanGenerator()
    evaluation_plan = plan_generator.generate_plan(
        profile=profile,
        selected_evaluators=selected_evaluators,
    )

    # ------------------------------------------------------------
    # Step 5: Build Evaluator Planner config.
    # ------------------------------------------------------------
    evaluator_planner = EvaluatorPlanner()

    planner_config = evaluator_planner.build_planner_config(
        collection_strategy="one_by_one_ordered",
        question_selection_strategy="unseen_only",
        progress_strategy="partial_eval",
        max_questions=2,
        repeat_probability=0.2,
        random_seed=42,
    )

    # ------------------------------------------------------------
    # Step 6: Create User Activity Logger.
    # ------------------------------------------------------------
    activity_logger = UserActivityLogger(
        log_path="data/results/user_activity.jsonl",
    )

    # ------------------------------------------------------------
    # Step 7: Create Evaluator Instance.
    # ------------------------------------------------------------
    instance = EvaluatorInstance(
        instance_id=str(uuid.uuid4()),
        user_id="demo_user",
        dataset_profile=profile,
        evaluation_plan=evaluation_plan,
        planner_config=planner_config,
        evaluator_planner=evaluator_planner,
        evaluator_executor=None,
        activity_logger=activity_logger,
    )

    # ------------------------------------------------------------
    # Step 8: Start the instance and get the first batch.
    # ------------------------------------------------------------
    first_batch = instance.start()

    print("\n===== FIRST BATCH =====")
    print(json.dumps(first_batch, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------
    # Step 9: Simulate user answer.
    # ------------------------------------------------------------
    if first_batch:
        question = first_batch[0]

        instance.submit_answers(
            answers=[
                {
                    "question_id": question["question_id"],
                    "user_answer": "Paris",
                }
            ]
        )

    # ------------------------------------------------------------
    # Step 10: Continue if the progress plan allows.
    # ------------------------------------------------------------
    if instance.should_continue():
        next_batch = instance.next_batch()

        print("\n===== NEXT BATCH =====")
        print(json.dumps(next_batch, indent=2, ensure_ascii=False))

        if next_batch:
            question = next_batch[0]

            instance.submit_answers(
                answers=[
                    {
                        "question_id": question["question_id"],
                        "user_answer": "France",
                    }
                ]
            )

    # ------------------------------------------------------------
    # Step 11: Finish the instance.
    # ------------------------------------------------------------
    summary = instance.finish()

    print("\n===== INSTANCE SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nUser activity log saved to:")
    print("data/results/user_activity.jsonl")


if __name__ == "__main__":
    main()