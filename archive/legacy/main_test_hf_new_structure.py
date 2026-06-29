# src/main_test_hf_new_structure.py

"""
Test script for real Hugging Face datasets.

This script tests the real framework pipeline:

Hugging Face dataset
    ↓
HF Dataset Loader
    ↓
HF Adapter
    ↓
Standard samples
    ↓
Dataset Analyzer
    ↓
Dataset Profile
    ↓
Evaluation Manager
    ↓
Evaluation Plan Generator
    ↓
Evaluator Planner
    ↓
Selected Questions

This script does not run real LLM evaluation yet.
Its purpose is to check whether the framework structure works
with real datasets.
"""


import argparse
import json

from src.data.hf_dataset_registry import list_available_hf_datasets
from src.data.hf_dataset_loader import load_hf_dataset_samples

from src.core.dataset_analyzer import analyze_dataset_to_profile
from src.core.evaluation_manager import EvaluationManager
from src.core.evaluation_plan_generator import EvaluationPlanGenerator
from src.core.evaluator_planner import EvaluatorPlanner


def run_one_dataset(dataset_key: str, sample_size: int) -> dict:
    """
    Run the framework planning pipeline for one Hugging Face dataset.

    This function does not evaluate answers yet.
    It only checks:
    - loading
    - adaptation
    - profiling
    - evaluator selection
    - evaluation plan generation
    - question selection
    """

    # ------------------------------------------------------------
    # Step 1: Load real HF dataset samples.
    # ------------------------------------------------------------
    loaded = load_hf_dataset_samples(
        dataset_key=dataset_key,
        sample_size=sample_size,
    )

    dataset_config = loaded["dataset_config"]
    raw_samples = loaded["raw_samples"]
    adapted_samples = loaded["adapted_samples"]

    # ------------------------------------------------------------
    # Step 2: Generate Dataset Profile from adapted samples.
    # ------------------------------------------------------------
    profile = analyze_dataset_to_profile(
        dataset_name=dataset_key,
        samples=adapted_samples,
        llm_client=None,
        dataset_description=dataset_config.get("description"),
        dataset_link=dataset_config.get("dataset_link"),
        source="huggingface",
        split=dataset_config.get("split", "unknown"),
        location=dataset_config.get("hf_path"),
    )

    # ------------------------------------------------------------
    # Step 3: Select evaluators based on Dataset Profile.
    # ------------------------------------------------------------
    manager = EvaluationManager()
    selected_evaluators = manager.select_evaluators(profile)

    # ------------------------------------------------------------
    # Step 4: Generate a full Evaluation Plan.
    # ------------------------------------------------------------
    plan_generator = EvaluationPlanGenerator()
    evaluation_plan = plan_generator.generate_plan(
        profile=profile,
        selected_evaluators=selected_evaluators,
    )

    # ------------------------------------------------------------
    # Step 5: Select questions using Evaluator Planner.
    # ------------------------------------------------------------
    evaluator_planner = EvaluatorPlanner()

    planner_config = evaluator_planner.build_planner_config(
        collection_strategy="one_by_one_ordered",
        question_selection_strategy="unseen_only",
        progress_strategy="partial_eval",
        max_questions=min(3, sample_size),
        random_seed=42,
    )

    selected_questions = evaluator_planner.select_questions(
        questions=profile["sample_profile"]["sample_questions"],
        answered_question_ids=set(),
        planner_config=planner_config,
    )

    return {
        "dataset_key": dataset_key,
        "dataset_config": dataset_config,
        "raw_preview": raw_samples[:2],
        "adapted_preview": adapted_samples[:2],
        "profile": profile,
        "selected_evaluators": selected_evaluators,
        "evaluation_plan": evaluation_plan,
        "planner_config": planner_config,
        "selected_questions": selected_questions,
    }


def print_result(result: dict) -> None:
    """
    Pretty print the result for one dataset.
    """

    print("\n" + "=" * 80)
    print(f"DATASET: {result['dataset_key']}")
    print("=" * 80)

    print("\n===== RAW HF PREVIEW =====")
    print(json.dumps(result["raw_preview"], indent=2, ensure_ascii=False, default=str))

    print("\n===== ADAPTED PREVIEW =====")
    print(json.dumps(result["adapted_preview"], indent=2, ensure_ascii=False, default=str))

    print("\n===== DATASET PROFILE =====")
    print(json.dumps(result["profile"], indent=2, ensure_ascii=False, default=str))

    print("\n===== SELECTED EVALUATORS =====")
    print(json.dumps(result["selected_evaluators"], indent=2, ensure_ascii=False, default=str))

    print("\n===== EVALUATION PLAN =====")
    print(json.dumps(result["evaluation_plan"], indent=2, ensure_ascii=False, default=str))

    print("\n===== PLANNER CONFIG =====")
    print(json.dumps(result["planner_config"], indent=2, ensure_ascii=False, default=str))

    print("\n===== SELECTED QUESTIONS =====")
    print(json.dumps(result["selected_questions"], indent=2, ensure_ascii=False, default=str))


def main():
    """
    Command line entry point.

    Examples:

    Test one dataset:
        python -m src.main_test_hf_new_structure --dataset hotpotqa --sample-size 3

    Test all registered datasets:
        python -m src.main_test_hf_new_structure --dataset all --sample-size 3
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        help="Dataset key to test. Use 'all' to test all registered datasets.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of samples to load from each dataset.",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        dataset_keys = list_available_hf_datasets()
    else:
        dataset_keys = [args.dataset]

    for dataset_key in dataset_keys:
        try:
            result = run_one_dataset(
                dataset_key=dataset_key,
                sample_size=args.sample_size,
            )
            print_result(result)

        except Exception as error:
            print("\n" + "=" * 80)
            print(f"FAILED DATASET: {dataset_key}")
            print("=" * 80)
            print(str(error))


if __name__ == "__main__":
    main()