# src/main_demo_profiles_only.py

"""
Profile-Only Demo.

This script demonstrates Dataset Profile generation for multiple
Hugging Face datasets.

Purpose:
- Show that the framework is not only designed for HotpotQA.
- Show that different datasets produce different profiles.
- Show that different profiles lead to different evaluator selections.
- Show that weak question labels are post-processed into formal taxonomy.

This script does NOT show the full planning pipeline.
It only shows:
    Dataset
        ↓
    Dataset Analyzer
        ↓
    Metadata Postprocessor
        ↓
    Dataset Profile
        ↓
    Evaluator Manager

For full pipeline demo, use:
    python -m src.main_demo_framework_summary --dataset hotpotqa --sample-size 3 --clear-log

Examples:
    python -m src.main_demo_profiles_only --dataset all --sample-size 3

    python -m src.main_demo_profiles_only --dataset hotpotqa --sample-size 3

    python -m src.main_demo_profiles_only --dataset ai2_arc_challenge --sample-size 3

    python -m src.main_demo_profiles_only --dataset strategyqa --sample-size 3
"""


import argparse

from src.data.hf_dataset_registry import list_available_hf_datasets
from src.data.hf_dataset_loader import load_hf_dataset_samples

from src.core.dataset_analyzer import analyze_dataset_to_profile
from src.core.metadata_postprocessor import postprocess_profile
from src.core.evaluation_manager import EvaluationManager


def build_profile(dataset_key: str, sample_size: int) -> dict:
    """
    Load dataset samples, generate an initial profile, and post-process it.

    Steps:
        1. Load Hugging Face dataset samples.
        2. Adapt samples to a unified format.
        3. Generate initial profile using Dataset Analyzer.
        4. Improve profile using Metadata Postprocessor.

    Why postprocess_profile is important:
        The basic analyzer may produce weak or generic labels.
        The postprocessor fixes dataset-specific cases.

        Examples:
        - HotpotQA -> multi_hop_reasoning
        - StrategyQA -> boolean_reasoning
        - ARC -> multiple_choice
        - RAGBench -> rag_response_evaluation
    """

    loaded = load_hf_dataset_samples(
        dataset_key=dataset_key,
        sample_size=sample_size,
    )

    dataset_config = loaded["dataset_config"]
    adapted_samples = loaded["adapted_samples"]

    # ------------------------------------------------------------
    # Step 1:
    # Generate initial Dataset Profile.
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
    # Step 2:
    # Post-process profile into stronger formal taxonomy.
    # ------------------------------------------------------------
    profile = postprocess_profile(
        profile=profile,
        samples=adapted_samples,
        dataset_key=dataset_key,
    )

    return profile


def print_profile_card(profile: dict) -> None:
    """
    Print one compact profile card.

    This output is designed for supervisor meeting demonstration.

    It shows:
    - dataset identity
    - dataset structure
    - question / answer profile
    - semantic profile
    - evaluation candidates
    - evaluator manager output
    """

    identity = profile.get("dataset_identity", {})
    structure = profile.get("dataset_structure", {})
    qa = profile.get("question_answer_profile", {})
    semantic = profile.get("semantic_profile", {})
    evaluation = profile.get("evaluation_profile", {})

    manager = EvaluationManager()
    manager_result = manager.select_evaluators_with_reasons(profile)

    print("\n" + "=" * 80)
    print(f"DATASET PROFILE: {identity.get('dataset_name')}")
    print("=" * 80)

    print(f"Source:                 {identity.get('source')}")
    print(f"Split:                  {identity.get('split')}")
    print(f"Location:               {identity.get('location')}")
    print(f"Category:               {structure.get('dataset_category')}")
    print(f"Sample count:           {structure.get('sample_count')}")
    print(f"Fields:                 {structure.get('fields')}")
    print(f"Has context:            {structure.get('has_context')}")
    print(f"Has reference answer:   {structure.get('has_reference_answer')}")
    print(f"Has generated response: {structure.get('has_generated_response')}")
    print(f"Has options:            {structure.get('has_options')}")

    print("\nQuestion / Answer Profile:")
    print(f"- question_type:        {qa.get('question_type')}")
    print(f"- answer_type:          {qa.get('answer_type')}")
    print(f"- reasoning_type:       {qa.get('reasoning_type')}")
    print(f"- difficulty_level:     {qa.get('difficulty_level')}")
    print(f"- context_dependency:   {qa.get('context_dependency')}")
    print(f"- expected_format:      {qa.get('expected_answer_format')}")

    print("\nSemantic Profile:")
    print(f"- domain:               {semantic.get('domain')}")
    print(f"- topic:                {semantic.get('topic')}")

    print("\nEvaluation Profile:")
    print(f"- candidate evaluators: {evaluation.get('evaluation_candidates')}")
    print(f"- possible metrics:     {evaluation.get('possible_metrics')}")

    notes = profile.get("notes")
    if notes:
        print("\nNotes:")
        print(f"- {notes}")

    print("\nEvaluator Manager Output:")
    for evaluator in manager_result["selected_evaluators"]:
        print(f"- {evaluator}")
        for reason in manager_result["selection_reasons"].get(evaluator, []):
            print(f"  reason: {reason}")


def main():
    """
    Run profile-only demo.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset key or 'all'.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of samples to load.",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        dataset_keys = list_available_hf_datasets()
    else:
        dataset_keys = [args.dataset]

    print("\nDataset Profile Demo")
    print("=" * 80)

    for dataset_key in dataset_keys:
        try:
            profile = build_profile(
                dataset_key=dataset_key,
                sample_size=args.sample_size,
            )

            print_profile_card(profile)

        except Exception as error:
            print("\n" + "=" * 80)
            print(f"FAILED DATASET: {dataset_key}")
            print("=" * 80)
            print(error)


if __name__ == "__main__":
    main()