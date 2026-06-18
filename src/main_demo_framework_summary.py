# src/main_demo_framework_summary.py

"""
Framework Summary Demo.

This script is designed for supervisor meeting demonstration.

It prints a compact summary of the framework pipeline instead of very long
raw JSON outputs.

Main pipeline:

User Request
    ↓
Hugging Face Dataset Loader
    ↓
Dataset Analyzer
    ↓
Metadata Postprocessor
    ↓
Dataset Profile
    ↓
Profile Selector
    ↓
Selected Profile
    ↓
Evaluator Manager
    ↓
Evaluation Plan Generator
    ↓
Evaluation Plan
    ↓
Evaluator Planner
    ↓
Question Delivery Plan
    ↓
User Activity Log

Important:
This demo focuses on the planning layer of the framework.

It does NOT yet perform full runtime evaluation, which means:
- no AI/model answer is generated yet
- no evaluator score is produced yet
- answered_questions is still empty

The next stage is:
Selected Questions
    ↓
AI / Model Candidate Answers
    ↓
Evaluator Executor
    ↓
Scores
    ↓
Updated User Activity Log
"""


import argparse
from pathlib import Path
import json

from src.data.hf_dataset_loader import load_hf_dataset_samples

from src.core.dataset_analyzer import analyze_dataset_to_profile
from src.core.metadata_postprocessor import postprocess_profile
from src.core.profile_selector import ProfileSelector
from src.core.evaluation_manager import EvaluationManager
from src.core.evaluation_plan_generator import EvaluationPlanGenerator
from src.core.evaluator_planner import EvaluatorPlanner
from src.core.planner_rulebook import explain_planner_config

from src.storage.user_activity_logger import UserActivityLogger


LOG_PATH = "data/results/user_activity_demo.jsonl"


def build_profile_for_dataset(dataset_key: str, sample_size: int) -> dict:
    """
    Load a Hugging Face dataset and generate an improved Dataset Profile.

    Steps:
        1. Load raw samples from Hugging Face.
        2. Convert raw samples into a unified adapted format.
        3. Generate an initial Dataset Profile using Dataset Analyzer.
        4. Improve the profile using Metadata Postprocessor.

    Why postprocess_profile is needed:
        The basic analyzer may produce weak labels or incomplete metadata.
        For example:
        - StrategyQA may look like multiple-choice because it has yes/no options.
        - HotpotQA may look like short-answer QA, but it is actually multi-hop QA.
        - RAGBench should be treated as RAG response evaluation.

        postprocess_profile fixes these cases and converts them into formal
        evaluation-oriented taxonomy.
    """

    loaded = load_hf_dataset_samples(
        dataset_key=dataset_key,
        sample_size=sample_size,
    )

    dataset_config = loaded["dataset_config"]
    adapted_samples = loaded["adapted_samples"]

    # ------------------------------------------------------------
    # Step 1:
    # Generate initial profile.
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
    # Improve profile with dataset-specific and taxonomy rules.
    # This avoids weak labels such as "short_question".
    # ------------------------------------------------------------
    profile = postprocess_profile(
        profile=profile,
        samples=adapted_samples,
        dataset_key=dataset_key,
    )

    return profile


def print_profile_summary(profile: dict) -> None:
    """
    Print a compact Dataset Profile summary.

    This section corresponds to:
        Profile

    in the supervisor's diagram.
    """

    identity = profile.get("dataset_identity", {})
    structure = profile.get("dataset_structure", {})
    qa = profile.get("question_answer_profile", {})
    semantic = profile.get("semantic_profile", {})
    evaluation = profile.get("evaluation_profile", {})

    print("\n[1] DATASET PROFILE SUMMARY")
    print("-" * 60)
    print(f"Dataset name:        {identity.get('dataset_name')}")
    print(f"Source:              {identity.get('source')}")
    print(f"Split:               {identity.get('split')}")
    print(f"Dataset category:    {structure.get('dataset_category')}")
    print(f"Sample count:        {structure.get('sample_count')}")
    print(f"Has context:         {structure.get('has_context')}")
    print(f"Has reference:       {structure.get('has_reference_answer')}")
    print(f"Has generated resp.: {structure.get('has_generated_response')}")
    print(f"Has options:         {structure.get('has_options')}")
    print(f"Question type:       {qa.get('question_type')}")
    print(f"Answer type:         {qa.get('answer_type')}")
    print(f"Reasoning type:      {qa.get('reasoning_type')}")
    print(f"Difficulty level:    {qa.get('difficulty_level')}")
    print(f"Context dependency:  {qa.get('context_dependency')}")
    print(f"Expected format:     {qa.get('expected_answer_format')}")
    print(f"Domain:              {semantic.get('domain')}")
    print(f"Topic:               {semantic.get('topic')}")
    print(f"Eval candidates:     {evaluation.get('evaluation_candidates')}")
    print(f"Possible metrics:    {evaluation.get('possible_metrics')}")

    notes = profile.get("notes")
    if notes:
        print(f"Notes:               {notes}")


def print_profile_selection_summary(selection_result: dict) -> None:
    """
    Print Profile Selector output.

    This section corresponds to:
        User Request
            ↓
        Profile Selector
            ↓
        Selected Profile
    """

    selected_profile = selection_result["selected_profile"]
    dataset_name = selected_profile["dataset_identity"]["dataset_name"]

    print("\n[2] PROFILE SELECTOR OUTPUT")
    print("-" * 60)
    print(f"Selected profile:    {dataset_name}")
    print(f"Selection score:     {selection_result.get('selection_score')}")

    print("Selection reason:")
    for reason in selection_result.get("selection_reason", []):
        print(f"- {reason}")


def print_evaluator_manager_summary(manager_result: dict) -> None:
    """
    Print Evaluator Manager output.

    This section corresponds to:
        Selected Profile
            ↓
        Evaluator Manager
            ↓
        Selected Evaluators

    The manager is rule-controlled:
        - AI may help generate metadata or candidate evaluators.
        - But final evaluator selection is controlled by framework rules.
    """

    print("\n[3] EVALUATOR MANAGER OUTPUT")
    print("-" * 60)

    selected_evaluators = manager_result["selected_evaluators"]
    selection_reasons = manager_result["selection_reasons"]

    print("Selected evaluators:")

    for evaluator in selected_evaluators:
        print(f"\n- {evaluator}")
        for reason in selection_reasons.get(evaluator, []):
            print(f"  reason: {reason}")

    profile_candidates = manager_result.get("profile_candidates", [])

    print("\nProfile-suggested candidates:")
    if profile_candidates:
        for candidate in profile_candidates:
            print(f"- {candidate}")
    else:
        print("- None in current profile")


def print_evaluation_plan_summary(plan: dict) -> None:
    """
    Print a compact Evaluation Plan summary.

    This section corresponds to:
        Selected Evaluators
            ↓
        Evaluation Plan Generator
            ↓
        Evaluation Plan

    Evaluation Plan defines how candidate answers are evaluated.
    """

    print("\n[4] EVALUATION PLAN SUMMARY")
    print("-" * 60)

    plan_identity = plan.get("plan_identity", {})
    aggregation = plan.get("aggregation", {})

    print(f"Plan ID:             {plan_identity.get('plan_id')}")
    print(f"Plan name:           {plan_identity.get('plan_name')}")
    print(f"Aggregation rule:    {aggregation.get('aggregation_rule')}")

    print("\nEvaluation sequence:")

    for step in plan.get("evaluation_sequence", []):
        print(f"\nStep {step.get('step')}:")
        print(f"  Evaluator:         {step.get('evaluator')}")
        print(f"  Rule name:         {step.get('rule_name')}")
        print(f"  Evaluator type:    {step.get('evaluator_type')}")
        print(f"  Metrics:           {', '.join(step.get('metrics', []))}")
        print(f"  Prompt:            {step.get('prompt_template')}")

        rules = step.get("rules", {})
        if rules:
            print("  Main evaluator rules:")
            for key, value in list(rules.items())[:5]:
                print(f"    - {key}: {value}")

    print("\nScoring strategy:")
    for metric, weight in aggregation.get("scoring_strategy", {}).items():
        print(f"- {metric}: {weight}")

    hard_constraints = aggregation.get("hard_constraints", {})
    if hard_constraints:
        print("\nHard constraints:")
        for name, rule in hard_constraints.items():
            print(f"- {name}: {rule.get('reason')}")


def print_evaluator_planner_summary(
    planner_config: dict,
    planner_rule_explanation: dict,
    selected_questions: list[dict],
) -> None:
    """
    Print Evaluator Planner output and planner rules.

    This section corresponds to the left large circle in the supervisor's diagram:

        Generate a plan:
        - collection plan
        - progress plan
        - question-selection plan

    Important:
        Planner rules are different from evaluator rules.

        Evaluator rules:
            define how answers are evaluated.

        Planner rules:
            define how questions are delivered to the answer generator/user.
    """

    print("\n[5] EVALUATOR PLANNER OUTPUT")
    print("-" * 60)

    print(f"Collection plan:         {planner_config.get('collection_strategy')}")
    print(f"Progress plan:           {planner_config.get('progress_strategy')}")
    print(f"Question selection plan: {planner_config.get('question_selection_strategy')}")
    print(f"Batch size:              {planner_config.get('batch_size')}")
    print(f"Max questions:           {planner_config.get('max_questions')}")
    print(f"Repeat probability:      {planner_config.get('repeat_probability')}")

    print("\nPlanner rules:")

    collection_rule = planner_rule_explanation["collection_plan"]["rule"]
    progress_rule = planner_rule_explanation["progress_plan"]["rule"]
    q_selection_rule = planner_rule_explanation["question_selection_plan"]["rule"]

    print(f"- Collection rule: {collection_rule['description']}")
    print(f"  behavior: {collection_rule['behavior']}")

    print(f"- Progress rule: {progress_rule['description']}")
    print(f"  behavior: {progress_rule['behavior']}")

    print(f"- Question selection rule: {q_selection_rule['description']}")
    print(f"  behavior: {q_selection_rule['behavior']}")

    print("\nSelected questions preview:")

    for index, question in enumerate(selected_questions, start=1):
        question_text = question.get("question", "")

        if len(question_text) > 120:
            question_text = question_text[:120] + "..."

        print(f"{index}. {question_text}")


def log_demo_activity(
    logger: UserActivityLogger,
    user_id: str,
    user_request: dict,
    selected_profile: dict,
    manager_result: dict,
    evaluation_plan: dict,
    planner_config: dict,
    selected_questions: list[dict],
) -> None:
    """
    Write user activity records for the demo.

    The supervisor asked for a user activity table containing:
    - date/time
    - selected dataset
    - selected evaluator
    - selected evaluator plan
    - answered question
    - not answered question

    At the current planning stage:
    - selected questions are known
    - answered_question_ids is empty
    - not_answered_question_ids contains selected question IDs

    After runtime evaluator execution is added:
    - answered_question_ids will be updated
    - metric scores and final score will also be logged
    """

    dataset_id = selected_profile["dataset_identity"]["dataset_name"]
    profile_id = dataset_id
    plan_id = evaluation_plan["plan_identity"]["plan_id"]
    selected_evaluators = manager_result["selected_evaluators"]

    selected_question_ids = [
        question.get("question_id")
        for question in selected_questions
    ]

    not_answered_question_ids = selected_question_ids[:]

    logger.log_event(
        user_id=user_id,
        event_type="user_request_received",
        dataset_id=dataset_id,
        status="success",
        metadata={
            "user_request": user_request,
        },
    )

    logger.log_event(
        user_id=user_id,
        event_type="profile_selected",
        dataset_id=dataset_id,
        profile_id=profile_id,
        status="success",
        metadata={
            "selected_profile": profile_id,
        },
    )

    logger.log_event(
        user_id=user_id,
        event_type="evaluators_selected",
        dataset_id=dataset_id,
        profile_id=profile_id,
        selected_evaluators=selected_evaluators,
        status="success",
        metadata={
            "selection_reasons": manager_result.get("selection_reasons", {}),
        },
    )

    logger.log_event(
        user_id=user_id,
        event_type="evaluation_plan_generated",
        dataset_id=dataset_id,
        profile_id=profile_id,
        plan_id=plan_id,
        selected_evaluators=selected_evaluators,
        status="success",
        metadata={
            "evaluation_sequence": evaluation_plan.get("evaluation_sequence", []),
            "aggregation": evaluation_plan.get("aggregation", {}),
        },
    )

    logger.log_event(
        user_id=user_id,
        event_type="question_delivery_plan_generated",
        dataset_id=dataset_id,
        profile_id=profile_id,
        plan_id=plan_id,
        selected_evaluators=selected_evaluators,
        status="success",
        metadata={
            "planner_config": planner_config,
            "selected_question_ids": selected_question_ids,
            "answered_question_ids": [],
            "not_answered_question_ids": not_answered_question_ids,
        },
    )


def print_activity_summary_table(log_path: str) -> None:
    """
    Print a compact user activity summary table.

    The raw user activity log is stored as JSONL event records.
    This function converts the latest run into a readable summary.

    It focuses on the fields required by the supervisor:
    - date/time
    - selected dataset
    - selected profile
    - selected evaluators
    - selected evaluator plan
    - answered questions
    - not answered questions
    """

    path = Path(log_path)

    if not path.exists():
        print("No user activity log found.")
        return

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    if not lines:
        print("User activity log is empty.")
        return

    records = [json.loads(line) for line in lines]

    latest_user_request = None
    latest_profile = None
    latest_evaluators = None
    latest_plan = None
    latest_question_plan = None

    for record in records:
        event_type = record.get("event_type")

        if event_type == "user_request_received":
            latest_user_request = record

        elif event_type == "profile_selected":
            latest_profile = record

        elif event_type == "evaluators_selected":
            latest_evaluators = record

        elif event_type == "evaluation_plan_generated":
            latest_plan = record

        elif event_type == "question_delivery_plan_generated":
            latest_question_plan = record

    latest_time = None

    for record in [
        latest_question_plan,
        latest_plan,
        latest_evaluators,
        latest_profile,
        latest_user_request,
    ]:
        if record:
            latest_time = record.get("created_at")
            break

    dataset_id = None
    profile_id = None
    selected_evaluators = []
    plan_id = None
    answered_question_ids = []
    not_answered_question_ids = []

    if latest_question_plan:
        dataset_id = latest_question_plan.get("dataset_id")
        profile_id = latest_question_plan.get("profile_id")
        plan_id = latest_question_plan.get("plan_id")
        selected_evaluators = latest_question_plan.get("selected_evaluators", [])

        metadata = latest_question_plan.get("metadata", {})
        answered_question_ids = metadata.get("answered_question_ids", [])
        not_answered_question_ids = metadata.get("not_answered_question_ids", [])

    elif latest_plan:
        dataset_id = latest_plan.get("dataset_id")
        profile_id = latest_plan.get("profile_id")
        plan_id = latest_plan.get("plan_id")
        selected_evaluators = latest_plan.get("selected_evaluators", [])

    elif latest_evaluators:
        dataset_id = latest_evaluators.get("dataset_id")
        profile_id = latest_evaluators.get("profile_id")
        selected_evaluators = latest_evaluators.get("selected_evaluators", [])

    print("\nUser activity summary table:")
    print("-" * 60)
    print(f"Date/time:                 {latest_time}")
    print(f"Selected dataset:          {dataset_id}")
    print(f"Selected profile:          {profile_id}")
    print(f"Selected evaluators:       {selected_evaluators}")
    print(f"Selected evaluator plan:   {plan_id}")
    print(f"Answered questions:        {answered_question_ids}")
    print(f"Not answered questions:    {not_answered_question_ids}")


def clear_log_file(log_path: str) -> None:
    """
    Clear the log file before running demo.

    This is useful for supervisor meetings because it prevents old demo runs
    from mixing with the current output.
    """

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def main():
    """
    Run compact framework demonstration.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        help="Dataset key, for example hotpotqa, ai2_arc_challenge, strategyqa.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of samples to load.",
    )

    parser.add_argument(
        "--clear-log",
        action="store_true",
        help="Clear user activity log before running the demo.",
    )

    args = parser.parse_args()

    if args.clear_log:
        clear_log_file(LOG_PATH)

    print("\nAI-Based Adaptive Quiz Evaluation Framework")
    print("=" * 60)

    user_id = "demo_user"

    # ------------------------------------------------------------
    # Step 0:
    # Simulated user request.
    # ------------------------------------------------------------
    user_request = {
        "dataset_key": args.dataset,
    }

    print("\n[0] USER REQUEST")
    print("-" * 60)
    print(f"User selected dataset: {user_request['dataset_key']}")

    # ------------------------------------------------------------
    # Step 1:
    # Build Dataset Profile from real Hugging Face dataset.
    # ------------------------------------------------------------
    selected_dataset_profile = build_profile_for_dataset(
        dataset_key=args.dataset,
        sample_size=args.sample_size,
    )

    available_profiles = [
        selected_dataset_profile,
    ]

    print_profile_summary(selected_dataset_profile)

    # ------------------------------------------------------------
    # Step 2:
    # Profile Selector.
    # ------------------------------------------------------------
    profile_selector = ProfileSelector()

    selection_result = profile_selector.select_profile(
        user_request=user_request,
        available_profiles=available_profiles,
    )

    print_profile_selection_summary(selection_result)

    selected_profile = selection_result["selected_profile"]

    # ------------------------------------------------------------
    # Step 3:
    # Evaluator Manager.
    # ------------------------------------------------------------
    evaluation_manager = EvaluationManager()

    manager_result = evaluation_manager.select_evaluators_with_reasons(
        selected_profile,
    )

    print_evaluator_manager_summary(manager_result)

    selected_evaluators = manager_result["selected_evaluators"]

    # ------------------------------------------------------------
    # Step 4:
    # Evaluation Plan Generator.
    # ------------------------------------------------------------
    plan_generator = EvaluationPlanGenerator()

    evaluation_plan = plan_generator.generate_plan(
        profile=selected_profile,
        selected_evaluators=selected_evaluators,
    )

    print_evaluation_plan_summary(evaluation_plan)

    # ------------------------------------------------------------
    # Step 5:
    # Evaluator Planner.
    # ------------------------------------------------------------
    evaluator_planner = EvaluatorPlanner()

    planner_config = evaluator_planner.build_planner_config(
        collection_strategy="one_by_one_ordered",
        question_selection_strategy="unseen_only",
        progress_strategy="partial_eval",
        batch_size=1,
        max_questions=min(3, args.sample_size),
        repeat_probability=0.0,
        random_seed=42,
    )

    planner_rule_explanation = explain_planner_config(planner_config)

    selected_questions = evaluator_planner.select_questions(
        questions=selected_profile["sample_profile"]["sample_questions"],
        answered_question_ids=set(),
        planner_config=planner_config,
    )

    print_evaluator_planner_summary(
        planner_config=planner_config,
        planner_rule_explanation=planner_rule_explanation,
        selected_questions=selected_questions,
    )

    # ------------------------------------------------------------
    # Step 6:
    # User Activity Log.
    # ------------------------------------------------------------
    logger = UserActivityLogger(
        log_path=LOG_PATH,
    )

    log_demo_activity(
        logger=logger,
        user_id=user_id,
        user_request=user_request,
        selected_profile=selected_profile,
        manager_result=manager_result,
        evaluation_plan=evaluation_plan,
        planner_config=planner_config,
        selected_questions=selected_questions,
    )

    print("\n[6] USER ACTIVITY LOG")
    print("-" * 60)
    print("User activity events saved to:")
    print(str(Path(LOG_PATH)))

    print_activity_summary_table(
        log_path=LOG_PATH,
    )

    print("\nFramework planning demo completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()