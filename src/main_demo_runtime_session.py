# src/main_demo_runtime_session.py

"""
Runtime Session Demo.

This script demonstrates the runtime flow of the AI-Based Adaptive
Quiz Evaluation Framework.

Correct system flow:

User Request JSON / Interactive Request
    ↓
Request Selector
    ↓
Selected Dataset Profile
    ↓
Quiz Plan Generator
    ↓
Round-Based Quiz Session
    ↓
Answer Provider
    ↓
Evaluation Engine
    ↓
Progressive Report / Final Report
    ↓
Runtime Log

Important:
The main entry should NOT be only command-line dataset selection.

Preferred demo:
    python -m src.main_demo_runtime_session \
      --request-file configs/request_science_manual.json \
      --clear-log

Interactive demo:
    python -m src.main_demo_runtime_session \
      --interactive-request \
      --clear-log
"""


import argparse
from pathlib import Path
import json

from src.core.env_loader import load_project_env

from src.data.hf_dataset_registry import list_available_hf_datasets
from src.data.hf_dataset_loader import load_hf_dataset_samples

from src.core.dataset_analyzer import analyze_dataset_to_profile
from src.core.metadata_postprocessor import postprocess_profile

from src.core.user_request import build_user_request_from_args
from src.core.request_loader import load_user_request_from_json
from src.core.interactive_request_builder import build_user_request_interactively
from src.core.request_selector import RequestSelector

from src.core.answer_provider import AnswerProvider
from src.core.quiz_plan_generator import QuizPlanGenerator
from src.core.quiz_session_manager import QuizSessionManager
from src.core.simple_evaluation_engine import SimpleEvaluationEngine

from src.storage.user_activity_logger import UserActivityLogger
from src.reports.report_generator import ReportGenerator


LOG_PATH = "data/results/runtime_user_activity.jsonl"
FINAL_REPORT_PATH = "data/results/final_report.json"
PROGRESSIVE_REPORT_PATH = "data/results/progressive_reports.jsonl"


def clear_file(path: str) -> None:
    """
    Clear a result file before running demo.
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")


def build_profile_for_dataset(
    dataset_key: str,
    sample_size: int,
) -> dict:
    """
    Load one Hugging Face dataset and build a postprocessed dataset profile.

    This corresponds to:
        Dataset
        → Dataset Analyzer
        → Dataset Profile
        → Metadata Postprocessor

    Important:
    The quiz planner needs actual question samples to generate a quiz.
    Therefore, after postprocessing, this function explicitly attaches
    adapted_samples back to the profile using several common keys.

    Without this step, some profile versions may contain metadata only,
    and the QuizPlanGenerator may see:
        Available questions in profile: 0
    """

    loaded = load_hf_dataset_samples(
        dataset_key=dataset_key,
        sample_size=sample_size,
    )

    dataset_config = loaded["dataset_config"]
    adapted_samples = loaded["adapted_samples"]

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

    profile = postprocess_profile(
        profile=profile,
        samples=adapted_samples,
        dataset_key=dataset_key,
    )

    # ------------------------------------------------------------
    # Critical fix:
    # Make sure QuizPlanGenerator can always find the real samples.
    # ------------------------------------------------------------
    profile["samples"] = adapted_samples
    profile["sample_questions"] = adapted_samples
    profile["questions"] = adapted_samples
    profile["adapted_samples"] = adapted_samples

    # ------------------------------------------------------------
    # Critical fix:
    # Store dataset identity at top level so later modules do not
    # show unknown_dataset / unknown_profile.
    # ------------------------------------------------------------
    profile["dataset_id"] = dataset_key
    profile["dataset_key"] = dataset_key
    profile["profile_id"] = dataset_key

    return profile


def build_available_profiles(
    user_request: dict,
    sample_size: int,
) -> list[dict]:
    """
    Build available profiles for the Request Selector.

    If user_request has dataset_key:
        only load that dataset.

    If dataset_key is missing:
        load all registered Hugging Face datasets.
    """

    dataset_key = user_request.get("dataset_key")

    if dataset_key:
        dataset_keys = [dataset_key]
    else:
        dataset_keys = list_available_hf_datasets()

    profiles = []

    for key in dataset_keys:
        try:
            profile = build_profile_for_dataset(
                dataset_key=key,
                sample_size=sample_size,
            )

            profiles.append(profile)

        except Exception as error:
            print(f"Skipped dataset '{key}' because of error: {error}")

    return profiles


def print_user_request(
    user_request: dict,
    request_source: str,
) -> None:
    """
    Print normalized user request.
    """

    print("\n[0] USER REQUEST")
    print("-" * 60)
    print(f"Request source: {request_source}")

    for key, value in user_request.items():
        print(f"{key}: {value}")


def print_selection_result(
    selection_result: dict,
) -> None:
    """
    Print Request Selector output.
    """

    selected_profile = selection_result["selected_profile"]
    completed_request = selection_result["completed_user_request"]

    identity = selected_profile.get("dataset_identity", {})
    structure = selected_profile.get("dataset_structure", {})
    semantic = selected_profile.get("semantic_profile", {})
    qa = selected_profile.get("question_answer_profile", {})

    print("\n[2] REQUEST SELECTOR OUTPUT")
    print("-" * 60)
    print(f"Selected dataset:     {identity.get('dataset_name')}")
    print(f"Dataset category:     {structure.get('dataset_category')}")
    print(f"Domain:               {semantic.get('domain')}")
    print(f"Topic:                {semantic.get('topic')}")
    print(f"Question type:        {qa.get('question_type')}")
    print(f"Answer type:          {qa.get('answer_type')}")
    print(f"Reasoning type:       {qa.get('reasoning_type')}")
    print(f"Context dependency:   {qa.get('context_dependency')}")

    print("\nCompleted user request:")
    for key, value in completed_request.items():
        print(f"- {key}: {value}")

    print("\nSelection reason:")
    for reason in selection_result.get("selection_reason", []):
        print(f"- {reason}")


def print_no_topic_match_plan(
    quiz_plan: dict,
) -> None:
    """
    Print a friendly message when no topic-matched questions are found.

    This is the terminal version of a future interface behavior:
        No matching questions found.
        [Change topic] [Choose another dataset] [Allow fallback]
    """

    print("\n[3] QUIZ PLAN")
    print("-" * 60)
    print("Quiz plan was not created because no topic-matched questions were found.")
    print(f"Dataset ID: {quiz_plan.get('dataset_id')}")
    print(f"Profile ID: {quiz_plan.get('profile_id')}")
    print(f"Stop reason: {quiz_plan.get('stop_reason')}")

    selection_summary = quiz_plan.get("selection_summary", {})
    warnings = selection_summary.get("warnings", [])

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    profile_debug = selection_summary.get("profile_topic_match_debug", {})

    if profile_debug:
        print("\nProfile topic match debug:")
        print(f"- matched: {profile_debug.get('matched')}")
        print(f"- reason: {profile_debug.get('reason')}")
        print(f"- matched_terms: {profile_debug.get('matched_terms')}")

    topic_match_debug = selection_summary.get("topic_match_debug", [])

    if topic_match_debug:
        print("\nQuestion topic match debug:")
        for item in topic_match_debug:
            question_text = str(item.get("question", ""))

            if len(question_text) > 120:
                question_text = question_text[:120] + "..."

            print(f"- Question ID: {item.get('question_id')}")
            print(f"  matched: {item.get('matched')}")
            print(f"  reason: {item.get('reason')}")
            print(f"  matched_terms: {item.get('matched_terms')}")
            print(f"  llm_checked: {item.get('llm_checked')}")
            print(f"  llm_relevance: {item.get('llm_relevance')}")
            print(f"  question: {question_text}")

    print("\nRuntime session stopped before quiz session.")
    print(
        "In the future interface, the user should be asked to change topic, "
        "select another dataset, or explicitly allow fallback questions."
    )


def print_quiz_plan_summary(
    quiz_plan: dict,
) -> None:
    """
    Print quiz plan summary.

    This includes:
    - session configuration
    - planner configuration
    - selection summary
    - topic matching debug
    - rounds
    - assigned evaluators
    """

    if quiz_plan.get("plan_status") == "no_topic_match":
        print_no_topic_match_plan(quiz_plan)
        return

    config = quiz_plan.get("session_config", {})
    planner = quiz_plan.get("planner_config", {})

    print("\n[3] QUIZ PLAN")
    print("-" * 60)
    print(f"Quiz plan ID:         {quiz_plan.get('quiz_plan_id')}")
    print(f"Dataset ID:           {quiz_plan.get('dataset_id')}")
    print(f"Profile ID:           {quiz_plan.get('profile_id')}")
    print(f"Plan status:          {quiz_plan.get('plan_status', 'ready')}")

    print("\nSession config:")
    print(f"- Total questions:      {config.get('total_questions')}")
    print(f"- Questions per round:  {config.get('questions_per_round')}")
    print(f"- Delivery mode:        {config.get('delivery_mode')}")
    print(f"- Report mode:          {config.get('report_mode')}")
    print(f"- Answer source:        {config.get('answer_source')}")
    print(f"- Evaluator mode:       {config.get('evaluator_mode')}")

    print("\nPlanner config:")
    print(f"- Collection plan:          {planner.get('collection_plan')}")
    print(f"- Progress plan:            {planner.get('progress_plan')}")
    print(f"- Question selection plan:  {planner.get('question_selection_plan')}")
    print(f"- Topic match mode:         {planner.get('topic_match_mode')}")
    print(f"- Insufficient policy:      {planner.get('insufficient_question_policy')}")
    print(f"- Repeat probability:       {planner.get('repeat_probability')}")
    print(f"- Random seed:              {planner.get('random_seed')}")
    print(f"- Topic query:              {planner.get('topic_query')}")
    print(f"- Requested question types: {planner.get('requested_question_types')}")

    selection_summary = quiz_plan.get("selection_summary", {})

    candidate_count = (
        selection_summary.get("candidate_questions_after_filter")
        if selection_summary.get("candidate_questions_after_filter") is not None
        else selection_summary.get("candidate_questions_after_filtering")
    )

    print("\nSelection summary:")
    print(f"- Requested questions:              {selection_summary.get('requested_questions')}")
    print(f"- Available questions in profile:   {selection_summary.get('available_questions_in_profile')}")
    print(f"- Candidate questions after filter: {candidate_count}")
    print(f"- Topic query:                      {selection_summary.get('topic_query')}")
    print(f"- Topic match mode:                 {selection_summary.get('topic_match_mode')}")
    print(f"- Topic match source:               {selection_summary.get('topic_match_source')}")
    print(f"- Profile topic match:              {selection_summary.get('profile_topic_match')}")
    print(f"- Topic matched questions:          {selection_summary.get('topic_matched_questions')}")
    print(f"- Selected questions:               {selection_summary.get('selected_questions')}")

    warnings = selection_summary.get("warnings", [])

    if warnings:
        print("\nSelection warnings:")
        for warning in warnings:
            print(f"- {warning}")

    profile_debug = selection_summary.get("profile_topic_match_debug", {})

    if profile_debug:
        print("\nProfile topic match debug:")
        print(f"- matched: {profile_debug.get('matched')}")
        print(f"- reason: {profile_debug.get('reason')}")
        print(f"- matched_terms: {profile_debug.get('matched_terms')}")

    topic_match_debug = selection_summary.get("topic_match_debug", [])

    matched_items = [
        item for item in topic_match_debug
        if item.get("matched")
    ]

    if matched_items:
        print("\nTopic match debug - matched questions:")
        for item in matched_items:
            question_text = str(item.get("question", ""))

            if len(question_text) > 120:
                question_text = question_text[:120] + "..."

            print(f"- MATCHED {item.get('question_id')}")
            print(f"  reason: {item.get('reason')}")
            print(f"  matched_terms: {item.get('matched_terms')}")
            print(f"  llm_checked: {item.get('llm_checked')}")
            print(f"  llm_relevance: {item.get('llm_relevance')}")
            print(f"  question: {question_text}")

    print("\nRounds:")
    for round_info in quiz_plan.get("rounds", []):
        print(
            f"- Round {round_info['round_index']}: "
            f"{round_info['question_ids']}"
        )

    print("\nAssigned evaluators:")
    for question_id, evaluators in quiz_plan.get("assigned_evaluators", {}).items():
        print(f"- {question_id}: {evaluators}")


def print_session_result_summary(
    session_result: dict,
) -> None:
    """
    Print completed session result.
    """

    print("\n[5] SESSION RESULT")
    print("-" * 60)
    print(f"Session ID:             {session_result.get('session_id')}")
    print(f"Dataset ID:             {session_result.get('dataset_id')}")
    print(f"Profile ID:             {session_result.get('profile_id')}")
    print(f"Total questions:        {session_result.get('total_questions')}")
    print(f"Answered questions:     {session_result.get('answered_question_ids')}")
    print(f"Not answered questions: {session_result.get('not_answered_question_ids')}")
    print(f"Final score:            {session_result.get('final_score')}")

    if session_result.get("stop_reason"):
        print(f"Stop reason:            {session_result.get('stop_reason')}")

    print("\nPer-question results:")

    for result in session_result.get("question_results", []):
        question_text = result.get("question", "")

        if len(question_text) > 120:
            question_text = question_text[:120] + "..."

        print(f"\nRound:       {result.get('round_index')}")
        print(f"Question ID: {result.get('question_id')}")
        print(f"Type:        {result.get('question_type')}")
        print(f"Question:    {question_text}")
        print(f"Answer:      {result.get('user_answer')}")
        print(f"Reference:   {result.get('reference_answer')}")
        print(f"Score:       {result.get('final_score')}")
        print(f"Correct:     {result.get('is_correct')}")
        print(f"Evaluators:  {result.get('assigned_evaluators')}")

        for output in result.get("evaluator_outputs", []):
            print(
                f"  - {output.get('evaluator_name')}: "
                f"score={output.get('score')}, "
                f"passed={output.get('passed')}"
            )
            print(f"    feedback: {output.get('feedback')}")

        answer_explanation = result.get("answer_explanation", {}) or {}

        if answer_explanation:
            print("  Detailed explanation summary:")

            short_feedback = answer_explanation.get("short_feedback")
            question_understanding = answer_explanation.get("question_understanding")
            error_type = answer_explanation.get("error_type")
            reasoning_steps = answer_explanation.get("reasoning_steps", [])

            if short_feedback:
                print(f"    short_feedback: {short_feedback}")

            if question_understanding:
                print(f"    question_understanding: {question_understanding}")

            if reasoning_steps:
                print("    reasoning_steps:")
                for index, step in enumerate(reasoning_steps, start=1):
                    print(f"      {index}. {step}")

            if error_type:
                print(f"    error_type: {error_type}")


def build_user_request(
    args,
) -> tuple[dict, str]:
    """
    Build user request from one of three entry modes.

    Priority:
    1. --request-file
    2. --interactive-request
    3. CLI debug arguments
    """

    if args.request_file:
        user_request = load_user_request_from_json(args.request_file)
        return user_request, args.request_file

    if args.interactive_request:
        user_request = build_user_request_interactively()
        return user_request, "interactive_terminal"

    user_request = build_user_request_from_args(args).to_dict()
    user_request["request_id"] = "debug_cli_request"
    user_request["user_id"] = "demo_user"

    return user_request, "debug_cli_args"


def generate_quiz_plan_safely(
    quiz_plan_generator: QuizPlanGenerator,
    selected_profile: dict,
    completed_user_request: dict,
) -> dict:
    """
    Generate quiz plan safely.

    This function handles possible parameter-name differences between versions:
        profile=
        selected_profile=
    """

    try:
        return quiz_plan_generator.generate_quiz_plan(
            profile=selected_profile,
            user_request=completed_user_request,
        )

    except TypeError:
        return quiz_plan_generator.generate_quiz_plan(
            selected_profile=selected_profile,
            user_request=completed_user_request,
        )


def main():
    """
    Run runtime session demo.
    """

    load_project_env()

    parser = argparse.ArgumentParser()

    # Preferred entry.
    parser.add_argument(
        "--request-file",
        type=str,
        default=None,
        help="Path to user request JSON file. Preferred system entry.",
    )

    # Interactive entry.
    parser.add_argument(
        "--interactive-request",
        action="store_true",
        help="Build user request interactively in terminal.",
    )

    # Debug CLI entry.
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--question-types", type=str, default=None)

    parser.add_argument("--sample-size", type=int, default=5)

    parser.add_argument("--total-questions", type=int, default=3)
    parser.add_argument("--questions-per-round", type=int, default=1)

    parser.add_argument(
        "--collection-plan",
        type=str,
        default="1b1-ordered",
        choices=[
            "1b1-ordered",
            "1b1-random",
            "bBb-ordered",
            "bBb-random",
            "incremental",
            "totally_random",
        ],
    )

    parser.add_argument(
        "--progress-plan",
        type=str,
        default="partial_eval",
        choices=[
            "partial_eval",
            "full_eval",
        ],
    )

    parser.add_argument(
        "--question-selection-plan",
        type=str,
        default="unseen_only",
        choices=[
            "unseen_only",
            "all_questions",
            "review_mixed",
        ],
    )

    parser.add_argument(
        "--topic-match-mode",
        type=str,
        default="soft",
        choices=[
            "soft",
            "strict",
            "none",
        ],
    )

    parser.add_argument(
        "--insufficient-question-policy",
        type=str,
        default="allow_fallback",
        choices=[
            "allow_fallback",
            "strict_only",
            "stop_if_no_topic_match",
        ],
    )

    parser.add_argument("--repeat-probability", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument(
        "--delivery-mode",
        type=str,
        default="one_by_one",
        choices=[
            "one_by_one",
            "batch",
        ],
    )

    parser.add_argument(
        "--report-mode",
        type=str,
        default="progressive_and_final",
        choices=[
            "final_only",
            "progressive",
            "progressive_and_final",
            "none",
        ],
    )

    parser.add_argument(
        "--answer-source",
        type=str,
        default="manual",
        choices=[
            "manual",
            "mock_reference",
            "dataset_generated",
            "llm_generated",
            "empty",
        ],
    )

    parser.add_argument(
        "--evaluator-mode",
        type=str,
        default="real",
        choices=[
            "real",
            "mock",
        ],
    )

    parser.add_argument("--allow-mock-fallback", action="store_true")
    parser.add_argument("--clear-log", action="store_true")

    args = parser.parse_args()

    if args.clear_log:
        clear_file(LOG_PATH)
        clear_file(FINAL_REPORT_PATH)
        clear_file(PROGRESSIVE_REPORT_PATH)

    print("\nAI-Based Adaptive Quiz Evaluation Framework: Runtime Session")
    print("=" * 60)

    # ------------------------------------------------------------
    # Step 0:
    # Build user request.
    # ------------------------------------------------------------
    user_request, request_source = build_user_request(args)

    print_user_request(
        user_request=user_request,
        request_source=request_source,
    )

    # ------------------------------------------------------------
    # Step 1:
    # Build available dataset profiles.
    # ------------------------------------------------------------
    print("\n[1] BUILD AVAILABLE DATASET PROFILES")
    print("-" * 60)

    available_profiles = build_available_profiles(
        user_request=user_request,
        sample_size=args.sample_size,
    )

    print(f"Available profiles built: {len(available_profiles)}")

    if not available_profiles:
        raise RuntimeError("No dataset profiles available.")

    # ------------------------------------------------------------
    # Step 2:
    # Request Selector.
    # ------------------------------------------------------------
    selector = RequestSelector()

    selection_result = selector.select(
        user_request=user_request,
        available_profiles=available_profiles,
        interactive=True,
    )

    print_selection_result(selection_result)

    completed_user_request = selection_result["completed_user_request"]
    selected_profile = selection_result["selected_profile"]

    # ------------------------------------------------------------
    # Step 3:
    # Generate quiz plan.
    # ------------------------------------------------------------
    quiz_plan_generator = QuizPlanGenerator()

    try:
        quiz_plan = generate_quiz_plan_safely(
            quiz_plan_generator=quiz_plan_generator,
            selected_profile=selected_profile,
            completed_user_request=completed_user_request,
        )

    except ValueError as error:
        print("\n[3] QUIZ PLAN")
        print("-" * 60)
        print("Quiz plan could not be generated.")
        print(f"Reason: {error}")
        print("\nRuntime session stopped before quiz session.")
        print("Please choose another topic, another dataset, or change topic matching settings.")
        return

    if quiz_plan.get("plan_status") == "no_topic_match":
        print_no_topic_match_plan(quiz_plan)
        print("\nRuntime session completed.")
        print("=" * 60)
        return

    print_quiz_plan_summary(quiz_plan)

    # ------------------------------------------------------------
    # Extra safety:
    # Do not run an empty quiz session.
    # ------------------------------------------------------------
    if quiz_plan.get("plan_status") == "ready" and not quiz_plan.get("selected_questions"):
        print("\nRuntime session stopped before quiz session.")
        print("Reason: quiz plan is ready, but no questions were selected.")
        print("Please check whether question samples are attached to the dataset profile.")
        print("\nRuntime session completed.")
        print("=" * 60)
        return

    # ------------------------------------------------------------
    # Step 4:
    # Run quiz session.
    # ------------------------------------------------------------
    print("\n[4] RUN ROUND-BASED QUIZ SESSION")
    print("-" * 60)

    logger = UserActivityLogger(
        log_path=LOG_PATH,
    )

    answer_provider = AnswerProvider()

    evaluation_engine = SimpleEvaluationEngine(
        evaluator_mode=completed_user_request.get("evaluator_mode", "real"),
        allow_mock_fallback=completed_user_request.get("allow_mock_fallback", False),
    )

    session_manager = QuizSessionManager(
        evaluation_engine=evaluation_engine,
        answer_provider=answer_provider,
        logger=logger,
    )

    session_result = session_manager.run_session(
        quiz_plan=quiz_plan,
        user_id=completed_user_request.get("user_id", "demo_user"),
    )

    print_session_result_summary(session_result)

    # ------------------------------------------------------------
    # Step 5:
    # Save reports.
    # ------------------------------------------------------------
    print("\n[6] SAVE REPORTS")
    print("-" * 60)

    report_generator = ReportGenerator()

    report_mode = completed_user_request.get("report_mode", "progressive_and_final")

    if report_mode in ["progressive", "progressive_and_final"]:
        report_generator.save_progressive_reports(
            progressive_reports=session_result.get("progressive_reports", []),
            output_path=PROGRESSIVE_REPORT_PATH,
        )
        print(f"Progressive reports saved to: {PROGRESSIVE_REPORT_PATH}")

    if report_mode in ["final_only", "progressive_and_final"]:
        final_report = report_generator.generate_report(session_result)

        report_generator.save_report(
            report=final_report,
            output_path=FINAL_REPORT_PATH,
        )

        print(f"Final report saved to:       {FINAL_REPORT_PATH}")

        print("\nFinal report summary:")
        print(
            json.dumps(
                {
                    "report_type": final_report.get("report_type"),
                    "dataset_id": final_report.get("dataset_id"),
                    "profile_id": final_report.get("profile_id"),
                    "total_questions": final_report.get("total_questions"),
                    "answered_questions": final_report.get("answered_questions"),
                    "not_answered_questions": final_report.get("not_answered_questions"),
                    "final_score": final_report.get("final_score"),
                    "evaluator_usage": final_report.get("evaluator_usage"),
                    "stop_reason": final_report.get("stop_reason"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    print(f"Runtime log saved to:        {LOG_PATH}")

    print("\nRuntime session completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()