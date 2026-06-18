# src/core/interactive_request_builder.py

"""
Interactive Request Builder.

This module builds a user_request dictionary through terminal interaction.

Important:
The final system does NOT require users to manually write JSON.

The user_request dictionary can come from:
- JSON request file
- terminal interactive input
- frontend form
- API request
- future LLM natural-language parser

This file implements the terminal interactive input.

Design:
1. quick mode:
   Ask only a few user-friendly questions.
   Suitable for demo and normal use.

2. advanced mode:
   Ask detailed planner configuration.
   Suitable for developer testing and supervisor discussion.
"""


from src.core.user_request import build_user_request_from_dict

try:
    from src.data.hf_dataset_registry import list_available_hf_datasets
except Exception:
    list_available_hf_datasets = None


def build_user_request_interactively() -> dict:
    """
    Build a normalized user_request dictionary through terminal questions.

    Returns:
        Normalized user_request dictionary.
    """

    print("\nINTERACTIVE USER REQUEST BUILDER")
    print("=" * 60)
    print("This builder creates the internal user_request dictionary.")
    print("You do not need to manually write JSON.")
    print()

    mode = _ask_choice(
        "Request mode",
        choices=["quick", "advanced"],
        default="quick",
    )

    if mode == "quick":
        return _build_quick_request()

    return _build_advanced_request()


# ============================================================
# Quick mode
# ============================================================

def _build_quick_request() -> dict:
    """
    Quick user request builder.

    This is the recommended interactive mode for demos.

    It asks only essential questions and fills the technical planner fields
    with reasonable defaults.
    """

    print("\nQUICK REQUEST MODE")
    print("-" * 60)

    request_id = _ask_text(
        "Request ID",
        default="interactive_quick_request",
    )

    user_id = _ask_text(
        "User ID",
        default="demo_user",
    )

    print("\nWhat does the user want?")
    print("-" * 60)
    print("1. Choose a known dataset directly")
    print("2. Search by topic, for example science / moon / history / mathematics")

    request_target = _ask_choice(
        "Request target",
        choices=["dataset", "topic"],
        default="topic",
    )

    dataset_key = None
    topic_query = None
    domain = None
    question_types = []

    if request_target == "dataset":
        available_keys = _get_available_dataset_keys()

        if available_keys:
            print("\nAvailable dataset keys:")
            for key in available_keys:
                print(f"- {key}")

        dataset_key = _ask_text(
            "Dataset key",
            default="hotpotqa",
        )

        # If the user typed something that is not a known dataset,
        # ask whether it should be treated as a topic instead.
        if available_keys and dataset_key not in available_keys:
            print()
            print(f"'{dataset_key}' is not a registered dataset key.")
            print("Registered dataset keys are:")
            for key in available_keys:
                print(f"- {key}")

            treat_as_topic = _ask_yes_no(
                f"Do you want to treat '{dataset_key}' as a topic instead?",
                default=True,
            )

            if treat_as_topic:
                topic_query = dataset_key
                dataset_key = None

    else:
        topic_query = _ask_text(
            "Topic query, for example science / moon / history / mathematics",
            default="science",
        )

        domain = _ask_text(
            "Domain, optional. Press Enter to skip",
            default=None,
        )

        question_types_text = _ask_text(
            "Question types, optional comma-separated. Example: multiple_choice, multi_hop_reasoning. Press Enter for no restriction",
            default=None,
        )

        question_types = _parse_list(question_types_text)

    print("\nQuiz settings")
    print("-" * 60)

    total_questions = _ask_int(
        "Total questions",
        default=5,
    )

    questions_per_round = _ask_int(
        "Questions per round",
        default=1,
    )

    print("\nQuestion order")
    print("-" * 60)

    random_questions = _ask_yes_no(
        "Randomize selected questions?",
        default=False,
    )

    if questions_per_round <= 1:
        collection_plan = "1b1-random" if random_questions else "1b1-ordered"
        delivery_mode = "one_by_one"
    else:
        collection_plan = "bBb-random" if random_questions else "bBb-ordered"
        delivery_mode = "batch"

    print("\nAnswer mode")
    print("-" * 60)
    print("manual = real user answers questions")
    print("llm_generated = AI answers questions, experimental evaluator test mode")

    answer_source = _ask_choice(
        "Answer source",
        choices=["manual", "llm_generated", "mock_reference", "empty"],
        default="manual",
    )

    evaluator_mode = _ask_choice(
        "Evaluator mode",
        choices=["real", "mock"],
        default="real",
    )

    raw_request = {
        "request_id": request_id,
        "user_id": user_id,

        "dataset_request": {
            "dataset_key": dataset_key,
            "topic_query": topic_query,
            "domain": domain,
            "question_types": question_types,
            "allow_profile_suggestion": True,
            "selection_scope": "single_profile",
        },

        "quiz_request": {
            "total_questions": total_questions,
            "questions_per_round": questions_per_round,
        },

        "planner": {
            "collection_plan": collection_plan,
            "progress_plan": "partial_eval",
            "question_selection_plan": "unseen_only",
            "topic_match_mode": "soft",
            "insufficient_question_policy": "allow_fallback",
            "repeat_probability": 0.0,
            "random_seed": 42,
            "seen_question_ids": [],
        },

        "answer_request": {
            "answer_source": answer_source,
            "evaluator_mode": evaluator_mode,
            "allow_mock_fallback": False,
        },

        "report_request": {
            "progressive_report": True,
            "final_report": True,
        },
    }

    user_request = build_user_request_from_dict(raw_request).to_dict()

    _print_generated_request(user_request)

    return user_request


# ============================================================
# Advanced mode
# ============================================================

def _build_advanced_request() -> dict:
    """
    Advanced user request builder.

    This mode exposes the planner fields from the supervisor's diagram:
    - collection plan
    - progress plan
    - question selection plan
    - topic matching mode
    - insufficient question policy
    - repeat probability
    """

    print("\nADVANCED REQUEST MODE")
    print("-" * 60)
    print("Press Enter to use default values.")
    print()

    request_id = _ask_text(
        "Request ID",
        default="interactive_advanced_request",
    )

    user_id = _ask_text(
        "User ID",
        default="demo_user",
    )

    print("\nDataset / Topic Request")
    print("-" * 60)

    available_keys = _get_available_dataset_keys()

    if available_keys:
        print("Available dataset keys:")
        for key in available_keys:
            print(f"- {key}")
        print()

    dataset_key = _ask_text(
        "Dataset key. Leave empty if you want the system to search by topic",
        default=None,
    )

    topic_query = None
    domain = None
    question_types = []

    if dataset_key and available_keys and dataset_key not in available_keys:
        print()
        print(f"'{dataset_key}' is not a registered dataset key.")

        treat_as_topic = _ask_yes_no(
            f"Do you want to treat '{dataset_key}' as a topic instead?",
            default=True,
        )

        if treat_as_topic:
            topic_query = dataset_key
            dataset_key = None

    if not dataset_key:
        topic_query = topic_query or _ask_text(
            "Topic query, for example science / moon / history / mathematics",
            default=None,
        )

        domain = _ask_text(
            "Domain, optional",
            default=None,
        )

        question_types_text = _ask_text(
            "Question types, comma-separated. Examples: multiple_choice, multi_hop_reasoning, boolean_reasoning. Leave empty for no restriction",
            default=None,
        )

        question_types = _parse_list(question_types_text)

    print("\nQuiz Request")
    print("-" * 60)

    total_questions = _ask_int(
        "Total questions",
        default=5,
    )

    questions_per_round = _ask_int(
        "Questions per round",
        default=1,
    )

    print("\nPlanner Request")
    print("-" * 60)
    print("Collection plan options:")
    print("- 1b1-ordered")
    print("- 1b1-random")
    print("- bBb-ordered")
    print("- bBb-random")
    print("- incremental")
    print("- totally_random")

    collection_plan = _ask_choice(
        "Collection plan",
        choices=[
            "1b1-ordered",
            "1b1-random",
            "bBb-ordered",
            "bBb-random",
            "incremental",
            "totally_random",
        ],
        default="1b1-ordered",
    )

    print("\nProgress plan options:")
    print("- partial_eval")
    print("- full_eval")

    progress_plan = _ask_choice(
        "Progress plan",
        choices=[
            "partial_eval",
            "full_eval",
        ],
        default="partial_eval",
    )

    print("\nQuestion selection plan options:")
    print("- unseen_only")
    print("- all_questions")
    print("- review_mixed")

    question_selection_plan = _ask_choice(
        "Question selection plan",
        choices=[
            "unseen_only",
            "all_questions",
            "review_mixed",
        ],
        default="unseen_only",
    )

    repeat_probability = 0.0

    if question_selection_plan == "review_mixed":
        repeat_probability = _ask_float(
            "Repeat probability for seen questions, for example 0.2",
            default=0.2,
        )

    print("\nTopic match mode options:")
    print("- soft: if topic-matched questions are not enough, allow fallback questions")
    print("- strict: only use topic-matched questions")
    print("- none: do not filter questions by topic")

    topic_match_mode = _ask_choice(
        "Topic match mode",
        choices=[
            "soft",
            "strict",
            "none",
        ],
        default="soft",
    )

    insufficient_question_policy = _ask_choice(
        "Insufficient question policy",
        choices=[
            "allow_fallback",
            "strict_only",
        ],
        default="allow_fallback",
    )

    random_seed = _ask_int(
        "Random seed",
        default=42,
    )

    print("\nAnswer Request")
    print("-" * 60)
    print("Answer source options:")
    print("- manual: real user answers in terminal")
    print("- llm_generated: AI generates candidate answers, experimental mode")
    print("- mock_reference: use reference answers, pipeline test only")
    print("- empty: empty answers, error case test")

    answer_source = _ask_choice(
        "Answer source",
        choices=[
            "manual",
            "llm_generated",
            "mock_reference",
            "empty",
        ],
        default="manual",
    )

    evaluator_mode = _ask_choice(
        "Evaluator mode",
        choices=[
            "real",
            "mock",
        ],
        default="real",
    )

    print("\nReport Request")
    print("-" * 60)

    progressive_report = _ask_yes_no(
        "Generate progressive reports after each round?",
        default=True,
    )

    final_report = _ask_yes_no(
        "Generate final report at the end?",
        default=True,
    )

    raw_request = {
        "request_id": request_id,
        "user_id": user_id,

        "dataset_request": {
            "dataset_key": dataset_key,
            "topic_query": topic_query,
            "domain": domain,
            "question_types": question_types,
            "allow_profile_suggestion": True,
            "selection_scope": "single_profile",
        },

        "quiz_request": {
            "total_questions": total_questions,
            "questions_per_round": questions_per_round,
        },

        "planner": {
            "collection_plan": collection_plan,
            "progress_plan": progress_plan,
            "question_selection_plan": question_selection_plan,
            "topic_match_mode": topic_match_mode,
            "insufficient_question_policy": insufficient_question_policy,
            "repeat_probability": repeat_probability,
            "random_seed": random_seed,
            "seen_question_ids": [],
        },

        "answer_request": {
            "answer_source": answer_source,
            "evaluator_mode": evaluator_mode,
            "allow_mock_fallback": False,
        },

        "report_request": {
            "progressive_report": progressive_report,
            "final_report": final_report,
        },
    }

    user_request = build_user_request_from_dict(raw_request).to_dict()

    _print_generated_request(user_request)

    return user_request


# ============================================================
# Helper functions
# ============================================================

def _get_available_dataset_keys() -> list[str]:
    """
    Return available dataset keys if registry is available.
    """

    if list_available_hf_datasets is None:
        return []

    try:
        return list_available_hf_datasets()
    except Exception:
        return []


def _print_generated_request(
    user_request: dict,
) -> None:
    """
    Print generated normalized request dictionary.
    """

    print("\nGenerated user_request dictionary:")
    print("-" * 60)

    for key, value in user_request.items():
        print(f"{key}: {value}")


def _ask_text(
    prompt: str,
    default=None,
):
    """
    Ask for text input.
    """

    if default is None:
        raw = input(f"{prompt}: ").strip()
    else:
        raw = input(f"{prompt} [{default}]: ").strip()

    if not raw:
        return default

    return raw


def _ask_int(
    prompt: str,
    default: int,
) -> int:
    """
    Ask for integer input.
    """

    while True:
        raw = input(f"{prompt} [{default}]: ").strip()

        if not raw:
            return default

        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def _ask_float(
    prompt: str,
    default: float,
) -> float:
    """
    Ask for float input.
    """

    while True:
        raw = input(f"{prompt} [{default}]: ").strip()

        if not raw:
            return default

        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def _ask_choice(
    prompt: str,
    choices: list[str],
    default: str,
) -> str:
    """
    Ask for one value from choices.
    """

    choices_text = " / ".join(choices)

    while True:
        raw = input(f"{prompt} ({choices_text}) [{default}]: ").strip()

        if not raw:
            return default

        if raw in choices:
            return raw

        print(f"Invalid choice. Please choose from: {choices_text}")


def _ask_yes_no(
    prompt: str,
    default: bool,
) -> bool:
    """
    Ask yes/no question.
    """

    default_text = "y" if default else "n"

    while True:
        raw = input(f"{prompt} (y/n) [{default_text}]: ").strip().lower()

        if not raw:
            return default

        if raw in ["y", "yes"]:
            return True

        if raw in ["n", "no"]:
            return False

        print("Please enter y or n.")


def _parse_list(
    value,
) -> list[str]:
    """
    Parse comma-separated values into list.
    """

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]