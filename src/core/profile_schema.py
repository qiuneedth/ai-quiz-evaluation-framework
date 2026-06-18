# src/core/profile_schema.py

"""
Dataset Profile Schema.

This file defines the standard structure of a Dataset Profile.

Dataset Profile is the core knowledge object of the framework.

It describes a dataset from the perspective of evaluation.

It is used by:
- Dataset Analyzer
- Evaluation Manager
- Evaluation Plan Generator
- Evaluator Planner
- Evaluator Executor
- Report Generator

Important:
The profile does not execute evaluation.
The profile only stores structured metadata that helps the system decide
how the dataset should be evaluated.
"""


# ============================================================
# 1. Question type taxonomy
# ============================================================
# These types should be more formal than simple labels such as:
# "short_answer", "open question", or "QA".
#
# The purpose of this taxonomy is to help the Evaluation Manager
# select suitable evaluators.
QUESTION_TYPES = [
    "multiple_choice",
    "boolean_reasoning",
    "factoid_qa",
    "extractive_qa",
    "abstractive_qa",
    "context_grounded_qa",
    "multi_hop_reasoning",
    "open_ended_reasoning",
    "rag_response_evaluation",
    "unknown",
]


# ============================================================
# 2. Answer type taxonomy
# ============================================================
# This describes the expected format of the answer.
ANSWER_TYPES = [
    "single_label",
    "boolean",
    "span",
    "short_text",
    "long_text",
    "free_form_text",
    "multiple_choice_option",
    "generated_response",
    "list",
    "unknown",
]


# ============================================================
# 3. Reasoning type taxonomy
# ============================================================
# This describes what kind of reasoning is required.
REASONING_TYPES = [
    "none",
    "factual_recall",
    "single_hop_reasoning",
    "multi_hop_reasoning",
    "commonsense_reasoning",
    "logical_reasoning",
    "contextual_reasoning",
    "open_ended_reasoning",
    "unknown",
]


# ============================================================
# 4. Context dependency taxonomy
# ============================================================
# This describes whether the question requires context.
CONTEXT_DEPENDENCY = [
    "none",
    "provided_context_required",
    "retrieved_context_required",
    "optional_context",
    "unknown",
]


# ============================================================
# 5. Difficulty levels
# ============================================================
DIFFICULTY_LEVELS = [
    "easy",
    "medium",
    "hard",
    "mixed",
    "unknown",
]


# ============================================================
# 6. Evaluator candidates
# ============================================================
# These names should match the evaluator names used later by
# Evaluation Manager, Evaluation Plan Generator, and Evaluator Registry.
EVALUATION_CANDIDATES = [
    "rule_based_evaluator",
    "keyword_evaluator",
    "llm_evaluator",
    "llm_context_evaluator",
    "relevance_evaluator",
    "completeness_evaluator",
    "groundedness_evaluator",
]


# ============================================================
# 7. Possible metrics
# ============================================================
# These metrics can appear in the Evaluation Plan.
POSSIBLE_METRICS = [
    "exact_match",
    "keyword_overlap",
    "correctness",
    "relevance",
    "completeness",
    "groundedness",
    "faithfulness",
    "context_usage",
    "reasoning_quality",
    "clarity",
]


def normalize_value(value: str | None, allowed_values: list[str], default: str = "unknown") -> str:
    """
    Normalize one value into a controlled vocabulary.

    Why this is needed:
    LLM output may be unstable.

    For example, an LLM may return:
    - "short answer"
    - "short_answer"
    - "factoid"
    - "factual question"

    But the framework needs standard values.

    If the value is not in the allowed list, this function returns default.
    """

    if value is None:
        return default

    normalized = str(value).strip().lower()

    if normalized in allowed_values:
        return normalized

    return default


def normalize_list(values: list[str] | None, allowed_values: list[str]) -> list[str]:
    """
    Normalize a list into a controlled vocabulary.

    Invalid values are removed.

    This is useful for:
    - evaluation_candidates
    - possible_metrics
    """

    if not values:
        return []

    normalized_values = []

    for value in values:
        normalized = str(value).strip().lower()

        if normalized in allowed_values and normalized not in normalized_values:
            normalized_values.append(normalized)

    return normalized_values


def make_dataset_profile(
    dataset_name: str,
    dataset_link: str | None = None,
    dataset_category: str = "unknown",
    source: str = "unknown",
    split: str = "unknown",
    sample_count: int = 0,
    fields: list[str] | None = None,
    has_context: bool = False,
    has_reference_answer: bool = False,
    has_generated_response: bool = False,
    has_options: bool = False,
    domain: str = "general",
    topic: str = "unknown",
    location: str | None = None,
    question_type: str = "unknown",
    answer_type: str = "unknown",
    reasoning_type: str = "unknown",
    difficulty_level: str = "unknown",
    context_dependency: str = "unknown",
    expected_answer_format: str = "unknown",
    evaluation_candidates: list[str] | None = None,
    possible_metrics: list[str] | None = None,
    sample_questions: list[dict] | None = None,
    notes: str = "",
) -> dict:
    """
    Create a standardized Dataset Profile.

    This is the main profile format used by the framework.

    The profile contains seven parts:

    1. dataset_identity
       Basic information about the dataset.

    2. dataset_structure
       Structural information detected from dataset fields.

    3. question_answer_profile
       Evaluation-related information about the questions and answers.

    4. semantic_profile
       Domain and topic information.

    5. evaluation_profile
       Candidate evaluators and possible metrics.

    6. sample_profile
       Sample questions used for preview, testing, and question selection.

    7. notes
       Extra explanation, warning, or analysis notes.
    """

    # Make sure list fields are always valid lists.
    if fields is None:
        fields = []

    if evaluation_candidates is None:
        evaluation_candidates = []

    if possible_metrics is None:
        possible_metrics = []

    if sample_questions is None:
        sample_questions = []

    # Normalize controlled vocabulary values.
    question_type = normalize_value(
        value=question_type,
        allowed_values=QUESTION_TYPES,
    )

    answer_type = normalize_value(
        value=answer_type,
        allowed_values=ANSWER_TYPES,
    )

    reasoning_type = normalize_value(
        value=reasoning_type,
        allowed_values=REASONING_TYPES,
    )

    difficulty_level = normalize_value(
        value=difficulty_level,
        allowed_values=DIFFICULTY_LEVELS,
    )

    context_dependency = normalize_value(
        value=context_dependency,
        allowed_values=CONTEXT_DEPENDENCY,
    )

    # Normalize evaluator and metric lists.
    evaluation_candidates = normalize_list(
        values=evaluation_candidates,
        allowed_values=EVALUATION_CANDIDATES,
    )

    possible_metrics = normalize_list(
        values=possible_metrics,
        allowed_values=POSSIBLE_METRICS,
    )

    # Return one complete Dataset Profile object.
    return {
        "dataset_identity": {
            "dataset_name": dataset_name,
            "dataset_link": dataset_link,
            "source": source,
            "split": split,
            "location": location,
        },

        "dataset_structure": {
            "dataset_category": dataset_category,
            "sample_count": sample_count,
            "fields": fields,
            "has_context": has_context,
            "has_reference_answer": has_reference_answer,
            "has_generated_response": has_generated_response,
            "has_options": has_options,
        },

        "question_answer_profile": {
            "question_type": question_type,
            "answer_type": answer_type,
            "reasoning_type": reasoning_type,
            "difficulty_level": difficulty_level,
            "context_dependency": context_dependency,
            "expected_answer_format": expected_answer_format,
        },

        "semantic_profile": {
            "domain": domain,
            "topic": topic,
        },

        "evaluation_profile": {
            "evaluation_candidates": evaluation_candidates,
            "possible_metrics": possible_metrics,
        },

        "sample_profile": {
            "sample_questions": sample_questions,
        },

        "notes": notes,
    }