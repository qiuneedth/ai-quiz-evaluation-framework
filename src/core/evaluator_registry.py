# src/core/evaluator_registry.py

"""
Evaluator Registry.

This file stores the catalog of available evaluator types.

Important:
This file does NOT define detailed scoring rules.
Detailed scoring rules are defined in evaluator_rulebook.py.

Difference:

1. evaluator_registry.py
   Tells the framework what evaluators exist.

2. evaluator_rulebook.py
   Defines how each evaluator evaluates an answer.

3. evaluation_manager.py
   Selects which evaluators should be used for a selected profile.

4. evaluation_plan_generator.py
   Combines selected evaluators and rulebook entries into an executable
   Evaluation Plan.
"""


EVALUATOR_REGISTRY = {
    "rule_based_evaluator": {
        "description": "Deterministic evaluator for multiple-choice, boolean, and exact-label questions.",
        "requires_context": False,
        "metrics": [
            "exact_match",
            "correctness",
        ],
    },

    "keyword_evaluator": {
        "description": "Checks keyword or concept overlap between reference answer and candidate answer.",
        "requires_context": False,
        "metrics": [
            "keyword_coverage",
        ],
    },

    "llm_evaluator": {
        "description": "LLM-assisted semantic correctness evaluator without context.",
        "requires_context": False,
        "metrics": [
            "correctness",
        ],
    },

    "llm_context_evaluator": {
        "description": "LLM-assisted context correctness evaluator using provided or retrieved context.",
        "requires_context": True,
        "metrics": [
            "correctness",
            "context_usage",
        ],
    },

    "relevance_evaluator": {
        "description": "Evaluates whether the candidate answer directly addresses the question.",
        "requires_context": False,
        "metrics": [
            "relevance",
        ],
    },

    "completeness_evaluator": {
        "description": "Evaluates whether the candidate answer covers all required information.",
        "requires_context": False,
        "metrics": [
            "completeness",
        ],
    },

    "groundedness_evaluator": {
        "description": "Evaluates whether factual claims are supported by the provided context.",
        "requires_context": True,
        "metrics": [
            "groundedness",
            "hallucination_rate",
        ],
    },

    "reasoning_quality_evaluator": {
        "description": "Evaluates whether the candidate answer shows valid reasoning.",
        "requires_context": False,
        "metrics": [
            "reasoning_quality",
        ],
    },
}


def get_evaluator_info(evaluator_name: str) -> dict:
    """
    Return information for one evaluator.

    If the evaluator is unknown, return a safe default.
    """

    return EVALUATOR_REGISTRY.get(
        evaluator_name,
        {
            "description": "Unknown evaluator",
            "requires_context": False,
            "metrics": [],
        },
    )


def list_available_evaluators() -> list[str]:
    """
    Return all available evaluator names.
    """

    return list(EVALUATOR_REGISTRY.keys())


def evaluator_exists(evaluator_name: str) -> bool:
    """
    Check whether an evaluator exists in the registry.
    """

    return evaluator_name in EVALUATOR_REGISTRY