# src/schemas.py

"""
Unified schema for all datasets.

This defines a STANDARD format for quiz evaluation.
"""


def make_unified_sample(
    sample_id: str,
    dataset: str,
    question_type: str,
    question: str,
    reference_answer: str = None,
    context: str = None,
    options: list = None,
    metadata: dict = None
) -> dict:
    """
    Create a unified sample.

    IMPORTANT:
    We DO NOT include candidate_answer here,
    because it will be generated later by the model.
    """

    return {
        "id": sample_id,
        "dataset": dataset,
        "question_type": question_type,
        "question": question,

        # ground truth
        "reference_answer": reference_answer,

        # context (for RAG)
        "context": context,

        # for MCQ
        "options": options,

        # will be filled later
        "candidate_answer": None,

        # extra info
        "metadata": metadata or {}
    }