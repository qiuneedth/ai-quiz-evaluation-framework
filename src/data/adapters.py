# src/data/adapters.py

"""
Convert different dataset formats into unified schema.
"""

from src.schemas import make_unified_sample


def adapt_custom_sample(row: dict, idx: int) -> dict:
    """
    Convert your local JSONL sample into unified format.
    """

    question = row.get("question")
    answer = row.get("answer")
    context = row.get("document")
    original_type = row.get("type", "short")

    # map type
    if original_type == "mcq":
        question_type = "mcq"
    elif original_type == "tf":
        question_type = "true_false"
    else:
        question_type = "short_answer"

    options = row.get("options", None)

    sample = make_unified_sample(
        sample_id=f"custom_{original_type}_{idx}",
        dataset="custom",

        question_type=question_type,
        question=question,

        reference_answer=answer,
        context=context,
        options=options,

        metadata={
            "original_type": original_type
        }
    )

    return sample