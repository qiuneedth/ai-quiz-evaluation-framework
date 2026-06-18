# src/core/evaluator_contract.py

"""
Evaluator Input / Output Contract.

This file defines the common input and output format for all evaluators.

Why this file is important:
The supervisor emphasized that all evaluator types should follow the same
input/output structure.

Evaluator families may be different:
- script-based evaluators
- LLM-based evaluators
- context-based LLM evaluators

But they should all receive a standard input bundle and return a standard
output object.

This makes the rest of the system simpler:
- Quiz Session Manager can call any evaluator in the same way.
- Result Logger can store evaluator outputs in the same format.
- Report Generator can summarize results consistently.

Important terminology:
"user_answer" means the candidate answer to be evaluated.
It can come from:
- a human user
- an AI model
- an existing generated response in a dataset
- a mock answer during testing
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluatorInput:
    """
    Standard input object for all evaluators.

    Fields:
        question_id:
            Unique question identifier.

        question_type:
            Formal question type, for example:
            - true_false
            - multiple_choice
            - factoid_qa
            - context_grounded_qa
            - multi_hop_reasoning
            - rag_response_evaluation

        question:
            Question text.

        reference_answer:
            Gold / expected answer if available.

        user_answer:
            Candidate answer to evaluate.
            This may come from a human user, an AI model, or a dataset.

        context:
            Optional context passage or retrieved documents.

        keywords:
            Optional list of key concepts for keyword/concept coverage.

        options:
            Optional choices for multiple-choice questions.

        metadata:
            Extra information, such as topic, difficulty, dataset name.
    """

    question_id: str
    question_type: str
    question: str
    reference_answer: str | None = None
    user_answer: str | None = None
    context: str | None = None
    keywords: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the evaluator input into a plain dictionary.
        """

        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "user_answer": self.user_answer,
            "context": self.context,
            "keywords": self.keywords,
            "options": self.options,
            "metadata": self.metadata,
        }


@dataclass
class EvaluatorOutput:
    """
    Standard output object for all evaluators.

    Fields:
        evaluator_name:
            Name of evaluator that produced this result.

        score:
            Main score between 0 and 1.

        passed:
            Boolean pass/fail decision.

        details:
            Metric-level scores or explanations.

        feedback:
            Short human-readable feedback.

        raw_output:
            Optional raw output from LLM or script.
    """

    evaluator_name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    raw_output: Any = None

    def to_dict(self) -> dict:
        """
        Convert the evaluator output into a plain dictionary.
        """

        return {
            "evaluator_name": self.evaluator_name,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "feedback": self.feedback,
            "raw_output": self.raw_output,
        }


def build_evaluator_input_from_question(
    question: dict,
    question_type: str,
    user_answer: str | None,
) -> EvaluatorInput:
    """
    Build EvaluatorInput from a normalized question dictionary.

    This helper makes the session manager simpler.

    The function is defensive because different datasets may use slightly
    different field names.
    """

    question_id = (
        question.get("question_id")
        or question.get("id")
        or question.get("qid")
        or "unknown_question_id"
    )

    question_text = (
        question.get("question")
        or question.get("query")
        or question.get("claim")
        or ""
    )

    reference_answer = (
        question.get("reference_answer")
        or question.get("answer")
        or question.get("label")
    )

    context = (
        question.get("context")
        or question.get("document")
        or question.get("documents")
        or question.get("passage")
    )

    if isinstance(context, list):
        context = "\n".join(str(item) for item in context)

    if isinstance(context, dict):
        context = str(context)

    options = (
        question.get("options")
        or question.get("choices")
        or []
    )

    if isinstance(options, dict):
        options = options.get("text") or options.get("label") or []

    keywords = question.get("keywords") or []

    metadata = question.get("metadata") or {}

    return EvaluatorInput(
        question_id=str(question_id),
        question_type=question_type,
        question=str(question_text),
        reference_answer=None if reference_answer is None else str(reference_answer),
        user_answer=None if user_answer is None else str(user_answer),
        context=None if context is None else str(context),
        keywords=list(keywords) if isinstance(keywords, list) else [],
        options=list(options) if isinstance(options, list) else [],
        metadata=metadata,
    )