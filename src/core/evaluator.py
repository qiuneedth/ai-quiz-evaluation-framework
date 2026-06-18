# src/core/evaluator.py

"""
Evaluation methods.

This file contains the main evaluation approaches:

1. RAG + Rubric Evaluation
   - uses context
   - evaluates groundedness
   - suitable for context-based datasets

2. LLM-as-a-Judge
   - does not use context
   - suitable for no-context datasets

3. Exact Match Baseline
   - simple baseline
   - useful for comparison

4. Rule-based Evaluation
   - suitable for MCQ and true/false questions
"""

import json


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for simple comparison.
    """

    return str(text).strip().lower()


def normalize_tf_answer(text: str) -> str:
    """
    Normalize true/false answers.
    """

    text = normalize_answer(text)

    if text in ["true", "t", "yes", "correct"]:
        return "true"

    if text in ["false", "f", "no", "incorrect"]:
        return "false"

    return text


def normalize_mcq_answer(text: str) -> str:
    """
    Normalize MCQ answer.

    Example:
    "A. First Amendment" → "a"
    "A" → "a"
    """

    text = normalize_answer(text)

    if not text:
        return text

    first_char = text[0]

    if first_char in ["a", "b", "c", "d", "e"]:
        return first_char

    return text


def evaluate_exact_match(sample: dict) -> dict:
    """
    Simple exact match baseline.

    This checks whether the reference answer appears in the candidate answer.
    """

    candidate = normalize_answer(sample["candidate_answer"])
    reference = normalize_answer(sample.get("reference_answer", ""))

    score = 1.0 if reference and reference in candidate else 0.0

    return {
        "scores": {
            "exact_match": score
        },
        "final_score": score,
        "label": "correct" if score == 1.0 else "incorrect",
        "explanation": "Exact match baseline."
    }


def evaluate_rule_based(sample: dict) -> dict:
    """
    Rule-based evaluation for MCQ and true/false.

    This corresponds to:
    Question-Type → Method Mapping:
    - MCQ → rule-based / exact match
    - True/False → rule-based / exact match
    """

    question_type = sample.get("question_type")

    candidate = sample.get("candidate_answer", "")
    reference = sample.get("reference_answer", "")

    if question_type == "mcq":
        candidate_norm = normalize_mcq_answer(candidate)
        reference_norm = normalize_mcq_answer(reference)

    elif question_type == "true_false":
        candidate_norm = normalize_tf_answer(candidate)
        reference_norm = normalize_tf_answer(reference)

    else:
        candidate_norm = normalize_answer(candidate)
        reference_norm = normalize_answer(reference)

    score = 1.0 if candidate_norm == reference_norm else 0.0

    return {
        "scores": {
            "correctness": score
        },
        "final_score": score,
        "label": "correct" if score == 1.0 else "incorrect",
        "explanation": (
            f"Rule-based comparison. "
            f"Normalized candidate='{candidate_norm}', "
            f"normalized reference='{reference_norm}'."
        )
    }


def evaluate_rag_rubric(sample: dict, llm_client) -> dict:
    """
    RAG + Rubric Evaluation.

    This method uses:
    - question
    - reference answer
    - candidate answer
    - context

    It evaluates:
    - correctness
    - relevance
    - completeness
    - groundedness
    """

    prompt = f"""
You are an AI quiz evaluator.

Evaluate the candidate answer using the question, reference answer, and context.

Question:
{sample["question"]}

Reference Answer:
{sample.get("reference_answer", "")}

Candidate Answer:
{sample["candidate_answer"]}

Context:
{sample.get("context", "")}

Evaluate these dimensions:
- correctness: Is the answer factually correct?
- relevance: Does it answer the question?
- completeness: Does it include enough required information?
- groundedness: Is it supported by the context?

Return only valid JSON:
{{
  "scores": {{
    "correctness": 0.0,
    "relevance": 0.0,
    "completeness": 0.0,
    "groundedness": 0.0
  }},
  "final_score": 0.0,
  "label": "correct | partially_correct | incorrect",
  "explanation": "short explanation"
}}
"""

    raw = llm_client.chat(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt
    )

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "scores": {},
            "final_score": 0.0,
            "label": "json_error",
            "explanation": raw
        }


def evaluate_llm_judge(sample: dict, llm_client) -> dict:
    """
    LLM-as-a-Judge.

    This method does NOT use context.

    It uses:
    - question
    - reference answer
    - candidate answer

    It evaluates:
    - correctness
    - relevance
    - completeness
    """

    prompt = f"""
You are an AI quiz evaluator.

Evaluate the candidate answer using only the question and reference answer.

Question:
{sample["question"]}

Reference Answer:
{sample.get("reference_answer", "")}

Candidate Answer:
{sample["candidate_answer"]}

Evaluate these dimensions:
- correctness: Is the answer correct?
- relevance: Does it answer the question?
- completeness: Is it complete enough?

Return only valid JSON:
{{
  "scores": {{
    "correctness": 0.0,
    "relevance": 0.0,
    "completeness": 0.0
  }},
  "final_score": 0.0,
  "label": "correct | partially_correct | incorrect",
  "explanation": "short explanation"
}}
"""

    raw = llm_client.chat(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt
    )

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "scores": {},
            "final_score": 0.0,
            "label": "json_error",
            "explanation": raw
        }