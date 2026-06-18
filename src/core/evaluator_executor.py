# src/core/evaluator_executor.py

"""
Evaluator Executor.

This file runs the selected evaluators.

Input:
- one unified sample
- one evaluation plan

Output:
- evaluation report
"""

import json

from src.core.evaluator import (
    evaluate_rule_based,
    evaluate_llm_judge,
    evaluate_rag_rubric,
    evaluate_exact_match,
)


def safe_json_loads(text: str) -> dict:
    """
    Safely parse JSON from LLM output.
    """

    try:
        return json.loads(text)
    except Exception:
        return {
            "scores": {},
            "final_score": 0.0,
            "label": "json_error",
            "explanation": text
        }


def evaluate_keyword(sample: dict) -> dict:
    """
    Keyword evaluator.

    It checks whether words from the reference answer
    appear in the candidate answer.
    """

    reference = str(sample.get("reference_answer", "")).lower()
    candidate = str(sample.get("candidate_answer", "")).lower()

    reference_words = set(reference.split())

    if not reference_words:
        return {
            "scores": {"keyword_coverage": 0.0},
            "final_score": 0.0,
            "label": "not_applicable",
            "explanation": "No reference answer available."
        }

    matched = [word for word in reference_words if word in candidate]
    score = len(matched) / len(reference_words)

    return {
        "scores": {"keyword_coverage": round(score, 3)},
        "final_score": round(score, 3),
        "label": "keyword_check",
        "explanation": f"Matched keywords: {matched}"
    }


def evaluate_relevancy(sample: dict, llm_client) -> dict:
    """
    LLM relevancy evaluator.

    Only checks if the candidate answer answers the question.
    """

    prompt = f"""
Evaluate only the relevance of the candidate answer.

Question:
{sample.get("question")}

Candidate Answer:
{sample.get("candidate_answer")}

Return only valid JSON:
{{
  "scores": {{
    "relevance": 0.0
  }},
  "final_score": 0.0,
  "label": "relevant | partially_relevant | irrelevant",
  "explanation": "short explanation"
}}
"""

    raw = llm_client.chat(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt
    )

    return safe_json_loads(raw)


def evaluate_completeness(sample: dict, llm_client) -> dict:
    """
    LLM completeness evaluator.

    Checks whether the candidate answer is complete enough.
    """

    prompt = f"""
Evaluate only the completeness of the candidate answer.

Question:
{sample.get("question")}

Reference Answer:
{sample.get("reference_answer")}

Candidate Answer:
{sample.get("candidate_answer")}

Return only valid JSON:
{{
  "scores": {{
    "completeness": 0.0
  }},
  "final_score": 0.0,
  "label": "complete | partially_complete | incomplete",
  "explanation": "short explanation"
}}
"""

    raw = llm_client.chat(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt
    )

    return safe_json_loads(raw)


def evaluate_groundedness(sample: dict, llm_client) -> dict:
    """
    LLM groundedness evaluator.

    Checks whether the answer is supported by the context.
    """

    prompt = f"""
Evaluate only the groundedness of the candidate answer.

Question:
{sample.get("question")}

Context:
{sample.get("context")}

Candidate Answer:
{sample.get("candidate_answer")}

Return only valid JSON:
{{
  "scores": {{
    "groundedness": 0.0
  }},
  "final_score": 0.0,
  "label": "grounded | partially_grounded | not_grounded",
  "explanation": "short explanation"
}}
"""

    raw = llm_client.chat(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt
    )

    return safe_json_loads(raw)


def run_single_evaluator(
    evaluator_name: str,
    sample: dict,
    llm_client=None,
) -> dict:
    """
    Run one evaluator by name.
    """

    if evaluator_name == "rule_based_evaluator":
        return evaluate_rule_based(sample)

    if evaluator_name == "keyword_evaluator":
        return evaluate_keyword(sample)

    if evaluator_name == "llm_evaluator":
        return evaluate_llm_judge(sample, llm_client)

    if evaluator_name == "llm_context_evaluator":
        return evaluate_rag_rubric(sample, llm_client)

    if evaluator_name == "relevancy_evaluator":
        return evaluate_relevancy(sample, llm_client)

    if evaluator_name == "completeness_evaluator":
        return evaluate_completeness(sample, llm_client)

    if evaluator_name == "groundedness_evaluator":
        return evaluate_groundedness(sample, llm_client)

    if evaluator_name == "exact_match_baseline":
        return evaluate_exact_match(sample)

    return {
        "scores": {},
        "final_score": 0.0,
        "label": "unknown_evaluator",
        "explanation": f"No function found for {evaluator_name}"
    }


def aggregate_scores(evaluation_results: dict) -> dict:
    """
    Aggregate evaluator scores into average scores per metric.
    """

    metric_values = {}

    for result in evaluation_results.values():
        scores = result.get("scores", {})

        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(metric, []).append(value)

    averaged = {}

    for metric, values in metric_values.items():
        averaged[metric] = round(sum(values) / len(values), 3)

    if averaged:
        final_score = round(sum(averaged.values()) / len(averaged), 3)
    else:
        final_score = 0.0

    return {
        "average_scores": averaged,
        "final_score": final_score
    }


def run_evaluation_plan(
    sample: dict,
    evaluation_plan: dict,
    llm_client=None,
) -> dict:
    """
    Run all selected evaluators and build report.
    """

    selected_evaluators = evaluation_plan.get("selected_evaluators", [])

    evaluation_results = {}

    for evaluator_name in selected_evaluators:
        result = run_single_evaluator(
            evaluator_name=evaluator_name,
            sample=sample,
            llm_client=llm_client,
        )

        evaluation_results[evaluator_name] = result

    summary = aggregate_scores(evaluation_results)

    return {
        "dataset_name": sample.get("dataset", "unknown"),
        "question_type": sample.get("question_type", "unknown"),
        "question": sample.get("question"),
        "candidate_answer": sample.get("candidate_answer"),
        "reference_answer": sample.get("reference_answer"),
        "has_context": bool(sample.get("context")),
        "selected_evaluators": selected_evaluators,
        "evaluation_results": evaluation_results,
        "final_summary": summary
    }