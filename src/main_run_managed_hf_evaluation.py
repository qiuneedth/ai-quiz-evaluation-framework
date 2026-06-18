# src/main_run_managed_hf_evaluation.py

"""
Managed Hugging Face Evaluation Demo.

This is the current complete framework flow:

Hugging Face dataset
→ adapter
→ sample profile
→ Evaluation Manager
→ selected evaluators
→ Evaluator Executor
→ evaluation report
→ save results
"""

import json
import os

from src.data.hf_adapters import (
    load_hotpotqa_samples,
    load_strategyqa_samples,
    load_ragbench_samples,
)

from src.core.evaluation_manager import build_evaluation_plan
from src.core.evaluator_executor import run_evaluation_plan
from src.llm.client import LLMClient


def build_profile_from_sample(sample: dict) -> dict:
    """
    Build a profile from one unified sample.

    This profile is used by the Evaluation Manager.
    """

    question_type = sample.get("question_type", "unknown")
    has_context = bool(sample.get("context"))
    has_options = bool(sample.get("options"))

    if question_type == "true_false":
        answer_type = "boolean"
    elif question_type == "mcq":
        answer_type = "multiple_choice"
    else:
        answer_type = "free_text"

    return {
        "dataset_name": sample.get("dataset", "unknown"),
        "question_type": question_type,
        "has_context": has_context,
        "has_options": has_options,
        "answer_type": answer_type,
        "reasoning_required": has_context or question_type in ["true_false", "open_ended"]
    }


def generate_candidate_answer(sample: dict, llm: LLMClient) -> str:
    """
    Generate or reuse candidate answer.

    For RAGBench:
    - use existing dataset response.

    For true/false:
    - force answer to true or false.

    For context datasets:
    - answer using context.
    """

    existing_response = sample.get("metadata", {}).get("existing_response")

    if existing_response:
        return existing_response

    question_type = sample.get("question_type")
    context = sample.get("context")

    if question_type == "true_false":
        prompt = f"""
Answer the following question.

Return only one word:
true or false

Question:
{sample["question"]}
"""
        system_prompt = "Return only true or false."

    elif context:
        prompt = f"""
Answer the question using only the provided context.

Question:
{sample["question"]}

Context:
{context}
"""
        system_prompt = "Use only the provided context."

    else:
        prompt = f"""
Answer the question.

Question:
{sample["question"]}
"""
        system_prompt = "Answer accurately."

    return llm.chat(
        system_prompt=system_prompt,
        user_prompt=prompt,
    )


def run_one_sample(sample: dict, llm: LLMClient) -> dict:
    """
    Run one sample through the full managed pipeline.
    """

    profile = build_profile_from_sample(sample)

    evaluation_plan = build_evaluation_plan(profile)

    candidate_answer = generate_candidate_answer(sample, llm)
    sample["candidate_answer"] = candidate_answer

    report = run_evaluation_plan(
        sample=sample,
        evaluation_plan=evaluation_plan,
        llm_client=llm,
    )

    return {
        "profile": profile,
        "evaluation_plan": evaluation_plan,
        "report": report
    }


def print_short_result(result: dict) -> None:
    """
    Print readable result in terminal.
    """

    profile = result["profile"]
    plan = result["evaluation_plan"]
    report = result["report"]

    print("=" * 100)
    print("Dataset:", profile["dataset_name"])
    print("Question Type:", profile["question_type"])
    print("Has Context:", profile["has_context"])

    print("\nQuestion:")
    print(report["question"])

    print("\nCandidate Answer:")
    print(report["candidate_answer"])

    print("\nSelected Evaluators:")
    print(plan["selected_evaluators"])

    print("\nFinal Summary:")
    print(json.dumps(report["final_summary"], indent=2, ensure_ascii=False))


def save_results(results: list[dict]) -> None:
    """
    Save reports to JSON file.
    """

    os.makedirs("data/results", exist_ok=True)

    output_path = "data/results/managed_hf_evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved results to:")
    print(output_path)


def main():
    """
    Run managed evaluation on real Hugging Face datasets.
    """

    llm = LLMClient()

    all_samples = []

    all_samples.extend(load_hotpotqa_samples(limit=1))
    all_samples.extend(load_strategyqa_samples(limit=1))
    all_samples.extend(load_ragbench_samples(limit=1))

    results = []

    for sample in all_samples:
        result = run_one_sample(sample, llm)
        results.append(result)
        print_short_result(result)

    save_results(results)

    print("\n===== DONE =====")
    print("Completed managed evaluation pipeline:")
    print("HF dataset → profile → manager → evaluators → report")


if __name__ == "__main__":
    main()