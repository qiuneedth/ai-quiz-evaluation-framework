# src/main.py

"""
Main entry point of the framework.

This experiment demonstrates:

1. Context-based dataset
   - longsamples.jsonl
   - compare RAG + Rubric vs LLM Judge

2. No-context dataset
   - NQ.jsonl
   - compare LLM Judge vs Exact Match

3. Objective question dataset
   - mcqsamples.jsonl
   - tfsamples.jsonl
   - use rule-based evaluation

Goal:
dataset/question type → choose suitable evaluation method → compare results
"""

import os

from src.utils.jsonl import load_jsonl
from src.data.adapters import adapt_custom_sample
from src.llm.client import LLMClient
from src.core.evaluator import (
    evaluate_rag_rubric,
    evaluate_llm_judge,
    evaluate_exact_match,
    evaluate_rule_based,
)


RAW_FILES = [
    "data/raw/longsamples.jsonl",   # context-based short answer
    "data/raw/NQ.jsonl",            # no-context QA
    "data/raw/mcqsamples.jsonl",    # multiple choice
    "data/raw/tfsamples.jsonl",     # true / false
]


def load_all_samples() -> list[dict]:
    """
    Load all local datasets and convert them into the unified schema.
    """

    samples = []

    for file_path in RAW_FILES:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        rows = load_jsonl(file_path)
        print(f"Loaded {len(rows)} from {file_path}")

        for idx, row in enumerate(rows):
            sample = adapt_custom_sample(row, idx)
            sample["source"] = file_path
            samples.append(sample)

    return samples


def generate_candidate_answer(sample: dict, llm: LLMClient) -> str:
    """
    Generate a candidate answer using the LLM.

    For context questions:
    - use context

    For no-context questions:
    - answer from model knowledge

    For MCQ:
    - return only the option letter

    For true/false:
    - return only true or false
    """

    question_type = sample.get("question_type")
    context = sample.get("context")
    options = sample.get("options")

    if question_type == "mcq":
        options_text = "\n".join(options or [])

        user_prompt = f"""
Answer the multiple-choice question.

Return only one option letter, such as A, B, C, or D.

Question:
{sample["question"]}

Options:
{options_text}
"""
        system_prompt = "You answer MCQ questions. Return only the option letter."

    elif question_type == "true_false":
        user_prompt = f"""
Answer the true/false question.

Return only true or false.

Question:
{sample["question"]}
"""
        system_prompt = "You answer true/false questions. Return only true or false."

    elif context:
        user_prompt = f"""
Answer the question using the provided context.

Question:
{sample["question"]}

Context:
{context}
"""
        system_prompt = "Use the provided context to answer."

    else:
        user_prompt = f"""
Answer the question.

Question:
{sample["question"]}
"""
        system_prompt = "Answer accurately."

    return llm.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def run_context_dataset_sample(sample: dict, llm: LLMClient) -> None:
    """
    Context dataset:
    Compare RAG + Rubric vs LLM Judge.
    """

    print("=" * 80)
    print("[EXPERIMENT A: CONTEXT DATASET]")
    print("Question:", sample["question"])

    candidate_answer = generate_candidate_answer(sample, llm)
    sample["candidate_answer"] = candidate_answer

    print("\nGenerated Candidate Answer:")
    print(candidate_answer)

    print("\n--- Method 1: RAG + Rubric Evaluation ---")
    rag_result = evaluate_rag_rubric(sample, llm)
    print(rag_result)

    print("\n--- Method 2: LLM-as-a-Judge Without Context ---")
    llm_result = evaluate_llm_judge(sample, llm)
    print(llm_result)


def run_no_context_dataset_sample(sample: dict, llm: LLMClient) -> None:
    """
    No-context dataset:
    Compare LLM Judge vs Exact Match baseline.
    """

    print("=" * 80)
    print("[EXPERIMENT B: NO-CONTEXT DATASET]")
    print("Question:", sample["question"])

    candidate_answer = generate_candidate_answer(sample, llm)
    sample["candidate_answer"] = candidate_answer

    print("\nGenerated Candidate Answer:")
    print(candidate_answer)

    print("\n--- Method 1: LLM-as-a-Judge ---")
    llm_result = evaluate_llm_judge(sample, llm)
    print(llm_result)

    print("\n--- Method 2: Exact Match Baseline ---")
    em_result = evaluate_exact_match(sample)
    print(em_result)


def run_objective_dataset_sample(sample: dict, llm: LLMClient) -> None:
    """
    Objective dataset:
    MCQ / true-false should use rule-based evaluation.
    """

    print("=" * 80)
    print("[EXPERIMENT C: OBJECTIVE QUESTION DATASET]")
    print("Question Type:", sample["question_type"])
    print("Question:", sample["question"])

    if sample.get("options"):
        print("\nOptions:")
        for option in sample["options"]:
            print(option)

    candidate_answer = generate_candidate_answer(sample, llm)
    sample["candidate_answer"] = candidate_answer

    print("\nGenerated Candidate Answer:")
    print(candidate_answer)

    print("\nReference Answer:")
    print(sample.get("reference_answer"))

    print("\n--- Method: Rule-based / Exact Match ---")
    rule_result = evaluate_rule_based(sample)
    print(rule_result)


def main():
    samples = load_all_samples()
    print(f"\nTotal samples: {len(samples)}")

    llm = LLMClient()

    context_samples = [
        s for s in samples
        if "longsamples" in s["source"]
    ]

    no_context_samples = [
        s for s in samples
        if "NQ" in s["source"]
    ]

    mcq_samples = [
        s for s in samples
        if s.get("question_type") == "mcq"
    ]

    tf_samples = [
        s for s in samples
        if s.get("question_type") == "true_false"
    ]

    print(f"\nContext samples: {len(context_samples)}")
    print(f"No-context samples: {len(no_context_samples)}")
    print(f"MCQ samples: {len(mcq_samples)}")
    print(f"True/False samples: {len(tf_samples)}")

    print("\n===== Experiment A: Context dataset =====")
    for sample in context_samples[:3]:
        run_context_dataset_sample(sample, llm)

    print("\n===== Experiment B: No-context dataset =====")
    for sample in no_context_samples[:3]:
        run_no_context_dataset_sample(sample, llm)

    print("\n===== Experiment C1: MCQ dataset =====")
    for sample in mcq_samples[:3]:
        run_objective_dataset_sample(sample, llm)

    print("\n===== Experiment C2: True/False dataset =====")
    for sample in tf_samples[:3]:
        run_objective_dataset_sample(sample, llm)

    print("\n===== SUMMARY =====")
    print("Context dataset → RAG vs LLM Judge")
    print("No-context dataset → LLM Judge vs Exact Match")
    print("MCQ / True-False dataset → Rule-based evaluation")


if __name__ == "__main__":
    main()