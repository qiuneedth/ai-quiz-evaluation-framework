# src/main_run_hf_evaluation.py

"""
Run real evaluation experiments on Hugging Face datasets.

Pipeline:

Hugging Face dataset
→ adapter
→ unified quiz samples
→ candidate answer
→ evaluator
→ report

This connects the dataset layer to the evaluation layer.
"""

from src.data.hf_adapters import (
    load_hotpotqa_samples,
    load_strategyqa_samples,
    load_ragbench_samples,
)

from src.core.evaluator import (
    evaluate_rag_rubric,
    evaluate_llm_judge,
    evaluate_rule_based,
)

from src.llm.client import LLMClient


def generate_candidate_answer(sample: dict, llm: LLMClient) -> str:
    """
    Generate or retrieve candidate answer.

    candidate_answer means:
    the answer that will be evaluated.

    For RAGBench:
    - the dataset already contains a generated response
    - we use it as candidate_answer

    For other datasets:
    - we ask the LLM to generate an answer
    """

    existing_response = sample.get("metadata", {}).get("existing_response")

    if existing_response:
        return existing_response

    context = sample.get("context")

    if context:
        user_prompt = f"""
Answer the question using the provided context.

Question:
{sample["question"]}

Context:
{context}
"""
        system_prompt = "Answer using only the provided context."

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


def run_one_sample(sample: dict, llm: LLMClient) -> None:
    """
    Run one sample through the evaluator.
    """

    print("=" * 100)
    print("Dataset:", sample["dataset"])
    print("Question Type:", sample["question_type"])

    print("\nQuestion:")
    print(sample["question"])

    candidate_answer = generate_candidate_answer(sample, llm)
    sample["candidate_answer"] = candidate_answer

    print("\nCandidate Answer:")
    print(candidate_answer)

    print("\nReference Answer:")
    print(sample.get("reference_answer"))

    if sample["question_type"] == "true_false":
        print("\n--- Rule-based Evaluation ---")
        result = evaluate_rule_based(sample)

    elif sample.get("context"):
        print("\n--- RAG + Rubric Evaluation ---")
        result = evaluate_rag_rubric(sample, llm)

    else:
        print("\n--- LLM-as-a-Judge Evaluation ---")
        result = evaluate_llm_judge(sample, llm)

    print("\nEvaluation Result:")
    print(result)


def main():
    """
    Run small Hugging Face evaluation demo.
    """

    llm = LLMClient()

    hotpotqa_samples = load_hotpotqa_samples(limit=2)
    strategyqa_samples = load_strategyqa_samples(limit=2)
    ragbench_samples = load_ragbench_samples(limit=2)

    print("\n===== HOTPOTQA =====")
    for sample in hotpotqa_samples:
        run_one_sample(sample, llm)

    print("\n===== STRATEGYQA =====")
    for sample in strategyqa_samples:
        run_one_sample(sample, llm)

    print("\n===== RAGBENCH =====")
    for sample in ragbench_samples:
        run_one_sample(sample, llm)


if __name__ == "__main__":
    main()