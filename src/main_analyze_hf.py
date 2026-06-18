# src/main_analyze_hf.py

"""
Analyze real Hugging Face datasets.

This file demonstrates the first major part of the framework:

Hugging Face dataset
→ load preview samples
→ extract dataset metadata
→ post-process metadata
→ route to evaluation plan
→ save dataset knowledge table

This is NOT a retrieval system.
This is a quiz evaluation framework component.

The goal is:
Given a dataset, automatically understand:
- what type of questions it contains
- whether it has context
- what evaluation methods are applicable
- what metrics should be used
"""

import json
import os

from src.data.hf_dataset_loader import load_hf_samples
from src.core.dataset_analyzer import analyze_dataset_with_llm
from src.core.metadata_postprocessor import postprocess_metadata
from src.core.evaluation_router import route_evaluation
from src.llm.client import LLMClient


# =========================================================
# Hugging Face datasets to analyze
# =========================================================

HF_DATASETS = [
    {
        "dataset_name": "hotpotqa/hotpot_qa",
        "subset_name": "distractor",
        "split": "train",
        "limit": 5,
        "description": (
            "HotpotQA is a multi-hop question answering dataset. "
            "It contains questions, answers, supporting facts, and context documents. "
            "It is useful for context-based reasoning and RAG-style evaluation."
        ),
    },
    {
        "dataset_name": "mandarjoshi/trivia_qa",
        "subset_name": "rc",
        "split": "train",
        "limit": 5,
        "description": (
            "TriviaQA is a reading comprehension QA dataset. "
            "It contains trivia questions, answers, and evidence documents. "
            "It is useful for answer correctness and evidence-based evaluation."
        ),
    },
    {
        "dataset_name": "ChilleD/StrategyQA",
        "subset_name": None,
        "split": "train",
        "limit": 5,
        "description": (
            "StrategyQA is a yes/no question answering dataset. "
            "Questions usually require implicit reasoning. "
            "It is useful for boolean QA and reasoning evaluation."
        ),
    },
    {
        "dataset_name": "galileo-ai/ragbench",
        "subset_name": "covidqa",
        "split": "train",
        "limit": 5,
        "description": (
            "RAGBench CovidQA is a RAG evaluation dataset. "
            "It contains questions, retrieved documents, generated responses, "
            "and evaluation-related scores. "
            "It is useful for groundedness, faithfulness, context relevance, "
            "and answer quality evaluation."
        ),
    },
]


# =========================================================
# Analyze one dataset
# =========================================================

def analyze_one_dataset(config: dict, llm: LLMClient) -> dict:
    """
    Analyze one Hugging Face dataset.

    This function performs the full first-layer process:

    1. Load preview samples from Hugging Face.
    2. Extract metadata with code + LLM.
    3. Post-process metadata using rules.
    4. Route metadata to an evaluation plan.
    5. Return one complete dataset record.
    """

    dataset_name = config["dataset_name"]
    subset_name = config["subset_name"]
    split = config["split"]
    limit = config["limit"]
    description = config["description"]

    print("=" * 100)
    print("Dataset:", dataset_name)
    print("Subset:", subset_name)
    print("Split:", split)

    # -----------------------------------------------------
    # Step 1: Load preview samples from Hugging Face
    # -----------------------------------------------------
    samples = load_hf_samples(
        dataset_name=dataset_name,
        subset_name=subset_name,
        split=split,
        limit=limit,
        streaming=True,
    )

    print(f"Loaded preview samples: {len(samples)}")

    if samples:
        print("Example fields:", list(samples[0].keys()))
    else:
        print("No samples loaded.")
        return {
            "dataset_name": dataset_name,
            "error": "No samples loaded"
        }

    # -----------------------------------------------------
    # Step 2: Analyze dataset and generate raw metadata
    # -----------------------------------------------------
    raw_metadata = analyze_dataset_with_llm(
        dataset_name=dataset_name,
        samples=samples,
        llm_client=llm,
        dataset_description=description,
    )

    # -----------------------------------------------------
    # Step 3: Post-process metadata
    # -----------------------------------------------------
    final_metadata = postprocess_metadata(raw_metadata)

    # -----------------------------------------------------
    # Step 4: Route metadata to evaluation plan
    # -----------------------------------------------------
    evaluation_plan = route_evaluation(final_metadata)

    # -----------------------------------------------------
    # Step 5: Combine everything into one dataset record
    # -----------------------------------------------------
    dataset_record = {
        "dataset_id": dataset_name,
        "subset_name": subset_name,
        "split": split,
        "preview_sample_count": len(samples),

        # Metadata table
        "metadata": final_metadata,

        # Evaluation plan generated from metadata
        "evaluation_plan": evaluation_plan,

        # Keep a few preview samples for debugging / demonstration
        "preview_samples": samples[:2],
    }

    print("\nFinal Metadata:")
    print(json.dumps(final_metadata, indent=2, ensure_ascii=False))

    print("\nEvaluation Plan:")
    print(json.dumps(evaluation_plan, indent=2, ensure_ascii=False))

    return dataset_record


# =========================================================
# Save dataset knowledge table
# =========================================================

def save_dataset_knowledge_table(records: list[dict]) -> None:
    """
    Save all dataset records to a JSON file.

    This file becomes the first version of the framework's
    dataset knowledge table.

    It contains:
    - dataset metadata
    - evaluation plan
    - preview samples
    """

    os.makedirs("data/metadata", exist_ok=True)

    output_path = "data/metadata/dataset_knowledge_table.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\nSaved dataset knowledge table to:")
    print(output_path)


# =========================================================
# Main
# =========================================================

def main():
    """
    Main process.

    This proves that the framework can:

    1. connect to real Hugging Face datasets
    2. inspect dataset samples
    3. extract dataset metadata
    4. clean metadata with rules
    5. generate an evaluation plan
    6. save the result as a dataset knowledge table
    """

    llm = LLMClient()

    dataset_records = []

    for config in HF_DATASETS:
        record = analyze_one_dataset(config, llm)
        dataset_records.append(record)

    save_dataset_knowledge_table(dataset_records)

    print("\n===== SUMMARY =====")
    print("Analyzed Hugging Face datasets:", len(dataset_records))
    print("Output: data/metadata/dataset_knowledge_table.json")
    print("This file is the dataset knowledge table for the quiz evaluator framework.")


if __name__ == "__main__":
    main()