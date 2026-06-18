# src/data/hf_dataset_registry.py

"""
Hugging Face Dataset Registry.

This file defines the Hugging Face datasets supported by this project.

The registry does NOT load datasets directly.
It only stores dataset configuration information.

Each dataset config contains:
- dataset_key: internal name used by our framework
- hf_path: Hugging Face dataset repository name
- hf_config: Hugging Face subset/config name
- split: dataset split
- dataset_link: browser link
- description: human-readable description

This registry is used by:
- hf_dataset_loader.py
- main_test_hf_new_structure.py
"""


HF_DATASET_REGISTRY = {
    # ------------------------------------------------------------
    # HotpotQA
    # ------------------------------------------------------------
    # Multi-hop QA dataset.
    # Useful for testing:
    # - context_grounded_qa
    # - multi_hop_reasoning
    # - completeness
    # - groundedness
    "hotpotqa": {
        "dataset_key": "hotpotqa",
        "hf_path": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "split": "train",
        "dataset_link": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "description": "HotpotQA distractor split for multi-hop context-grounded question answering.",
    },

    # ------------------------------------------------------------
    # RAGBench covidqa
    # ------------------------------------------------------------
    # RAG response evaluation dataset.
    # Useful for testing:
    # - rag_response_evaluation
    # - groundedness
    # - faithfulness
    # - context relevance
    "ragbench_covidqa": {
        "dataset_key": "ragbench_covidqa",
        "hf_path": "galileo-ai/ragbench",
        "hf_config": "covidqa",
        "split": "train",
        "dataset_link": "https://huggingface.co/datasets/galileo-ai/ragbench",
        "description": "RAGBench covidqa subset for RAG response evaluation.",
    },

    # ------------------------------------------------------------
    # Natural Questions pair
    # ------------------------------------------------------------
    # Question-answer pair dataset.
    # Useful for testing:
    # - factoid_qa
    # - short_text answers
    # - keyword / LLM correctness evaluation
    "natural_questions_pair": {
        "dataset_key": "natural_questions_pair",
        "hf_path": "sentence-transformers/natural-questions",
        "hf_config": "pair",
        "split": "train",
        "dataset_link": "https://huggingface.co/datasets/sentence-transformers/natural-questions",
        "description": "Natural Questions reformatted as question-answer pairs.",
    },

    # ------------------------------------------------------------
    # FEVER
    # ------------------------------------------------------------
    # Fact verification dataset.
    # Useful for testing:
    # - claim verification
    # - boolean / label evaluation
    # - factual correctness
    "fever": {
        "dataset_key": "fever",
        "hf_path": "fever/fever",
        "hf_config": None,
        "split": "train",
        "dataset_link": "https://huggingface.co/datasets/fever/fever",
        "description": "FEVER fact verification dataset.",
    },

    # ------------------------------------------------------------
    # AI2 ARC Challenge
    # ------------------------------------------------------------
    # Multiple-choice science QA dataset.
    # Useful for testing:
    # - multiple_choice
    # - rule_based_evaluator
    # - reasoning QA
    "ai2_arc_challenge": {
        "dataset_key": "ai2_arc_challenge",
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "train",
        "dataset_link": "https://huggingface.co/datasets/allenai/ai2_arc",
        "description": "AI2 ARC Challenge multiple-choice science QA dataset.",
    },

    # ------------------------------------------------------------
    # StrategyQA
    # ------------------------------------------------------------
    # Boolean reasoning dataset.
    # Useful for testing:
    # - boolean_reasoning
    # - commonsense reasoning
    # - yes/no evaluation
    "strategyqa": {
        "dataset_key": "strategyqa",
        "hf_path": "ChilleD/StrategyQA",
        "hf_config": None,
        "split": "train",
        "dataset_link": "https://huggingface.co/datasets/ChilleD/StrategyQA",
        "description": "StrategyQA boolean commonsense reasoning dataset.",
    },
}


def get_hf_dataset_config(dataset_key: str) -> dict:
    """
    Get one dataset config from the registry.

    Parameters:
        dataset_key:
            Internal dataset name, for example:
            - hotpotqa
            - ragbench_covidqa
            - natural_questions_pair
            - fever
            - ai2_arc_challenge
            - strategyqa

    Returns:
        A dataset config dictionary.

    Raises:
        ValueError if the dataset_key is not registered.
    """

    if dataset_key not in HF_DATASET_REGISTRY:
        available = list(HF_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset_key: {dataset_key}. "
            f"Available dataset keys: {available}"
        )

    return HF_DATASET_REGISTRY[dataset_key]


def list_available_hf_datasets() -> list[str]:
    """
    Return all available dataset keys.

    This is useful for command line testing.
    """

    return list(HF_DATASET_REGISTRY.keys())