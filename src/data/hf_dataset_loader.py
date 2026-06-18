# src/data/hf_dataset_loader.py

"""
Hugging Face Dataset Loader.

This module loads real datasets from Hugging Face using datasets.load_dataset.

It does not analyze the dataset.
It only loads raw rows and sends them to hf_adapters.py for normalization.

Workflow:

dataset_key
    ↓
hf_dataset_registry.py
    ↓
datasets.load_dataset()
    ↓
raw Hugging Face rows
    ↓
hf_adapters.py
    ↓
standard samples
"""


from datasets import load_dataset

from src.data.hf_dataset_registry import get_hf_dataset_config
from src.data.hf_adapters import adapt_hf_records


def load_hf_dataset_samples(
    dataset_key: str,
    sample_size: int = 5,
) -> dict:
    """
    Load a small number of samples from a Hugging Face dataset.

    Parameters:
        dataset_key:
            Internal dataset key defined in hf_dataset_registry.py.

        sample_size:
            Number of examples to load.
            Use a small number during testing to avoid slow loading.

    Returns:
        A dictionary containing:
        - dataset_config
        - raw_samples
        - adapted_samples

    Why adapted_samples are needed:
        Different HF datasets use different field names.

        Example:
        - HotpotQA has question, answer, context
        - ARC has question, choices, answerKey
        - RAGBench has question, documents, response
        - StrategyQA has question, answer

        Our framework needs one standard format:
        {
            "question_id": "...",
            "question": "...",
            "context": "...",
            "reference_answer": "...",
            "choices": [...],
            "model_answer": "..."
        }
    """

    dataset_config = get_hf_dataset_config(dataset_key)

    hf_path = dataset_config["hf_path"]
    hf_config = dataset_config.get("hf_config")
    split = dataset_config.get("split", "train")

    # Load only a slice of the split.
    # Example: train[:5]
    split_slice = f"{split}[:{sample_size}]"

    try:
        if hf_config is None:
            dataset = load_dataset(
                hf_path,
                split=split_slice,
            )
        else:
            dataset = load_dataset(
                hf_path,
                hf_config,
                split=split_slice,
            )

    except Exception as first_error:
        # Some datasets may not support train split or may have config changes.
        # This fallback makes debugging easier instead of failing silently.
        raise RuntimeError(
            f"Failed to load Hugging Face dataset.\n"
            f"dataset_key={dataset_key}\n"
            f"hf_path={hf_path}\n"
            f"hf_config={hf_config}\n"
            f"split={split_slice}\n"
            f"Original error: {first_error}"
        )

    raw_samples = [dict(row) for row in dataset]

    adapted_samples = adapt_hf_records(
        dataset_key=dataset_key,
        raw_records=raw_samples,
    )

    return {
        "dataset_config": dataset_config,
        "raw_samples": raw_samples,
        "adapted_samples": adapted_samples,
    }