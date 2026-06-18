# src/data/hf_adapters.py

"""
Hugging Face Dataset Adapters.

Different Hugging Face datasets have different schemas.

This module converts raw HF records into the standard sample format used
by this framework.

Standard sample format:

{
    "question_id": "...",
    "question": "...",
    "context": "...",
    "reference_answer": "...",
    "choices": [...],
    "model_answer": "...",
    "metadata": {...}
}

Not every dataset has every field.

For example:
- ARC has choices but no context.
- HotpotQA has context and reference answer.
- RAGBench has context documents and generated response.
- StrategyQA has boolean answer.
"""


def adapt_hf_records(dataset_key: str, raw_records: list[dict]) -> list[dict]:
    """
    Convert a list of raw Hugging Face records into standard samples.

    Parameters:
        dataset_key:
            Internal dataset key.

        raw_records:
            Raw rows loaded from Hugging Face.

    Returns:
        List of standardized sample dictionaries.
    """

    adapted_records = []

    for index, record in enumerate(raw_records):
        adapted = adapt_one_hf_record(
            dataset_key=dataset_key,
            record=record,
            index=index,
        )
        adapted_records.append(adapted)

    return adapted_records


def adapt_one_hf_record(dataset_key: str, record: dict, index: int) -> dict:
    """
    Adapt one Hugging Face record according to dataset type.

    If a dataset-specific adapter exists, use it.
    Otherwise, use the generic adapter.
    """

    if dataset_key == "hotpotqa":
        return adapt_hotpotqa(record, index)

    if dataset_key == "ragbench_covidqa":
        return adapt_ragbench(record, index)

    if dataset_key == "natural_questions_pair":
        return adapt_natural_questions_pair(record, index)

    if dataset_key == "fever":
        return adapt_fever(record, index)

    if dataset_key == "ai2_arc_challenge":
        return adapt_ai2_arc(record, index)

    if dataset_key == "strategyqa":
        return adapt_strategyqa(record, index)

    return adapt_generic(record, index)


# ============================================================
# Dataset-specific adapters
# ============================================================

def adapt_hotpotqa(record: dict, index: int) -> dict:
    """
    Adapter for HotpotQA.

    Expected useful fields:
    - id
    - question
    - answer
    - context
    - supporting_facts

    HotpotQA context is usually stored as:
    {
        "title": [...],
        "sentences": [[...], [...]]
    }

    This adapter converts context into readable text.
    """

    question_id = str(record.get("id", f"hotpotqa_{index}"))

    question = record.get("question", "")

    reference_answer = record.get("answer", "")

    context = convert_hotpot_context_to_text(record.get("context"))

    return {
        "question_id": question_id,
        "question": question,
        "context": context,
        "reference_answer": reference_answer,
        "choices": None,
        "model_answer": None,
        "metadata": {
            "source_dataset": "hotpotqa",
            "supporting_facts": record.get("supporting_facts"),
            "raw_fields": list(record.keys()),
        },
    }


def adapt_ragbench(record: dict, index: int) -> dict:
    """
    Adapter for RAGBench covidqa.

    RAGBench is useful for evaluating generated RAG responses.

    Expected useful fields may include:
    - id
    - question
    - documents
    - response
    - relevance_score
    - utilization_score
    - completeness_score
    - adherence_score
    - trulens_groundedness
    - ragas_faithfulness

    This adapter treats:
    - response as model_answer
    - documents as context
    """

    question_id = str(record.get("id", f"ragbench_covidqa_{index}"))

    question = record.get("question", "")

    context = convert_any_context_to_text(
        record.get("documents")
        or record.get("documents_sentences")
        or record.get("context")
    )

    model_answer = (
        record.get("response")
        or record.get("generated_answer")
        or record.get("model_answer")
    )

    reference_answer = extract_reference_answer_from_ragbench(record)

    return {
        "question_id": question_id,
        "question": question,
        "context": context,
        "reference_answer": reference_answer,
        "choices": None,
        "model_answer": model_answer,
        "metadata": {
            "source_dataset": "ragbench_covidqa",
            "dataset_name": record.get("dataset_name"),
            "adherence_score": record.get("adherence_score"),
            "relevance_score": record.get("relevance_score"),
            "utilization_score": record.get("utilization_score"),
            "completeness_score": record.get("completeness_score"),
            "trulens_groundedness": record.get("trulens_groundedness"),
            "ragas_faithfulness": record.get("ragas_faithfulness"),
            "raw_fields": list(record.keys()),
        },
    }


def adapt_natural_questions_pair(record: dict, index: int) -> dict:
    """
    Adapter for sentence-transformers/natural-questions pair subset.

    This dataset is a question-answer pair dataset.

    Possible field names may include:
    - question
    - query
    - anchor
    - answer
    - positive
    - text
    """

    question_id = str(record.get("id", f"natural_questions_pair_{index}"))

    question = first_non_empty_value(
        record,
        ["question", "query", "anchor", "sentence1"],
    )

    reference_answer = first_non_empty_value(
        record,
        ["answer", "positive", "text", "sentence2"],
    )

    return {
        "question_id": question_id,
        "question": question,
        "context": None,
        "reference_answer": reference_answer,
        "choices": None,
        "model_answer": None,
        "metadata": {
            "source_dataset": "natural_questions_pair",
            "raw_fields": list(record.keys()),
        },
    }


def adapt_fever(record: dict, index: int) -> dict:
    """
    Adapter for FEVER.

    FEVER is a fact verification dataset.

    The main input is usually a claim.
    The label is usually the reference answer.

    This adapter converts:
    - claim -> question
    - label -> reference_answer
    """

    question_id = str(record.get("id", f"fever_{index}"))

    claim = first_non_empty_value(
        record,
        ["claim", "question", "text"],
    )

    label = first_non_empty_value(
        record,
        ["label", "answer", "verdict"],
    )

    evidence_text = convert_any_context_to_text(
        record.get("evidence")
        or record.get("context")
        or record.get("documents")
    )

    return {
        "question_id": question_id,
        "question": claim,
        "context": evidence_text,
        "reference_answer": label,
        "choices": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        "model_answer": None,
        "metadata": {
            "source_dataset": "fever",
            "raw_fields": list(record.keys()),
        },
    }


def adapt_ai2_arc(record: dict, index: int) -> dict:
    """
    Adapter for AI2 ARC Challenge.

    ARC is a multiple-choice science QA dataset.

    Expected useful fields:
    - id
    - question
    - choices
    - answerKey
    """

    question_id = str(record.get("id", f"ai2_arc_challenge_{index}"))

    question = record.get("question", "")

    choices = convert_arc_choices(record.get("choices"))

    reference_answer = record.get("answerKey", "")

    return {
        "question_id": question_id,
        "question": question,
        "context": None,
        "reference_answer": reference_answer,
        "choices": choices,
        "model_answer": None,
        "metadata": {
            "source_dataset": "ai2_arc_challenge",
            "raw_choices": record.get("choices"),
            "raw_fields": list(record.keys()),
        },
    }


def adapt_strategyqa(record: dict, index: int) -> dict:
    """
    Adapter for StrategyQA.

    StrategyQA is usually boolean reasoning.

    Expected useful fields may include:
    - question
    - answer
    - facts
    - decomposition

    This adapter converts boolean answers into yes/no style strings.
    """

    question_id = str(record.get("id", f"strategyqa_{index}"))

    question = record.get("question", "")

    raw_answer = record.get("answer")

    if isinstance(raw_answer, bool):
        reference_answer = "yes" if raw_answer else "no"
    else:
        reference_answer = str(raw_answer)

    context = convert_any_context_to_text(
        record.get("facts")
        or record.get("decomposition")
        or record.get("context")
    )

    return {
        "question_id": question_id,
        "question": question,
        "context": context,
        "reference_answer": reference_answer,
        "choices": ["yes", "no"],
        "model_answer": None,
        "metadata": {
            "source_dataset": "strategyqa",
            "raw_answer": raw_answer,
            "raw_fields": list(record.keys()),
        },
    }


def adapt_generic(record: dict, index: int) -> dict:
    """
    Generic fallback adapter.

    This is used when no dataset-specific adapter exists.

    It tries to find common fields:
    - question
    - context
    - answer
    - choices
    """

    question_id = str(record.get("id", f"generic_{index}"))

    question = first_non_empty_value(
        record,
        ["question", "query", "claim", "prompt", "input"],
    )

    context = first_non_empty_value(
        record,
        ["context", "document", "documents", "passage", "evidence"],
    )

    reference_answer = first_non_empty_value(
        record,
        ["answer", "answers", "label", "target", "reference_answer"],
    )

    choices = first_non_empty_value(
        record,
        ["choices", "options", "candidates"],
    )

    model_answer = first_non_empty_value(
        record,
        ["model_answer", "generated_answer", "response", "prediction"],
    )

    return {
        "question_id": question_id,
        "question": question,
        "context": convert_any_context_to_text(context),
        "reference_answer": reference_answer,
        "choices": choices,
        "model_answer": model_answer,
        "metadata": {
            "source_dataset": "generic",
            "raw_fields": list(record.keys()),
        },
    }


# ============================================================
# Helper functions
# ============================================================

def first_non_empty_value(record: dict, keys: list[str]):
    """
    Return the first non-empty value from a dictionary.

    This helps handle datasets with different field names.
    """

    for key in keys:
        value = record.get(key)

        if value not in [None, "", []]:
            return value

    return None


def convert_hotpot_context_to_text(context) -> str | None:
    """
    Convert HotpotQA context into readable text.

    HotpotQA context often looks like:
    {
        "title": ["Title A", "Title B"],
        "sentences": [["sentence1", "sentence2"], ["sentence3"]]
    }

    This function converts it into:
    Title A: sentence1 sentence2
    Title B: sentence3
    """

    if context in [None, "", []]:
        return None

    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])

        parts = []

        for title, sentence_list in zip(titles, sentences):
            sentence_text = " ".join(str(sentence) for sentence in sentence_list)
            parts.append(f"{title}: {sentence_text}")

        return "\n".join(parts)

    return convert_any_context_to_text(context)


def convert_arc_choices(choices) -> list[dict] | None:
    """
    Convert ARC choices into a standard list.

    ARC choices usually look like:
    {
        "text": [...],
        "label": [...]
    }

    Output:
    [
        {"label": "A", "text": "..."},
        {"label": "B", "text": "..."}
    ]
    """

    if choices in [None, "", []]:
        return None

    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])

        converted = []

        for label, text in zip(labels, texts):
            converted.append(
                {
                    "label": label,
                    "text": text,
                }
            )

        return converted

    if isinstance(choices, list):
        return choices

    return None


def extract_reference_answer_from_ragbench(record: dict):
    """
    Extract reference answer from RAGBench if available.

    RAGBench is often designed for evaluating generated responses.
    Some rows may not have a traditional gold answer.

    If no reference answer exists, return None.
    """

    possible_answer = first_non_empty_value(
        record,
        [
            "answer",
            "answers",
            "reference_answer",
            "gold_answer",
            "ground_truth",
        ],
    )

    if possible_answer not in [None, "", []]:
        return possible_answer

    # Some RAGBench rows contain response-level labels instead of gold answer.
    # We keep reference_answer as None and let the evaluation plan use
    # groundedness / faithfulness / relevance instead.
    return None


def convert_any_context_to_text(context) -> str | None:
    """
    Convert different context formats into readable text.

    Supported input:
    - string
    - list of strings
    - list of dictionaries/lists
    - dictionary
    """

    if context in [None, "", []]:
        return None

    if isinstance(context, str):
        return context

    if isinstance(context, list):
        parts = []

        for item in context:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                parts.append(convert_dict_to_text(item))

            elif isinstance(item, list):
                parts.append(convert_any_context_to_text(item) or "")

            else:
                parts.append(str(item))

        return "\n".join(part for part in parts if part)

    if isinstance(context, dict):
        return convert_dict_to_text(context)

    return str(context)


def convert_dict_to_text(value: dict) -> str:
    """
    Convert a dictionary into simple readable text.

    This is used for nested context fields.
    """

    parts = []

    for key, item in value.items():
        parts.append(f"{key}: {item}")

    return "\n".join(parts)