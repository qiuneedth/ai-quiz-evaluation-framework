# src/core/dataset_analyzer.py

"""
Dataset Analyzer.

This module is the first major layer of the framework.

Input:
- dataset name
- dataset samples
- optional dataset description
- optional dataset link/source/split

Output:
- standardized Dataset Profile

This module corresponds to the supervisor's idea:

Dataset
    ↓
Feature Extraction
    ↓
Dataset Metadata
    ↓
Dataset Profile
    ↓
Evaluation Manager
"""


import json

from src.core.profile_schema import make_dataset_profile


def detect_basic_features(samples: list[dict]) -> dict:
    """
    Extract basic dataset features using deterministic Python rules.

    These features do not require an LLM.

    The goal of this function is to detect structural information:
    - What fields exist in the dataset?
    - Does the dataset contain context?
    - Does the dataset contain answer options?
    - Does the dataset contain reference answers?
    - Does the dataset contain generated responses?
    """

    if not samples:
        return {
            "num_preview_samples": 0,
            "fields": [],
            "has_context": False,
            "has_options": False,
            "has_reference_answer": False,
            "has_generated_response": False,
        }

    # Collect all field names from preview samples.
    # Using all samples is safer than using only samples[0],
    # because some datasets may have missing fields in some rows.
    all_fields = set()
    for row in samples:
        all_fields.update(row.keys())

    fields = sorted(list(all_fields))

    # Common field names that usually contain context or documents.
    context_fields = [
        "context",
        "contexts",
        "document",
        "documents",
        "passage",
        "passages",
        "paragraph",
        "paragraphs",
        "retrieved_context",
        "retrieved_contexts",
        "retrieved_documents",
        "evidence",
        "supporting_facts",
    ]

    # Common field names that usually contain multiple-choice options.
    option_fields = [
        "options",
        "choices",
        "candidates",
        "choice",
    ]

    # Common field names that usually contain gold/reference answers.
    answer_fields = [
        "answer",
        "answers",
        "label",
        "target",
        "reference_answer",
        "gold_answer",
        "final_answer",
    ]

    # Common field names that usually contain model-generated responses.
    generated_response_fields = [
        "model_answer",
        "generated_answer",
        "prediction",
        "response",
        "generated_response",
    ]

    # Check whether any preview sample contains non-empty context.
    has_context = any(
        any(field in row and row[field] not in [None, "", []] for field in context_fields)
        for row in samples
    )

    # Check whether any preview sample contains options.
    has_options = any(
        any(field in row and row[field] not in [None, "", []] for field in option_fields)
        for row in samples
    )

    # Check whether any preview sample contains reference answers.
    has_reference_answer = any(
        any(field in row and row[field] not in [None, "", []] for field in answer_fields)
        for row in samples
    )

    # Check whether any preview sample contains generated model responses.
    has_generated_response = any(
        any(field in row and row[field] not in [None, "", []] for field in generated_response_fields)
        for row in samples
    )

    return {
        "num_preview_samples": len(samples),
        "fields": fields,
        "has_context": has_context,
        "has_options": has_options,
        "has_reference_answer": has_reference_answer,
        "has_generated_response": has_generated_response,
    }


def infer_question_type_from_rules(samples: list[dict], basic_features: dict) -> str:
    """
    Infer question type using simple deterministic rules.

    This function provides a first-pass classification before using an LLM.

    The LLM can still improve this result later, but this gives the framework
    a stable fallback when LLM output is unavailable or invalid.
    """

    fields = basic_features.get("fields", [])
    has_context = basic_features.get("has_context", False)
    has_options = basic_features.get("has_options", False)

    # Multiple-choice datasets usually contain choices/options.
    if has_options:
        return "multiple_choice"

    # Context QA datasets contain context and expected answers.
    if has_context:
        return "context_grounded_qa"

    # Try to detect boolean reasoning by answer values.
    possible_answers = []

    for row in samples:
        for key in ["answer", "answers", "label", "target", "reference_answer"]:
            if key in row:
                possible_answers.append(str(row[key]).lower())

    boolean_values = {"true", "false", "yes", "no", "0", "1"}

    if possible_answers:
        boolean_like_count = sum(
            1 for answer in possible_answers
            if answer.strip() in boolean_values
        )

        if boolean_like_count / len(possible_answers) >= 0.7:
            return "boolean_reasoning"

    # If the dataset has question and answer fields but no context,
    # it is likely a factoid QA dataset.
    if "question" in fields and any(field in fields for field in ["answer", "answers", "reference_answer"]):
        return "factoid_qa"

    return "unknown"


def infer_answer_type_from_rules(samples: list[dict], basic_features: dict) -> str:
    """
    Infer answer type using deterministic rules.

    Answer type describes the expected format of the answer.
    """

    if basic_features.get("has_options", False):
        return "multiple_choice_option"

    possible_answers = []

    for row in samples:
        for key in ["answer", "answers", "label", "target", "reference_answer"]:
            if key in row:
                possible_answers.append(row[key])

    if not possible_answers:
        return "unknown"

    # If answers are boolean-like, classify them as boolean.
    boolean_values = {"true", "false", "yes", "no", "0", "1"}

    str_answers = [str(answer).lower().strip() for answer in possible_answers]

    if all(answer in boolean_values for answer in str_answers):
        return "boolean"

    # If answers are lists, classify them as list.
    if any(isinstance(answer, list) for answer in possible_answers):
        return "list"

    # Estimate text length.
    avg_answer_length = sum(len(str(answer).split()) for answer in possible_answers) / len(possible_answers)

    if avg_answer_length <= 5:
        return "short_text"

    if avg_answer_length <= 30:
        return "long_text"

    return "free_form_text"


def infer_context_dependency(basic_features: dict) -> str:
    """
    Infer whether context is required.

    If the dataset contains context fields, then the context is likely required
    for evaluation. If not, context dependency is none.
    """

    if basic_features.get("has_context", False):
        return "provided_context_required"

    return "none"


def build_llm_prompt(
    dataset_name: str,
    samples: list[dict],
    basic_features: dict,
    rule_based_guess: dict,
    dataset_description: str | None = None,
) -> str:
    """
    Build prompt for LLM-based dataset analysis.

    The LLM is used to infer higher-level metadata that is difficult to detect
    with rules only, such as:
    - dataset category
    - domain
    - topic
    - reasoning type
    - difficulty level
    - expected answer format
    - recommended evaluators
    - recommended metrics

    Important:
    The prompt requires the LLM to return only valid JSON.
    """

    sample_text = json.dumps(samples[:5], ensure_ascii=False, indent=2)
    feature_text = json.dumps(basic_features, ensure_ascii=False, indent=2)
    guess_text = json.dumps(rule_based_guess, ensure_ascii=False, indent=2)

    return f"""
You are a dataset analysis assistant for an AI-Based Adaptive Quiz Evaluation Framework.

Your task is to analyze a dataset and generate evaluation-oriented metadata.

The Dataset Profile is the heart of this framework.
It will be used to automatically select evaluators and generate evaluation plans.

Dataset name:
{dataset_name}

Dataset description:
{dataset_description or "No description provided."}

Basic features detected by code:
{feature_text}

Rule-based initial guess:
{guess_text}

Preview samples:
{sample_text}

Use ONLY the following controlled vocabularies.

question_type:
- multiple_choice
- boolean_reasoning
- factoid_qa
- extractive_qa
- abstractive_qa
- context_grounded_qa
- multi_hop_reasoning
- open_ended_reasoning
- rag_response_evaluation
- unknown

answer_type:
- single_label
- boolean
- span
- short_text
- long_text
- free_form_text
- multiple_choice_option
- generated_response
- list
- unknown

reasoning_type:
- none
- factual_recall
- single_hop_reasoning
- multi_hop_reasoning
- commonsense_reasoning
- logical_reasoning
- contextual_reasoning
- open_ended_reasoning
- unknown

difficulty_level:
- easy
- medium
- hard
- mixed
- unknown

context_dependency:
- none
- provided_context_required
- retrieved_context_required
- optional_context
- unknown

evaluation_candidates:
- rule_based_evaluator
- keyword_evaluator
- llm_evaluator
- llm_context_evaluator
- relevance_evaluator
- completeness_evaluator
- groundedness_evaluator

possible_metrics:
- exact_match
- keyword_overlap
- correctness
- relevance
- completeness
- groundedness
- faithfulness
- context_usage
- reasoning_quality
- clarity

Return only valid JSON with this structure:

{{
  "dataset_category": "...",
  "domain": "...",
  "topic": "...",
  "question_type": "...",
  "answer_type": "...",
  "reasoning_type": "...",
  "difficulty_level": "...",
  "context_dependency": "...",
  "expected_answer_format": "...",
  "evaluation_candidates": [],
  "possible_metrics": [],
  "summary": "...",
  "notes": "..."
}}
"""


def analyze_dataset_with_llm(
    dataset_name: str,
    samples: list[dict],
    llm_client,
    dataset_description: str | None = None,
) -> dict:
    """
    Analyze a dataset with both rule-based features and LLM inference.

    This function returns raw metadata from the LLM.
    It does not yet build the final Dataset Profile.

    Steps:
    1. Extract basic structural features using Python rules.
    2. Infer basic question and answer type using rules.
    3. Ask the LLM to infer higher-level metadata.
    4. Parse the LLM JSON output.
    5. Return metadata.
    """

    basic_features = detect_basic_features(samples)

    rule_based_guess = {
        "question_type": infer_question_type_from_rules(samples, basic_features),
        "answer_type": infer_answer_type_from_rules(samples, basic_features),
        "context_dependency": infer_context_dependency(basic_features),
    }

    prompt = build_llm_prompt(
        dataset_name=dataset_name,
        samples=samples,
        basic_features=basic_features,
        rule_based_guess=rule_based_guess,
        dataset_description=dataset_description,
    )

    raw = llm_client.chat(
        system_prompt="Return only valid JSON.",
        user_prompt=prompt,
    )

    try:
        metadata = json.loads(raw)

    except json.JSONDecodeError:
        metadata = {
            "dataset_name": dataset_name,
            "json_error": True,
            "raw_output": raw,
            "basic_features": basic_features,
            "rule_based_guess": rule_based_guess,
        }

    return metadata


def analyze_dataset_to_profile(
    dataset_name: str,
    samples: list[dict],
    llm_client=None,
    dataset_description: str | None = None,
    dataset_link: str | None = None,
    source: str = "unknown",
    split: str = "unknown",
    location: str | None = None,
) -> dict:
    """
    Main function of Dataset Analyzer.

    This function generates the final standardized Dataset Profile.

    It can work in two modes:

    Mode 1: with LLM client
        - Use rules for basic feature extraction.
        - Use LLM for semantic metadata.
        - Merge both into Dataset Profile.

    Mode 2: without LLM client
        - Use rule-based inference only.
        - Still return a valid Dataset Profile.

    This makes the framework more stable because the whole system will not fail
    when the LLM is unavailable.
    """

    # Step 1: Extract basic structural features.
    basic_features = detect_basic_features(samples)

    # Step 2: Create rule-based fallback values.
    rule_question_type = infer_question_type_from_rules(samples, basic_features)
    rule_answer_type = infer_answer_type_from_rules(samples, basic_features)
    rule_context_dependency = infer_context_dependency(basic_features)

    rule_based_metadata = {
        "dataset_category": "quiz_qa",
        "domain": "general",
        "topic": "unknown",
        "question_type": rule_question_type,
        "answer_type": rule_answer_type,
        "reasoning_type": "unknown",
        "difficulty_level": "unknown",
        "context_dependency": rule_context_dependency,
        "expected_answer_format": rule_answer_type,
        "evaluation_candidates": [],
        "possible_metrics": [],
        "summary": "",
        "notes": "Generated by rule-based dataset analyzer.",
    }

    # Step 3: If an LLM client is provided, ask the LLM for richer metadata.
    if llm_client is not None:
        llm_metadata = analyze_dataset_with_llm(
            dataset_name=dataset_name,
            samples=samples,
            llm_client=llm_client,
            dataset_description=dataset_description,
        )

        # If the LLM returned invalid JSON, keep rule-based metadata.
        if not llm_metadata.get("json_error", False):
            merged_metadata = {
                **rule_based_metadata,
                **llm_metadata,
            }
        else:
            merged_metadata = rule_based_metadata
            merged_metadata["notes"] += " LLM metadata failed to parse."
    else:
        merged_metadata = rule_based_metadata

    # Step 4: Build standardized Dataset Profile.
    profile = make_dataset_profile(
        dataset_name=dataset_name,
        dataset_link=dataset_link,
        dataset_category=merged_metadata.get("dataset_category", "unknown"),
        source=source,
        split=split,
        sample_count=len(samples),
        fields=basic_features.get("fields", []),
        has_context=basic_features.get("has_context", False),
        has_reference_answer=basic_features.get("has_reference_answer", False),
        has_generated_response=basic_features.get("has_generated_response", False),
        has_options=basic_features.get("has_options", False),
        domain=merged_metadata.get("domain", "general"),
        topic=merged_metadata.get("topic", "unknown"),
        location=location,
        question_type=merged_metadata.get("question_type", rule_question_type),
        answer_type=merged_metadata.get("answer_type", rule_answer_type),
        reasoning_type=merged_metadata.get("reasoning_type", "unknown"),
        difficulty_level=merged_metadata.get("difficulty_level", "unknown"),
        context_dependency=merged_metadata.get("context_dependency", rule_context_dependency),
        expected_answer_format=merged_metadata.get("expected_answer_format", rule_answer_type),
        evaluation_candidates=merged_metadata.get("evaluation_candidates", []),
        possible_metrics=merged_metadata.get("possible_metrics", []),
        sample_questions=samples[:10],
        notes=merged_metadata.get("notes", ""),
    )

    return profile