# src/evaluation/run_prompt_sensitivity_analysis.py

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from src.evaluation.prompt_sensitivity_prompts import (
    PROMPT_VARIANTS,
    TEST_CASES,
)


DEFAULT_OUTPUT_PATH = "data/results/prompt_sensitivity_results.json"


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def get_client():
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install openai python-dotenv"
        )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Please set it in your environment or .env file."
        )

    return OpenAI(api_key=api_key)


def call_llm(
    client: Any,
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful educational answer evaluator. "
                    "Return only valid JSON and do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


def extract_json(text: str) -> tuple[dict[str, Any], bool]:
    text = text.strip()

    try:
        return json.loads(text), True
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return {}, False

    try:
        return json.loads(match.group(0)), True
    except Exception:
        return {}, False


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == "true":
            return True
        if lower == "false":
            return False

    return None


def evaluate_case_with_prompt(
    client: Any,
    model: str,
    prompt_name: str,
    prompt_template: str,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    prompt = prompt_template.format(
        question=test_case["question"],
        reference_answer=test_case["reference_answer"],
        student_answer=test_case["student_answer"],
    )

    try:
        raw_output = call_llm(
            client=client,
            model=model,
            prompt=prompt,
            temperature=0.0,
        )

        parsed_output, json_valid = extract_json(raw_output)

        score = safe_float(parsed_output.get("score"))
        passed = safe_bool(parsed_output.get("passed"))

        return {
            "prompt_name": prompt_name,
            "json_valid": json_valid,
            "score": score,
            "passed": passed,
            "feedback": parsed_output.get("feedback"),
            "parsed_output": parsed_output,
            "raw_output": raw_output,
            "error": None,
        }

    except Exception as error:
        return {
            "prompt_name": prompt_name,
            "json_valid": False,
            "score": None,
            "passed": None,
            "feedback": None,
            "parsed_output": {},
            "raw_output": "",
            "error": str(error),
        }


def compute_case_metrics(
    prompt_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scores = [
        result["score"]
        for result in prompt_results.values()
        if result.get("score") is not None
    ]

    passed_values = [
        result["passed"]
        for result in prompt_results.values()
        if result.get("passed") is not None
    ]

    json_valid_values = [
        bool(result.get("json_valid"))
        for result in prompt_results.values()
    ]

    if scores:
        score_range = max(scores) - min(scores)
        avg_score = mean(scores)
    else:
        score_range = None
        avg_score = None

    pass_fail_consistent = (
        len(set(passed_values)) == 1
        if passed_values
        else False
    )

    return {
        "json_valid_all_prompts": all(json_valid_values),
        "json_valid_count": sum(1 for value in json_valid_values if value),
        "score_range": round(score_range, 4) if score_range is not None else None,
        "avg_score": round(avg_score, 4) if avg_score is not None else None,
        "pass_fail_consistent": pass_fail_consistent,
    }


def compute_overall_summary(
    test_case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total_cases = len(test_case_results)
    total_prompts = len(PROMPT_VARIANTS)

    if total_cases == 0:
        return {}

    all_evaluations = []

    for case in test_case_results:
        for result in case["prompt_results"].values():
            all_evaluations.append(result)

    json_valid_count = sum(
        1 for result in all_evaluations
        if result.get("json_valid")
    )

    pass_consistent_count = sum(
        1 for case in test_case_results
        if case["metrics"].get("pass_fail_consistent")
    )

    score_ranges = [
        case["metrics"]["score_range"]
        for case in test_case_results
        if case["metrics"].get("score_range") is not None
    ]

    max_score_range = max(score_ranges) if score_ranges else None

    highest_variation_cases = [
        {
            "id": case["id"],
            "type": case["type"],
            "score_range": case["metrics"]["score_range"],
        }
        for case in test_case_results
        if case["metrics"].get("score_range") == max_score_range
    ]

    return {
        "total_test_cases": total_cases,
        "prompt_variants": list(PROMPT_VARIANTS.keys()),
        "total_evaluations": total_cases * total_prompts,
        "json_valid_rate": round(json_valid_count / len(all_evaluations), 4),
        "pass_fail_consistency_rate": round(pass_consistent_count / total_cases, 4),
        "max_score_range": max_score_range,
        "highest_variation_cases": highest_variation_cases,
        "main_observation": (
            "Prompt sensitivity is expected to be most visible in borderline cases, "
            "such as partial answers or answers with extra information. Clear correct "
            "and clear wrong answers are expected to be more stable across prompts."
        ),
    }


def run_prompt_sensitivity_analysis(model: str) -> dict[str, Any]:
    load_environment()
    client = get_client()

    results: dict[str, Any] = {
        "experiment": "prompt_sensitivity_analysis",
        "purpose": (
            "Evaluate whether different LLM evaluator prompt formulations produce "
            "different scores, pass/fail decisions, and feedback for the same fixed "
            "short-answer cases."
        ),
        "model": model,
        "temperature": 0.0,
        "prompt_variants": {
            name: {
                "style": config["style"],
                "score_type": config["score_type"],
                "purpose": config["purpose"],
            }
            for name, config in PROMPT_VARIANTS.items()
        },
        "test_cases": [],
        "summary": {},
    }

    for test_case in TEST_CASES:
        print(f"\nRunning {test_case['id']} - {test_case['type']}")

        prompt_results: dict[str, dict[str, Any]] = {}

        for prompt_name, prompt_config in PROMPT_VARIANTS.items():
            result = evaluate_case_with_prompt(
                client=client,
                model=model,
                prompt_name=prompt_name,
                prompt_template=prompt_config["prompt"],
                test_case=test_case,
            )

            prompt_results[prompt_name] = result

            print(
                f"  {prompt_name}: "
                f"score={result['score']}, "
                f"passed={result['passed']}, "
                f"json_valid={result['json_valid']}, "
                f"error={result['error']}"
            )

        metrics = compute_case_metrics(prompt_results)

        results["test_cases"].append(
            {
                "id": test_case["id"],
                "type": test_case["type"],
                "question": test_case["question"],
                "reference_answer": test_case["reference_answer"],
                "student_answer": test_case["student_answer"],
                "expected_behavior": test_case["expected_behavior"],
                "prompt_results": prompt_results,
                "metrics": metrics,
            }
        )

    results["summary"] = compute_overall_summary(results["test_cases"])

    return results


def save_json(data: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def print_table(results: dict[str, Any]) -> None:
    print("\nPrompt sensitivity result table:")
    print(
        f"{'TC':<6}"
        f"{'Type':<34}"
        f"{'Basic':>10}"
        f"{'Strict':>10}"
        f"{'Rubric':>10}"
        f"{'Range':>10}"
        f"{'PassCons':>12}"
    )
    print("-" * 92)

    for case in results["test_cases"]:
        prompt_results = case["prompt_results"]

        basic_score = prompt_results.get("basic_lenient", {}).get("score")
        strict_score = prompt_results.get("strict", {}).get("score")
        rubric_score = prompt_results.get("rubric_based", {}).get("score")

        score_range = case["metrics"].get("score_range")
        pass_consistent = case["metrics"].get("pass_fail_consistent")

        print(
            f"{case['id']:<6}"
            f"{case['type']:<34}"
            f"{str(basic_score):>10}"
            f"{str(strict_score):>10}"
            f"{str(rubric_score):>10}"
            f"{str(score_range):>10}"
            f"{str(pass_consistent):>12}"
        )

    print("\nOverall summary:")
    for key, value in results["summary"].items():
        print(f"- {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        help="Model used for LLM prompt sensitivity analysis.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file path.",
    )

    args = parser.parse_args()

    results = run_prompt_sensitivity_analysis(model=args.model)
    save_json(results, args.output)
    print_table(results)

    print("\nSaved prompt sensitivity results to:")
    print(args.output)


if __name__ == "__main__":
    main()