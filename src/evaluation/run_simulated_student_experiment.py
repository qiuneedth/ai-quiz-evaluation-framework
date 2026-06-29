# src/evaluation/run_simulated_student_experiment.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.core.simple_evaluation_engine import SimpleEvaluationEngine
from src.evaluation.simulated_student_agent import (
    STUDENT_PROFILES,
    SimulatedStudentAgent,
)


DEFAULT_OUTPUT_PATH = "data/results/simulated_student_results.json"


def load_questions(path: str | None) -> list[dict[str, Any]]:
    """
    Load multiple-choice questions.

    Expected format:
    [
      {
        "id": "q1",
        "question": "...",
        "choices": {
          "A": "...",
          "B": "...",
          "C": "...",
          "D": "..."
        },
        "answer": "A",
        "question_type": "multiple_choice"
      }
    ]

    If no file is provided, use a small built-in sample.
    """

    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("questions") or data.get("selected_questions") or data

        if not isinstance(data, list):
            raise ValueError("Question file must contain a list of questions.")

        return data

    return [
        {
            "id": "sample_q1",
            "question": "Which material is usually attracted by a magnet?",
            "choices": {
                "A": "Wood",
                "B": "Iron",
                "C": "Paper",
                "D": "Plastic",
            },
            "answer": "B",
            "question_type": "multiple_choice",
        },
        {
            "id": "sample_q2",
            "question": "Which force pulls objects toward Earth?",
            "choices": {
                "A": "Gravity",
                "B": "Friction",
                "C": "Magnetism",
                "D": "Electricity",
            },
            "answer": "A",
            "question_type": "multiple_choice",
        },
        {
            "id": "sample_q3",
            "question": "What do plants need for photosynthesis?",
            "choices": {
                "A": "Sunlight",
                "B": "Plastic",
                "C": "Metal",
                "D": "Sand only",
            },
            "answer": "A",
            "question_type": "multiple_choice",
        },
        {
            "id": "sample_q4",
            "question": "What is water's freezing point at standard pressure?",
            "choices": {
                "A": "0 degrees Celsius",
                "B": "50 degrees Celsius",
                "C": "100 degrees Celsius",
                "D": "200 degrees Celsius",
            },
            "answer": "A",
            "question_type": "multiple_choice",
        },
        {
            "id": "sample_q5",
            "question": "Which object is most likely to float in water?",
            "choices": {
                "A": "A stone",
                "B": "A metal coin",
                "C": "A wooden block",
                "D": "A glass marble",
            },
            "answer": "C",
            "question_type": "multiple_choice",
        },
    ]


def build_hint_text(question: dict[str, Any]) -> str:
    """
    For the simulated experiment, we only need to record hint usage.
    This avoids extra LLM calls and keeps the experiment controlled.
    """

    return "Think about the key concept and compare the answer options carefully."


def evaluate_with_framework(
    engine: SimpleEvaluationEngine,
    question: dict[str, Any],
    user_answer: str,
    hint_used: bool,
    hint_text: str | None,
    round_index: int,
) -> dict[str, Any]:
    """
    Evaluate generated answer using the existing evaluation engine.
    """

    return engine.evaluate_question(
        question=question,
        user_answer=user_answer,
        assigned_evaluators=["script_multiple_choice"],
        answer_source="simulated_student",
        round_index=round_index,
        hint_used=hint_used,
        hint_text=hint_text,
    )


def summarize_run(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(question_results)

    if total == 0:
        return {
            "total_questions": 0,
            "correct_count": 0,
            "hint_used_count": 0,
            "correct_rate": 0.0,
            "hint_rate": 0.0,
            "avg_raw_score": 0.0,
            "avg_final_score": 0.0,
            "avg_final_not_greater_than_raw": True,
            "errors": 0,
        }

    correct_count = sum(1 for r in question_results if r.get("raw_is_correct") or r.get("is_correct"))
    hint_used_count = sum(1 for r in question_results if r.get("hint_used"))

    raw_scores = [float(r.get("raw_score", 0.0)) for r in question_results]
    final_scores = [float(r.get("final_score", 0.0)) for r in question_results]

    avg_raw = mean(raw_scores)
    avg_final = mean(final_scores)

    return {
        "total_questions": total,
        "correct_count": correct_count,
        "hint_used_count": hint_used_count,
        "correct_rate": round(correct_count / total, 4),
        "hint_rate": round(hint_used_count / total, 4),
        "avg_raw_score": round(avg_raw, 4),
        "avg_final_score": round(avg_final, 4),
        "avg_final_not_greater_than_raw": avg_final <= avg_raw + 1e-9,
        "errors": 0,
    }


def run_one_session(
    profile_index: int,
    run_index: int,
    questions: list[dict[str, Any]],
    n_questions: int,
    evaluator_mode: str,
) -> dict[str, Any]:
    profile = STUDENT_PROFILES[profile_index]
    seed = 1000 + profile_index * 100 + run_index

    agent = SimulatedStudentAgent(profile=profile, seed=seed)
    engine = SimpleEvaluationEngine(
        evaluator_mode=evaluator_mode,
        allow_mock_fallback=True,
    )

    selected_questions = questions[:n_questions]
    question_results = []
    errors = []

    for i, question in enumerate(selected_questions, start=1):
        try:
            simulated_answer = agent.answer_multiple_choice(question)

            hint_used = simulated_answer["hint_used"]
            hint_text = build_hint_text(question) if hint_used else None

            evaluated = evaluate_with_framework(
                engine=engine,
                question=question,
                user_answer=simulated_answer["user_answer"],
                hint_used=hint_used,
                hint_text=hint_text,
                round_index=i,
            )

            evaluated["simulated_student"] = {
                "profile": profile.name,
                "correct_probability": profile.correct_probability,
                "hint_probability": profile.hint_probability,
                "intended_correct": simulated_answer["intended_correct"],
            }

            question_results.append(evaluated)

        except Exception as e:
            errors.append(
                {
                    "question_index": i,
                    "question_id": question.get("id") or question.get("question_id"),
                    "error": str(e),
                }
            )

    summary = summarize_run(question_results)
    summary["errors"] = len(errors)

    return {
        "run_id": f"{profile.name}_run_{run_index}",
        "profile": profile.name,
        "correct_probability": profile.correct_probability,
        "hint_probability": profile.hint_probability,
        "seed": seed,
        "summary": summary,
        "errors": errors,
        "question_results": question_results,
    }


def aggregate_profile(profile_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [r["summary"] for r in runs]

    return {
        "profile": profile_name,
        "runs": len(runs),
        "questions_per_run": summaries[0]["total_questions"] if summaries else 0,
        "avg_correct_rate": round(mean(s["correct_rate"] for s in summaries), 4),
        "avg_hint_rate": round(mean(s["hint_rate"] for s in summaries), 4),
        "avg_raw_score": round(mean(s["avg_raw_score"] for s in summaries), 4),
        "avg_final_score": round(mean(s["avg_final_score"] for s in summaries), 4),
        "final_not_greater_than_raw_all_runs": all(
            s["avg_final_not_greater_than_raw"] for s in summaries
        ),
        "total_errors": sum(s["errors"] for s in summaries),
    }


def run_experiment(
    questions: list[dict[str, Any]],
    n_questions: int,
    n_runs: int,
    evaluator_mode: str,
) -> dict[str, Any]:
    results = {
        "experiment": "simulated_student_agent_experiment",
        "purpose": (
            "Test whether the framework can process different simulated learner "
            "behavior patterns and still produce consistent scores, hint records, "
            "evaluator outputs, and report summaries."
        ),
        "question_type": "multiple_choice",
        "assigned_evaluator": "script_multiple_choice",
        "n_questions": n_questions,
        "n_runs_per_profile": n_runs,
        "profiles": [],
        "profile_summaries": [],
    }

    for profile_index, profile in enumerate(STUDENT_PROFILES):
        runs = []

        for run_index in range(1, n_runs + 1):
            run_result = run_one_session(
                profile_index=profile_index,
                run_index=run_index,
                questions=questions,
                n_questions=n_questions,
                evaluator_mode=evaluator_mode,
            )
            runs.append(run_result)

            s = run_result["summary"]
            print(
                f"[{run_result['run_id']}] "
                f"correct_rate={s['correct_rate']}, "
                f"hint_rate={s['hint_rate']}, "
                f"avg_raw={s['avg_raw_score']}, "
                f"avg_final={s['avg_final_score']}, "
                f"errors={s['errors']}"
            )

        results["profiles"].append(
            {
                "profile": profile.name,
                "correct_probability": profile.correct_probability,
                "hint_probability": profile.hint_probability,
                "description": profile.description,
                "runs": runs,
            }
        )

        results["profile_summaries"].append(
            aggregate_profile(profile.name, runs)
        )

    return results


def save_json(data: dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Optional path to exported AI2 ARC multiple-choice questions.",
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=5,
        help="Number of questions per run.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=2,
        help="Number of runs per student profile.",
    )
    parser.add_argument(
        "--evaluator_mode",
        type=str,
        default="real",
        choices=["real", "mock"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
    )

    args = parser.parse_args()

    questions = load_questions(args.questions)
    n_questions = min(args.n_questions, len(questions))

    results = run_experiment(
        questions=questions,
        n_questions=n_questions,
        n_runs=args.n_runs,
        evaluator_mode=args.evaluator_mode,
    )

    save_json(results, args.output)

    print("\nSaved results to:")
    print(args.output)

    print("\nProfile summaries:")
    for row in results["profile_summaries"]:
        print(row)


if __name__ == "__main__":
    main()