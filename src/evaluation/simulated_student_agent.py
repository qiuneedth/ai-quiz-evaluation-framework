# src/evaluation/simulated_student_agent.py

"""
Simulated Student Agent for evaluation experiments.

This module is used only for Chapter 5 evaluation experiments.

Purpose:
- Simulate different learner behavior patterns.
- Control correctness probability and hint usage probability.
- Generate answers for multiple-choice questions.
- Do NOT evaluate answers here.

The generated answers should be passed to the framework's normal
EvaluationEngine, so the experiment tests the framework rather than
the simulated student itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StudentProfile:
    """
    A simulated learner behavior profile.

    correct_probability:
        Probability that the simulated student intends to answer correctly.

    hint_probability:
        Probability that the simulated student requests a hint before answering.
    """

    name: str
    correct_probability: float
    hint_probability: float
    description: str


STUDENT_PROFILES: list[StudentProfile] = [
    StudentProfile(
        name="strong",
        correct_probability=0.85,
        hint_probability=0.20,
        description="Usually answers correctly and rarely requests hints.",
    ),
    StudentProfile(
        name="average",
        correct_probability=0.60,
        hint_probability=0.40,
        description="Shows mixed correctness and moderate hint usage.",
    ),
    StudentProfile(
        name="weak",
        correct_probability=0.35,
        hint_probability=0.70,
        description="Often answers incorrectly and frequently requests hints.",
    ),
]


class SimulatedStudentAgent:
    """
    Rule-based simulated student for multiple-choice questions.

    The agent controls only the learner behavior:
    - whether a hint is requested
    - whether the submitted answer is intended to be correct
    - which option is submitted

    It does NOT compute raw_score or final_score.
    """

    def __init__(
        self,
        profile: StudentProfile,
        seed: int | None = None,
    ) -> None:
        self.profile = profile
        self.random = random.Random(seed)

    def decide_hint_used(self) -> bool:
        """
        Decide whether the simulated student requests a hint.
        """

        return self.random.random() < self.profile.hint_probability

    def decide_intended_correctness(self) -> bool:
        """
        Decide whether the simulated student intends to answer correctly.
        """

        return self.random.random() < self.profile.correct_probability

    def answer_multiple_choice(
        self,
        question: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a multiple-choice answer for one question.

        Expected question format:
        {
            "question": "...",
            "answer": "A",
            "choices": {
                "A": "...",
                "B": "...",
                "C": "...",
                "D": "..."
            }
        }

        Also supports options as:
        [
            {"label": "A", "text": "..."},
            {"label": "B", "text": "..."}
        ]
        """

        correct_answer = get_reference_answer(question)
        option_labels = get_option_labels(question)

        if not correct_answer:
            raise ValueError("Question does not contain a reference answer.")

        if not option_labels:
            raise ValueError("Question does not contain multiple-choice options.")

        correct_answer = normalize_label(correct_answer)

        if correct_answer not in option_labels:
            raise ValueError(
                f"Reference answer '{correct_answer}' is not in option labels {option_labels}."
            )

        hint_used = self.decide_hint_used()
        intended_correct = self.decide_intended_correctness()

        if intended_correct:
            user_answer = correct_answer
        else:
            wrong_options = [
                label for label in option_labels
                if label != correct_answer
            ]

            if not wrong_options:
                user_answer = correct_answer
            else:
                user_answer = self.random.choice(wrong_options)

        return {
            "student_profile": self.profile.name,
            "hint_used": hint_used,
            "intended_correct": intended_correct,
            "user_answer": user_answer,
            "reference_answer": correct_answer,
        }


def get_reference_answer(question: dict[str, Any]) -> str:
    """
    Extract reference answer from common field names.
    """

    return str(
        question.get("reference_answer")
        or question.get("answer")
        or question.get("label")
        or question.get("target")
        or ""
    ).strip()


def get_option_labels(question: dict[str, Any]) -> list[str]:
    """
    Extract option labels from question choices/options.
    """

    options = question.get("choices") or question.get("options") or []

    labels: list[str] = []

    if isinstance(options, dict):
        labels = [normalize_label(label) for label in options.keys()]

    elif isinstance(options, list):
        for item in options:
            if isinstance(item, dict):
                label = item.get("label") or item.get("key") or item.get("id")
                if label is not None:
                    labels.append(normalize_label(str(label)))
            else:
                # fallback for ["A", "B", "C", "D"]
                labels.append(normalize_label(str(item)))

    return labels


def normalize_label(value: str) -> str:
    """
    Normalize option label.
    """

    return str(value).strip().upper()