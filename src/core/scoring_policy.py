# src/core/scoring_policy.py

"""
Scoring policy for quiz evaluation.

This module separates:
- raw_score: produced by the evaluator, always in [0, 1]
- final_score: computed after optional hint adjustment, always in [0, 1]

Important:
The evaluator result must not be overwritten by the hint penalty.
Hint usage only affects the final aggregated score, not the evaluator's original judgment.
"""

from __future__ import annotations


def clamp_score(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Clamp a score to the range [0, 1].
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0

    return max(min_value, min(max_value, value))


def apply_hint_adjustment(
    raw_score: float,
    hint_used: bool,
    hint_penalty: float = 0.10,
) -> dict:
    """
    Apply hint penalty to the evaluator raw score.

    Rules:
    - raw_score is produced by evaluator and remains in [0, 1]
    - if hint_used is False, final_score = raw_score
    - if hint_used is True, final_score = clamp(raw_score - 0.10, 0, 1)
    - final_score is never negative

    Examples:
    - correct without hint: raw=1.0, final=1.0
    - correct with hint: raw=1.0, final=0.9
    - wrong without hint: raw=0.0, final=0.0
    - wrong with hint: raw=0.0, final=0.0
    - partial with hint: raw=0.7, final=0.6
    """

    raw_score = clamp_score(raw_score)
    penalty = hint_penalty if hint_used else 0.0
    final_score = clamp_score(raw_score - penalty)

    return {
        "raw_score": raw_score,
        "hint_used": bool(hint_used),
        "hint_penalty": penalty,
        "hint_adjustment": -penalty,
        "final_score": final_score,
        "scoring_policy": {
            "policy_name": "hint_adjusted_non_negative_v1",
            "raw_score_range": "[0, 1]",
            "final_score_range": "[0, 1]",
            "hint_penalty": hint_penalty,
            "negative_scores_allowed": False,
        },
    }


def infer_passed_from_raw_score(raw_score: float, threshold: float = 0.5) -> bool:
    """
    Determine correctness/pass status from raw_score.

    For learning-oriented quiz systems:
    - raw_score should decide whether the answer is correct or passed.
    - final_score is used for score aggregation after hint penalty.
    """
    return clamp_score(raw_score) >= threshold