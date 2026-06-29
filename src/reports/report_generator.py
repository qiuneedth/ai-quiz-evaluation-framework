# src/reports/report_generator.py

"""
Report Generator.

This module creates and saves:
1. progressive reports
2. final report

The final report should be useful for:
- system evaluation
- learner feedback
- supervisor demonstration
- future web interface display

The report includes:
- raw_score
- final_score
- hint_adjustment
- hint_used
- hint_text
- hint_summary
- hint_score_policy
- evaluator feedback
- teaching explanation
"""


from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ReportGenerator:
    """
    Generate and save reports.
    """

    def generate_report(
        self,
        session_result: dict,
    ) -> dict:
        question_results = session_result.get("question_results", [])

        final_scores = [
            float(result.get("final_score", 0.0))
            for result in question_results
        ]

        raw_scores = [
            float(result.get("raw_score", result.get("final_score", 0.0)))
            for result in question_results
        ]

        if final_scores:
            score_summary = {
                "average": round(sum(final_scores) / len(final_scores), 4),
                "min": round(min(final_scores), 4),
                "max": round(max(final_scores), 4),
            }
        else:
            score_summary = {
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        if raw_scores:
            raw_score_summary = {
                "average": round(sum(raw_scores) / len(raw_scores), 4),
                "min": round(min(raw_scores), 4),
                "max": round(max(raw_scores), 4),
            }
        else:
            raw_score_summary = {
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        wrong_questions = [
            self._build_question_feedback_item(result)
            for result in question_results
            if not result.get("is_correct")
        ]

        correct_questions = [
            self._build_question_feedback_item(result)
            for result in question_results
            if result.get("is_correct")
        ]

        hint_summary = (
            session_result.get("hint_summary")
            or self._build_hint_summary(question_results)
        )

        hint_score_policy = self._get_hint_score_policy(question_results)

        final_report = {
            "report_type": "final",
            "report_id": f"report_{session_result.get('session_id')}",
            "dataset_id": session_result.get("dataset_id"),
            "profile_id": session_result.get("profile_id"),
            "quiz_plan_id": session_result.get("quiz_plan_id"),
            "session_id": session_result.get("session_id"),
            "user_id": session_result.get("user_id"),
            "answer_source": session_result.get("answer_source"),

            "total_questions": session_result.get("total_questions", 0),
            "answered_questions": session_result.get("answered_questions", 0),
            "not_answered_questions": session_result.get("not_answered_questions", 0),
            "answered_question_ids": session_result.get("answered_question_ids", []),
            "not_answered_question_ids": session_result.get("not_answered_question_ids", []),

            # ------------------------------------------------------------
            # Scoring summary.
            # final_score includes hint adjustment.
            # raw_score_summary shows evaluator score before hint adjustment.
            # ------------------------------------------------------------
            "final_score": session_result.get("final_score", 0.0),
            "score_summary": score_summary,
            "raw_score_summary": raw_score_summary,
            "hint_score_policy": hint_score_policy,

            "evaluator_usage": session_result.get("evaluator_usage", {}),
            "hint_summary": hint_summary,

            "wrong_question_count": len(wrong_questions),
            "correct_question_count": len(correct_questions),
            "wrong_questions": wrong_questions,
            "correct_questions": correct_questions,
            "question_results": question_results,
            "progressive_reports": session_result.get("progressive_reports", []),
            "overall_feedback": self._build_overall_feedback(
                final_score=session_result.get("final_score", 0.0),
                wrong_question_count=len(wrong_questions),
                total_questions=session_result.get("total_questions", 0),
                stop_reason=session_result.get("stop_reason"),
                hint_summary=hint_summary,
            ),
            "stop_reason": session_result.get("stop_reason"),
            "created_at": datetime.utcnow().isoformat(),
        }

        return final_report

    def save_report(
        self,
        report: dict,
        output_path: str,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_progressive_reports(
        self,
        progressive_reports: list[dict],
        output_path: str,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            for report in progressive_reports:
                file.write(
                    json.dumps(
                        report,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _build_question_feedback_item(
        self,
        result: dict,
    ) -> dict:
        return {
            "round_index": result.get("round_index"),
            "question_id": result.get("question_id"),
            "question_type": result.get("question_type"),
            "question": result.get("question"),
            "user_answer": result.get("user_answer"),
            "reference_answer": result.get("reference_answer"),
            "correct_answer": result.get("correct_answer"),

            # ------------------------------------------------------------
            # Important score fields.
            # ------------------------------------------------------------
            "raw_score": result.get("raw_score", result.get("final_score")),
            "hint_adjustment": result.get("hint_adjustment", 0.0),
            "hint_penalty": result.get("hint_penalty", 0.0),
            "score": result.get("final_score"),
            "final_score": result.get("final_score"),
            "is_correct": result.get("is_correct"),
            "raw_is_correct": result.get("raw_is_correct"),
            "passed": result.get("passed"),

            # ------------------------------------------------------------
            # Hint fields.
            # ------------------------------------------------------------
            "hint_used": result.get("hint_used", False),
            "hint_text": result.get("hint_text"),
            "hint_payload": result.get("hint_payload"),
            "hint_score_policy": result.get("hint_score_policy"),

            # ------------------------------------------------------------
            # Explanation fields.
            # ------------------------------------------------------------
            "feedback": result.get("feedback"),
            "wrong_answer_explanation": result.get("wrong_answer_explanation"),
            "correct_answer_explanation": result.get("correct_answer_explanation"),
            "learning_feedback": result.get("learning_feedback"),
            "answer_explanation": result.get("answer_explanation", {}),
            "assigned_evaluators": result.get("assigned_evaluators", []),
        }

    def _build_hint_summary(
        self,
        question_results: list[dict],
    ) -> dict:
        total = len(question_results)

        hint_used = [
            result for result in question_results
            if result.get("hint_used")
        ]

        no_hint = [
            result for result in question_results
            if not result.get("hint_used")
        ]

        correct_with_hint = [
            result for result in hint_used
            if result.get("is_correct")
        ]

        wrong_with_hint = [
            result for result in hint_used
            if not result.get("is_correct")
        ]

        correct_without_hint = [
            result for result in no_hint
            if result.get("is_correct")
        ]

        wrong_without_hint = [
            result for result in no_hint
            if not result.get("is_correct")
        ]

        return {
            "total_answered_questions": total,
            "hint_used_count": len(hint_used),
            "hint_not_used_count": len(no_hint),
            "hint_usage_rate": round(len(hint_used) / total, 4) if total else 0.0,
            "correct_with_hint_count": len(correct_with_hint),
            "wrong_with_hint_count": len(wrong_with_hint),
            "correct_without_hint_count": len(correct_without_hint),
            "wrong_without_hint_count": len(wrong_without_hint),
        }

    def _get_hint_score_policy(
        self,
        question_results: list[dict],
    ) -> dict:
        for result in question_results:
            policy = result.get("hint_score_policy")

            if policy:
                return policy

        return {
            "policy_name": "hint_adjusted_non_negative_v1_fallback",
            "raw_score_range": "[0, 1]",
            "final_score_range": "[0, 1]",
            "hint_penalty": 0.10,
            "negative_scores_allowed": False,
            "description": (
                "Default fallback policy. The evaluator produces raw_score in [0, 1]. "
                "If a hint was used, final_score = clamp(raw_score - 0.10, 0, 1). "
                "Negative final scores are not used."
            ),
        }

    def _build_overall_feedback(
        self,
        final_score: float,
        wrong_question_count: int,
        total_questions: int,
        stop_reason: str | None = None,
        hint_summary: dict | None = None,
    ) -> str:
        if stop_reason:
            return stop_reason

        if total_questions == 0:
            return "No questions were answered in this session."

        hint_summary = hint_summary or {}
        hint_used_count = hint_summary.get("hint_used_count", 0)

        if final_score >= 0.85:
            base = (
                "Overall performance is strong. Most answers were correct and well supported."
            )
        elif final_score >= 0.6:
            base = (
                "Overall performance is moderate. Some answers were correct, "
                "but there are still questions that need review."
            )
        elif wrong_question_count > 0:
            base = (
                "Overall performance needs improvement. Review the wrong questions, "
                "correct answers, hints, and explanations carefully."
            )
        else:
            base = "Review the session results for more details."

        if hint_used_count > 0:
            base += (
                f" The user used hints for {hint_used_count} question(s). "
                "The final score reflects the hint-adjusted scoring policy."
            )

        return base
    
    