# src/core/simple_evaluation_engine.py

"""
Simple Evaluation Engine.

This module evaluates a user's answer to a quiz question.

Important design:
- Hint happens BEFORE the user submits an answer.
- Evaluation happens AFTER the user submits an answer.
- Explanation happens AFTER evaluation.
- The evaluator first produces a raw_score.
- If hint was used, the system applies a hint-adjusted scoring policy.
- The final_score is used in progressive report and final report.
- raw_score is kept for transparency.

Teacher feedback implemented:
For correctness-style evaluation:
    Correct + no hint  -> 1.0
    Wrong + no hint    -> 0.0
    Correct + hint     -> slightly less than 1, e.g. 0.9
    Wrong + hint       -> small negative, e.g. -0.01

For other evaluation dimensions such as completeness, relevance, and groundedness,
hint may have a smaller effect. This engine keeps a simple general policy for now,
but stores the policy explicitly so it can be extended later.
"""


from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from src.core.answer_explanation_generator import AnswerExplanationGenerator


class SimpleEvaluationEngine:
    """
    Evaluate one question answer using assigned evaluators.

    Supported evaluator types:
    - script_multiple_choice
    - script_true_false
    - keyword
    - llm_semantic
    - context_llm_semantic
    """

    def __init__(
        self,
        evaluator_mode: str = "real",
        allow_mock_fallback: bool = False,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.evaluator_mode = evaluator_mode
        self.allow_mock_fallback = allow_mock_fallback

        self._load_env()

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.client = None

        if OpenAI is not None and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)

        self.explanation_generator = AnswerExplanationGenerator(
            openai_api_key=self.openai_api_key,
            model=self.model,
        )

        # ------------------------------------------------------------
        # Hint scoring policy.
        #
        # This directly follows the latest supervisor feedback:
        # Correct without hint = 1.0
        # Wrong without hint   = 0.0
        # Correct with hint    = 0.9
        # Wrong with hint      = -0.01
        #
        # For non-binary / partial-score evaluators, this policy also keeps
        # a smaller hint penalty for partial answers.
        # ------------------------------------------------------------
        self.hint_score_policy = {
            "policy_name": "correctness_hint_adjusted_score_v1",
            "correct_no_hint": 1.0,
            "wrong_no_hint": 0.0,
            "correct_with_hint": 0.9,
            "wrong_with_hint": -0.01,
            "partial_with_hint_penalty": 0.05,
            "description": (
                "Hint usage affects final_score but does not erase raw_score. "
                "For correctness-style evaluation, correct answers with hint receive 0.9, "
                "wrong answers with hint receive -0.01, correct answers without hint receive 1.0, "
                "and wrong answers without hint receive 0.0."
            ),
        }

    # ============================================================
    # Public method
    # ============================================================

    def evaluate_question(
        self,
        question: dict,
        user_answer: str,
        assigned_evaluators: list[str],
        answer_source: str | None = None,
        round_index: int | None = None,
        hint_used: bool = False,
        hint_text: str | None = None,
    ) -> dict:
        """
        Evaluate one question.

        Important:
            raw_score is the evaluator's original score.
            final_score is the hint-adjusted score.
        """

        question_id = self._get_question_id(question)
        question_text = question.get("question", "")
        reference_answer = self._get_reference_answer(question)
        correct_answer = reference_answer
        question_type = self._get_question_type(question)

        user_answer = "" if user_answer is None else str(user_answer)

        if not assigned_evaluators:
            assigned_evaluators = [self._choose_default_evaluator(question)]

        evaluator_outputs = []

        for evaluator_name in assigned_evaluators:
            output = self._run_single_evaluator(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

            evaluator_outputs.append(output)

        raw_score = self._aggregate_scores(evaluator_outputs)

        # is_correct should be based on the evaluator result before hint penalty.
        raw_is_correct = raw_score >= 0.999

        hint_adjustment_result = self._apply_hint_score_adjustment(
            raw_score=raw_score,
            hint_used=hint_used,
            assigned_evaluators=assigned_evaluators,
        )

        final_score = hint_adjustment_result["final_score"]
        hint_adjustment = hint_adjustment_result["hint_adjustment"]

        # Correctness label remains based on whether the answer itself is correct.
        # For example, correct with hint should still be is_correct=True,
        # even though final_score becomes 0.9.
        is_correct = raw_is_correct

        # passed can be based on final_score.
        passed = final_score >= 0.5

        feedback = self._combine_feedback(evaluator_outputs)

        wrong_answer_explanation = self._combine_wrong_answer_explanations(
            evaluator_outputs=evaluator_outputs,
            is_correct=is_correct,
        )

        correct_answer_explanation = self._combine_correct_answer_explanations(
            evaluator_outputs=evaluator_outputs,
        )

        learning_feedback = self._combine_learning_feedback(evaluator_outputs)

        answer_explanation = self.explanation_generator.generate_explanation(
            question=question,
            user_answer=user_answer,
            reference_answer=reference_answer,
            final_score=round(final_score, 4),
            is_correct=is_correct,
            evaluator_feedback=feedback,
            evaluator_outputs=evaluator_outputs,
            hint_used=hint_used,
            hint_text=hint_text,
        )

        result = {
            "round_index": round_index,
            "question_id": question_id,
            "question_type": question_type,
            "question": question_text,
            "context": question.get("context"),
            "options": question.get("options") or question.get("choices"),
            "reference_answer": reference_answer,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "answer_source": answer_source,
            "assigned_evaluators": assigned_evaluators,
            "evaluator_outputs": evaluator_outputs,

            # ------------------------------------------------------------
            # Score fields.
            # ------------------------------------------------------------
            "raw_score": round(raw_score, 4),
            "hint_adjustment": round(hint_adjustment, 4),
            "final_score": round(final_score, 4),
            "hint_score_policy": self.hint_score_policy,

            # ------------------------------------------------------------
            # Correctness fields.
            # ------------------------------------------------------------
            "is_correct": is_correct,
            "raw_is_correct": raw_is_correct,
            "passed": passed,

            # ------------------------------------------------------------
            # Feedback and explanation.
            # ------------------------------------------------------------
            "feedback": feedback,
            "wrong_answer_explanation": wrong_answer_explanation,
            "correct_answer_explanation": correct_answer_explanation,
            "learning_feedback": learning_feedback,
            "answer_explanation": answer_explanation,

            # ------------------------------------------------------------
            # Hint fields.
            # ------------------------------------------------------------
            "hint_used": hint_used,
            "hint_text": hint_text,
        }

        return result

    # ============================================================
    # Hint score adjustment
    # ============================================================

    def _apply_hint_score_adjustment(
        self,
        raw_score: float,
        hint_used: bool,
        assigned_evaluators: list[str],
    ) -> dict:
        """
        Apply hint-adjusted scoring policy.

        Main correctness policy:
            Correct + no hint  -> 1.0
            Wrong + no hint    -> 0.0
            Correct + hint     -> 0.9
            Wrong + hint       -> -0.01

        For partial scores:
            If hint is used and raw_score is between 0 and 1,
            apply a small penalty.
        """

        raw_score = self._safe_float(raw_score)

        if not hint_used:
            return {
                "raw_score": raw_score,
                "final_score": raw_score,
                "hint_adjustment": 0.0,
                "policy_applied": "no_hint_no_adjustment",
            }

        metric_family = self._infer_metric_family(assigned_evaluators)

        # Correctness-style binary case.
        if raw_score >= 0.999:
            if metric_family == "correctness":
                final_score = self.hint_score_policy["correct_with_hint"]
            else:
                # For completeness / relevance / groundedness,
                # hint has slightly less effect.
                final_score = max(
                    0.0,
                    raw_score - self.hint_score_policy["partial_with_hint_penalty"],
                )

            return {
                "raw_score": raw_score,
                "final_score": final_score,
                "hint_adjustment": final_score - raw_score,
                "policy_applied": f"{metric_family}_correct_with_hint",
            }

        # Clearly wrong case.
        if raw_score <= 0.0:
            if metric_family == "correctness":
                final_score = self.hint_score_policy["wrong_with_hint"]
            else:
                # For other dimensions, keep a smaller negative effect.
                final_score = -0.005

            return {
                "raw_score": raw_score,
                "final_score": final_score,
                "hint_adjustment": final_score - raw_score,
                "policy_applied": f"{metric_family}_wrong_with_hint",
            }

        # Partial score case.
        final_score = max(
            -0.01,
            raw_score - self.hint_score_policy["partial_with_hint_penalty"],
        )

        return {
            "raw_score": raw_score,
            "final_score": final_score,
            "hint_adjustment": final_score - raw_score,
            "policy_applied": f"{metric_family}_partial_with_hint",
        }

    def _infer_metric_family(
        self,
        assigned_evaluators: list[str],
    ) -> str:
        """
        Infer metric family from evaluator names.

        This is useful because teacher mentioned that hint may have
        different effect for correctness, completeness, relevance, etc.
        """

        names = " ".join(str(name).lower() for name in assigned_evaluators)

        if "complete" in names:
            return "completeness"

        if "relevance" in names or "relevant" in names:
            return "relevance"

        if "ground" in names or "context" in names:
            return "groundedness"

        if "multiple_choice" in names or "true_false" in names or "correct" in names:
            return "correctness"

        return "correctness"

    # ============================================================
    # Evaluator routing
    # ============================================================

    def _run_single_evaluator(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        """
        Run one evaluator.
        """

        if self.evaluator_mode == "mock":
            return self._mock_evaluate(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

        invalid_output = self._precheck_invalid_answer(
            evaluator_name=evaluator_name,
            question=question,
            user_answer=user_answer,
            reference_answer=reference_answer,
        )

        if invalid_output is not None:
            return invalid_output

        normalized_name = str(evaluator_name).lower()

        if "multiple_choice" in normalized_name or "mcq" in normalized_name:
            return self._evaluate_multiple_choice(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

        if "true_false" in normalized_name or "boolean" in normalized_name:
            return self._evaluate_true_false(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

        if "keyword" in normalized_name:
            return self._evaluate_keyword(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

        if "context" in normalized_name or "grounded" in normalized_name:
            return self._evaluate_context_llm_semantic(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

        if "llm" in normalized_name or "semantic" in normalized_name:
            return self._evaluate_llm_semantic(
                evaluator_name=evaluator_name,
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
            )

        return self._evaluate_llm_semantic(
            evaluator_name=evaluator_name,
            question=question,
            user_answer=user_answer,
            reference_answer=reference_answer,
        )

    # ============================================================
    # Invalid answer pre-check
    # ============================================================

    def _precheck_invalid_answer(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict | None:
        """
        Deterministic pre-check for clearly invalid answers.
        """

        answer = str(user_answer or "").strip()
        answer_lower = answer.lower()

        if not answer:
            return self._build_invalid_answer_output(
                evaluator_name=evaluator_name,
                score=0.0,
                feedback="The submitted answer is empty.",
                error_type="empty_answer",
                wrong_answer_explanation=(
                    "No answer was submitted, so the system cannot evaluate it as correct."
                ),
                correct_answer_explanation=(
                    f"The expected answer is: {reference_answer}."
                ),
                learning_feedback="Submit an answer before requesting evaluation.",
            )

        unknown_answers = {
            "idk",
            "i don't know",
            "i dont know",
            "don't know",
            "dont know",
            "unknown",
            "not sure",
            "no idea",
            "不知道",
            "不清楚",
            "不会",
            "不知道。",
        }

        if answer_lower in unknown_answers:
            return self._build_invalid_answer_output(
                evaluator_name=evaluator_name,
                score=0.0,
                feedback="The submitted answer indicates that the user does not know the answer.",
                error_type="unknown_answer",
                wrong_answer_explanation=(
                    "The answer does not provide the requested information."
                ),
                correct_answer_explanation=(
                    f"The expected answer is: {reference_answer}."
                ),
                learning_feedback=(
                    "Try to identify what type of answer the question expects before answering."
                ),
            )

        question_has_options = bool(question.get("options") or question.get("choices"))

        if not question_has_options:
            if re.fullmatch(r"\d+(\.\d+)?", answer):
                if answer.strip() != str(reference_answer).strip():
                    return self._build_invalid_answer_output(
                        evaluator_name=evaluator_name,
                        score=0.0,
                        feedback="The submitted answer is a numeric value that does not match the expected answer.",
                        error_type="meaningless_numeric_answer",
                        wrong_answer_explanation=(
                            f"The numeric answer '{answer}' does not answer the question correctly."
                        ),
                        correct_answer_explanation=(
                            f"The expected answer is: {reference_answer}."
                        ),
                        learning_feedback=(
                            "Check whether the question expects a name, concept, date, or explanation rather than a random number."
                        ),
                    )

        return None

    def _build_invalid_answer_output(
        self,
        evaluator_name: str,
        score: float,
        feedback: str,
        error_type: str,
        wrong_answer_explanation: str,
        correct_answer_explanation: str,
        learning_feedback: str,
    ) -> dict:
        return {
            "evaluator_name": evaluator_name,
            "score": score,
            "passed": False,
            "details": {
                "error_type": error_type,
            },
            "feedback": feedback,
            "wrong_answer_explanation": wrong_answer_explanation,
            "correct_answer_explanation": correct_answer_explanation,
            "learning_feedback": learning_feedback,
            "raw_output": {
                "score": score,
                "passed": False,
                "error_type": error_type,
                "feedback": feedback,
            },
        }

    # ============================================================
    # Script evaluators
    # ============================================================

    def _evaluate_multiple_choice(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        """
        Deterministic multiple-choice evaluator.

        Supports:
        - answer by label: A / B / C / D
        - answer by text: option text
        """

        options = question.get("options") or question.get("choices") or []

        candidate_label, candidate_text = self._normalize_mcq_answer(
            answer=user_answer,
            options=options,
        )

        reference_label, reference_text = self._normalize_mcq_answer(
            answer=reference_answer,
            options=options,
        )

        is_correct = (
            candidate_label is not None
            and reference_label is not None
            and candidate_label == reference_label
        )

        score = 1.0 if is_correct else 0.0

        if is_correct:
            feedback = (
                f"Score {score}: candidate answer '{user_answer}' maps to option "
                f"{candidate_label}, which matches the reference answer '{reference_answer}'."
            )

            wrong_answer_explanation = ""
            correct_answer_explanation = (
                f"The correct answer is option {reference_label}: {reference_text}."
            )
            learning_feedback = "Good job. You selected the correct option."

        else:
            feedback = (
                f"Score {score}: candidate answer '{user_answer}' maps to option "
                f"{candidate_label}, but the reference answer '{reference_answer}' maps to option "
                f"{reference_label}. Candidate option text: '{candidate_text}'. "
                f"Reference option text: '{reference_text}'."
            )

            wrong_answer_explanation = (
                f"Your answer is incorrect because it maps to option {candidate_label}, "
                f"while the correct option is {reference_label}. You selected "
                f"'{candidate_text}', but the expected answer is '{reference_text}'."
            )

            correct_answer_explanation = (
                f"The correct answer is option {reference_label}: {reference_text}."
            )

            learning_feedback = (
                "Review the question carefully and compare each option with the expected answer."
            )

        return {
            "evaluator_name": evaluator_name,
            "score": score,
            "passed": is_correct,
            "details": {
                "candidate_label": candidate_label,
                "reference_label": reference_label,
                "candidate_option_text": candidate_text,
                "reference_option_text": reference_text,
            },
            "feedback": feedback,
            "wrong_answer_explanation": wrong_answer_explanation,
            "correct_answer_explanation": correct_answer_explanation,
            "learning_feedback": learning_feedback,
            "raw_output": {
                "score": score,
                "passed": is_correct,
                "details": {
                    "candidate_label": candidate_label,
                    "reference_label": reference_label,
                    "candidate_option_text": candidate_text,
                    "reference_option_text": reference_text,
                },
                "feedback": feedback,
            },
        }

    def _normalize_mcq_answer(
        self,
        answer: str,
        options: Any,
    ) -> tuple[str | None, str | None]:
        """
        Normalize MCQ answer to option label and option text.
        """

        if answer is None:
            return None, None

        answer_text = str(answer).strip()
        answer_lower = answer_text.lower()

        option_list = self._normalize_options(options)

        for option in option_list:
            label = str(option.get("label", "")).strip()
            text = str(option.get("text", "")).strip()

            if answer_lower == label.lower():
                return label, text

        for option in option_list:
            label = str(option.get("label", "")).strip()
            text = str(option.get("text", "")).strip()

            if answer_lower == text.lower():
                return label, text

        cleaned = re.sub(r"[^a-zA-Z0-9]", "", answer_lower)

        for option in option_list:
            label = str(option.get("label", "")).strip()

            if cleaned == label.lower():
                return label, str(option.get("text", "")).strip()

        return answer_text.upper() if len(answer_text) == 1 else None, None

    def _normalize_options(
        self,
        options: Any,
    ) -> list[dict]:
        """
        Normalize option structures.

        Supports:
        - [{"label": "A", "text": "..."}]
        - {"label": ["A", "B"], "text": ["...", "..."]}
        - ["A. ...", "B. ..."]
        """

        if not options:
            return []

        if isinstance(options, list):
            normalized = []

            for index, option in enumerate(options):
                if isinstance(option, dict):
                    label = option.get("label")
                    text = option.get("text")

                    if label is None:
                        label = chr(ord("A") + index)

                    if text is None:
                        text = option.get("value") or option.get("option") or ""

                    normalized.append(
                        {
                            "label": str(label),
                            "text": str(text),
                        }
                    )

                else:
                    label = chr(ord("A") + index)
                    normalized.append(
                        {
                            "label": label,
                            "text": str(option),
                        }
                    )

            return normalized

        if isinstance(options, dict):
            labels = options.get("label") or options.get("labels") or []
            texts = options.get("text") or options.get("texts") or []

            if isinstance(labels, list) and isinstance(texts, list):
                return [
                    {
                        "label": str(label),
                        "text": str(text),
                    }
                    for label, text in zip(labels, texts)
                ]

        return []

    def _evaluate_true_false(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        candidate = self._normalize_boolean(user_answer)
        reference = self._normalize_boolean(reference_answer)

        is_correct = candidate is not None and reference is not None and candidate == reference
        score = 1.0 if is_correct else 0.0

        return {
            "evaluator_name": evaluator_name,
            "score": score,
            "passed": is_correct,
            "details": {
                "candidate_boolean": candidate,
                "reference_boolean": reference,
            },
            "feedback": (
                f"Score {score}: candidate boolean={candidate}, reference boolean={reference}."
            ),
            "wrong_answer_explanation": (
                "" if is_correct else f"The submitted answer '{user_answer}' does not match the expected boolean answer '{reference_answer}'."
            ),
            "correct_answer_explanation": (
                f"The correct answer is {reference_answer}."
            ),
            "learning_feedback": (
                "Check whether the statement is true or false based on the question."
            ),
            "raw_output": {
                "score": score,
                "passed": is_correct,
            },
        }

    def _normalize_boolean(
        self,
        value: str,
    ) -> bool | None:
        value_lower = str(value).strip().lower()

        true_values = {"true", "t", "yes", "y", "1", "correct"}
        false_values = {"false", "f", "no", "n", "0", "incorrect"}

        if value_lower in true_values:
            return True

        if value_lower in false_values:
            return False

        return None

    def _evaluate_keyword(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        candidate_tokens = self._tokenize(user_answer)
        reference_tokens = self._tokenize(reference_answer)

        if not reference_tokens:
            score = 0.0
        else:
            overlap = candidate_tokens.intersection(reference_tokens)
            score = len(overlap) / len(reference_tokens)

        score = round(score, 4)
        passed = score >= 0.6

        return {
            "evaluator_name": evaluator_name,
            "score": score,
            "passed": passed,
            "details": {
                "candidate_tokens": sorted(candidate_tokens),
                "reference_tokens": sorted(reference_tokens),
                "overlap": sorted(candidate_tokens.intersection(reference_tokens)),
            },
            "feedback": (
                f"Keyword overlap score is {score}."
            ),
            "wrong_answer_explanation": (
                "" if passed else "The answer does not cover enough key terms from the expected answer."
            ),
            "correct_answer_explanation": (
                f"The expected answer contains key information related to: {reference_answer}."
            ),
            "learning_feedback": (
                "Try to include the key concepts from the expected answer."
            ),
            "raw_output": {
                "score": score,
                "passed": passed,
            },
        }

    # ============================================================
    # LLM evaluators
    # ============================================================

    def _evaluate_llm_semantic(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        prompt = f"""
You are evaluating a student's answer.

Evaluate semantic correctness compared with the reference answer.

Return only valid JSON:
{{
  "score": 0.0,
  "passed": false,
  "feedback": "...",
  "wrong_answer_explanation": "...",
  "correct_answer_explanation": "...",
  "learning_feedback": "..."
}}

Question:
{question.get("question", "")}

Student answer:
{user_answer}

Reference answer:
{reference_answer}
"""

        return self._call_llm_evaluator(
            evaluator_name=evaluator_name,
            prompt=prompt,
        )

    def _evaluate_context_llm_semantic(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        context = self._stringify_context(question.get("context", ""))

        prompt = f"""
You are evaluating a student's answer for a context-based QA task.

Evaluate:
- correctness
- relevance
- whether the answer is supported by the provided context

Return only valid JSON:
{{
  "score": 0.0,
  "passed": false,
  "feedback": "...",
  "wrong_answer_explanation": "...",
  "correct_answer_explanation": "...",
  "learning_feedback": "..."
}}

Question:
{question.get("question", "")}

Context:
{context[:12000]}

Student answer:
{user_answer}

Reference answer:
{reference_answer}
"""

        return self._call_llm_evaluator(
            evaluator_name=evaluator_name,
            prompt=prompt,
        )

    def _call_llm_evaluator(
        self,
        evaluator_name: str,
        prompt: str,
    ) -> dict:
        if self.client is None:
            if self.allow_mock_fallback:
                return self._mock_llm_output(evaluator_name)
            raise RuntimeError(
                "OpenAI client unavailable and allow_mock_fallback=False."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict quiz evaluator. Return only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
            )

            raw_text = response.choices[0].message.content or ""
            parsed = self._parse_json_from_text(raw_text)

            score = self._safe_float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))

            passed = bool(parsed.get("passed", score >= 0.5))

            return {
                "evaluator_name": evaluator_name,
                "score": round(score, 4),
                "passed": passed,
                "details": parsed.get("details", {}),
                "feedback": parsed.get("feedback", ""),
                "wrong_answer_explanation": parsed.get("wrong_answer_explanation", ""),
                "correct_answer_explanation": parsed.get("correct_answer_explanation", ""),
                "learning_feedback": parsed.get("learning_feedback", ""),
                "raw_output": parsed,
            }

        except Exception as error:
            if self.allow_mock_fallback:
                return self._mock_llm_output(evaluator_name)

            return {
                "evaluator_name": evaluator_name,
                "score": 0.0,
                "passed": False,
                "details": {
                    "error": str(error),
                },
                "feedback": f"LLM evaluator failed: {error}",
                "wrong_answer_explanation": "The answer could not be evaluated because the LLM evaluator failed.",
                "correct_answer_explanation": "",
                "learning_feedback": "Please retry evaluation or use another evaluator.",
                "raw_output": {},
            }

    # ============================================================
    # Mock evaluators
    # ============================================================

    def _mock_evaluate(
        self,
        evaluator_name: str,
        question: dict,
        user_answer: str,
        reference_answer: str,
    ) -> dict:
        score = 1.0 if str(user_answer).strip().lower() == str(reference_answer).strip().lower() else 0.0
        passed = score >= 0.5

        return {
            "evaluator_name": evaluator_name,
            "score": score,
            "passed": passed,
            "details": {
                "mock": True,
            },
            "feedback": f"Mock score {score}.",
            "wrong_answer_explanation": "" if passed else "Mock evaluator marked this answer as incorrect.",
            "correct_answer_explanation": f"The reference answer is {reference_answer}.",
            "learning_feedback": "This is mock evaluation feedback.",
            "raw_output": {
                "score": score,
                "passed": passed,
            },
        }

    def _mock_llm_output(
        self,
        evaluator_name: str,
    ) -> dict:
        return {
            "evaluator_name": evaluator_name,
            "score": 0.5,
            "passed": True,
            "details": {
                "mock_llm": True,
            },
            "feedback": "Mock LLM evaluator output.",
            "wrong_answer_explanation": "",
            "correct_answer_explanation": "This is a mock explanation.",
            "learning_feedback": "This is mock learning feedback.",
            "raw_output": {
                "score": 0.5,
                "passed": True,
            },
        }

    # ============================================================
    # Aggregation
    # ============================================================

    def _aggregate_scores(
        self,
        evaluator_outputs: list[dict],
    ) -> float:
        if not evaluator_outputs:
            return 0.0

        scores = [
            self._safe_float(output.get("score", 0.0))
            for output in evaluator_outputs
        ]

        return sum(scores) / len(scores)

    def _combine_feedback(
        self,
        evaluator_outputs: list[dict],
    ) -> str:
        parts = []

        for output in evaluator_outputs:
            evaluator_name = output.get("evaluator_name")
            feedback = output.get("feedback", "")

            if feedback:
                parts.append(f"[{evaluator_name}] {feedback}")

        return "\n".join(parts)

    def _combine_wrong_answer_explanations(
        self,
        evaluator_outputs: list[dict],
        is_correct: bool,
    ) -> str:
        if is_correct:
            return ""

        parts = []

        for output in evaluator_outputs:
            text = output.get("wrong_answer_explanation", "")

            if text:
                parts.append(text)

        return " ".join(parts)

    def _combine_correct_answer_explanations(
        self,
        evaluator_outputs: list[dict],
    ) -> str:
        parts = []

        for output in evaluator_outputs:
            text = output.get("correct_answer_explanation", "")

            if text:
                parts.append(text)

        return " ".join(parts)

    def _combine_learning_feedback(
        self,
        evaluator_outputs: list[dict],
    ) -> str:
        parts = []

        for output in evaluator_outputs:
            text = output.get("learning_feedback", "")

            if text:
                parts.append(text)

        return " ".join(parts)

    # ============================================================
    # Helpers
    # ============================================================

    def _choose_default_evaluator(
        self,
        question: dict,
    ) -> str:
        options = question.get("options") or question.get("choices")
        context = question.get("context")

        if options:
            return "script_multiple_choice"

        if context:
            return "context_llm_semantic"

        return "llm_semantic"

    def _get_question_id(
        self,
        question: dict,
    ) -> str:
        return str(
            question.get("question_id")
            or question.get("id")
            or question.get("qid")
            or "unknown_question"
        )

    def _get_reference_answer(
        self,
        question: dict,
    ) -> str:
        return str(
            question.get("reference_answer")
            or question.get("answer")
            or question.get("label")
            or question.get("target")
            or ""
        )

    def _get_question_type(
        self,
        question: dict,
    ) -> str:
        metadata = question.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        return str(
            metadata.get("question_type")
            or question.get("question_type")
            or "unknown"
        )

    def _tokenize(
        self,
        text: str,
    ) -> set[str]:
        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                str(text).lower(),
            )
        )

    def _stringify_context(
        self,
        context: Any,
    ) -> str:
        if context is None:
            return ""

        if isinstance(context, str):
            return context

        return json.dumps(context, ensure_ascii=False)

    def _safe_float(
        self,
        value: Any,
    ) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _parse_json_from_text(
        self,
        text: str,
    ) -> dict:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```json", "", text)
            text = re.sub(r"^```", "", text)
            text = re.sub(r"```$", "", text)
            text = text.strip()

        return json.loads(text)

    def _load_env(
        self,
    ) -> None:
        if load_dotenv is None:
            return

        possible_paths = [
            Path.cwd() / ".env",
            Path.cwd() / "src" / ".env",
        ]

        for path in possible_paths:
            if path.exists():
                load_dotenv(path)