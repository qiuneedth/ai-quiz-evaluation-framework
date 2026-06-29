# src/core/answer_explanation_generator.py

"""
Answer Explanation Generator.

This module generates detailed learner-facing explanations after evaluation.

Important distinction:
- Hint happens BEFORE the user answers.
- Evaluation Engine decides the score AFTER the user answers.
- Explanation Generator explains the result AFTER evaluation.

The explanation should be similar to teaching material:
1. What the question is asking.
2. What concept is being tested.
3. Which evidence or option supports the correct answer.
4. How to reason from the question to the answer.
5. Why the user's answer is wrong or incomplete.
6. Why the correct answer is correct.
7. What the student should remember next time.

This is useful for:
- immediate feedback after each question
- progressive report
- final report
- future web interface "Show detailed explanation" button
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


class AnswerExplanationGenerator:
    """
    Generate detailed learner-facing answer explanations.

    The generator does not rescore the answer.
    It only explains a score that has already been decided by the evaluator.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self._load_env()

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.client = None

        if OpenAI is not None and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)

    def generate_explanation(
        self,
        question: dict,
        user_answer: str,
        reference_answer: str,
        final_score: float,
        is_correct: bool,
        evaluator_feedback: str,
        evaluator_outputs: list[dict] | None = None,
        hint_used: bool = False,
        hint_text: str | None = None,
    ) -> dict:
        """
        Generate detailed answer explanation.

        Returns a structured explanation object that can be displayed
        in terminal or later in a web interface.
        """

        if self.client is None:
            return self._fallback_explanation(
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
                final_score=final_score,
                is_correct=is_correct,
                evaluator_feedback=evaluator_feedback,
                hint_used=hint_used,
                hint_text=hint_text,
            )

        question_text = question.get("question", "")
        context = self._stringify_context(question.get("context", ""))
        options = question.get("options") or question.get("choices") or None

        prompt = self._build_prompt(
            question_text=question_text,
            context=context,
            options=options,
            user_answer=user_answer,
            reference_answer=reference_answer,
            final_score=final_score,
            is_correct=is_correct,
            evaluator_feedback=evaluator_feedback,
            evaluator_outputs=evaluator_outputs or [],
            hint_used=hint_used,
            hint_text=hint_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful quiz teacher. "
                            "You explain quiz answers like teaching material. "
                            "Return only valid JSON."
                        ),
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

            return self._normalize_explanation(
                parsed=parsed,
                final_score=final_score,
                is_correct=is_correct,
            )

        except Exception as error:
            fallback = self._fallback_explanation(
                question=question,
                user_answer=user_answer,
                reference_answer=reference_answer,
                final_score=final_score,
                is_correct=is_correct,
                evaluator_feedback=evaluator_feedback,
                hint_used=hint_used,
                hint_text=hint_text,
            )
            fallback["generation_error"] = str(error)
            return fallback

    def _build_prompt(
        self,
        question_text: str,
        context: str,
        options: Any,
        user_answer: str,
        reference_answer: str,
        final_score: float,
        is_correct: bool,
        evaluator_feedback: str,
        evaluator_outputs: list[dict],
        hint_used: bool,
        hint_text: str | None,
    ) -> str:
        """
        Build LLM prompt for detailed teaching-style answer explanation.
        """

        options_text = json.dumps(options, ensure_ascii=False, indent=2)

        return f"""
You are generating a detailed teaching-style answer explanation for a quiz system.

The goal is NOT to rescore the answer.
The score has already been decided by the evaluator.
Your job is to explain the result to the learner like a teacher.

Return ONLY valid JSON with this exact structure:

{{
  "score_confirmed": 0.0,
  "passed_confirmed": false,
  "short_feedback": "...",
  "question_understanding": "...",
  "key_concept": "...",
  "teaching_explanation": "...",
  "detailed_explanation": "...",
  "evidence_from_context": [
    {{
      "snippet": "...",
      "how_it_supports_answer": "..."
    }}
  ],
  "reasoning_steps": [
    "...",
    "..."
  ],
  "why_user_answer_is_wrong": "...",
  "why_correct_answer_is_correct": "...",
  "option_analysis": [
    {{
      "option_label": "...",
      "option_text": "...",
      "is_correct": false,
      "explanation": "..."
    }}
  ],
  "hint_used_comment": "...",
  "hint_if_retried": "...",
  "error_type": "...",
  "learning_feedback": "..."
}}

Rules:
- Do not rescore the answer.
- Do not change the given score.
- Preserve the already assigned score in score_confirmed.
- Preserve the already assigned correctness in passed_confirmed.
- Do not say the user is correct if is_correct is false.
- Do not say the user is wrong if is_correct is true.
- Make the explanation useful as teaching material.
- Explain the concept being tested.
- Explain how a student should think through the question.
- Use the provided context as evidence when context is available.
- If context is available, identify the most relevant evidence snippet.
- If this is multiple-choice, analyze each option when options are available.
- If options are not available, keep option_analysis as an empty list.
- If the user's answer is wrong, explain specifically what is wrong:
  wrong option, wrong entity, unsupported claim, irrelevant answer,
  empty answer, unknown answer, meaningless numeric answer, partial answer,
  or misread question.
- Do not only say "because the reference answer says so".
- Avoid vague explanation.
- If hint was used, briefly comment on whether the hint should have helped.
- hint_if_retried should be a short hint the student could use if trying again.
  It should not simply reveal the answer.

Question:
{question_text}

Context:
{context[:12000]}

Options:
{options_text}

User answer:
{user_answer}

Correct / reference answer:
{reference_answer}

Already assigned final score:
{final_score}

Already assigned correctness:
{is_correct}

Was hint used before answering?
{hint_used}

Hint text shown to the user:
{hint_text or ""}

Evaluator feedback:
{evaluator_feedback}

Evaluator raw outputs:
{json.dumps(evaluator_outputs, ensure_ascii=False)[:4000]}
"""

    def _normalize_explanation(
        self,
        parsed: dict,
        final_score: float,
        is_correct: bool,
    ) -> dict:
        """
        Ensure all expected fields exist.

        The explanation generator must not rescore the answer. Therefore,
        score_confirmed and passed_confirmed are forced to match the already
        assigned evaluation result.
        """

        return {
            "score_confirmed": final_score,
            "passed_confirmed": is_correct,
            "short_feedback": parsed.get("short_feedback", ""),
            "question_understanding": parsed.get("question_understanding", ""),
            "key_concept": parsed.get("key_concept", ""),
            "teaching_explanation": parsed.get("teaching_explanation", ""),
            "detailed_explanation": parsed.get("detailed_explanation", ""),
            "evidence_from_context": parsed.get("evidence_from_context", []),
            "reasoning_steps": parsed.get("reasoning_steps", []),
            "why_user_answer_is_wrong": parsed.get("why_user_answer_is_wrong", ""),
            "why_correct_answer_is_correct": parsed.get("why_correct_answer_is_correct", ""),
            "option_analysis": parsed.get("option_analysis", []),
            "hint_used_comment": parsed.get("hint_used_comment", ""),
            "hint_if_retried": parsed.get("hint_if_retried", ""),
            "error_type": parsed.get("error_type", ""),
            "learning_feedback": parsed.get("learning_feedback", ""),
        }

    def _fallback_explanation(
        self,
        question: dict,
        user_answer: str,
        reference_answer: str,
        final_score: float,
        is_correct: bool,
        evaluator_feedback: str,
        hint_used: bool = False,
        hint_text: str | None = None,
    ) -> dict:
        """
        Fallback explanation if LLM explanation generation is unavailable.
        """

        question_text = question.get("question", "")

        if is_correct:
            short_feedback = "Your answer is correct."
            why_wrong = ""
        else:
            short_feedback = "Your answer is not correct."
            why_wrong = (
                f"Your answer '{user_answer}' does not match the expected answer "
                f"'{reference_answer}'."
            )

        hint_comment = ""

        if hint_used:
            hint_comment = (
                "A hint was used before answering. The final evaluation still depends "
                "on whether the submitted answer matches the expected answer."
            )

        return {
            "score_confirmed": final_score,
            "passed_confirmed": is_correct,
            "short_feedback": short_feedback,
            "question_understanding": f"The question asks: {question_text}",
            "key_concept": "Identify the main concept or fact required by the question.",
            "teaching_explanation": (
                "To solve this question, first understand what type of information is being asked, "
                "then compare the answer with the relevant evidence or options."
            ),
            "detailed_explanation": (
                f"The expected answer is '{reference_answer}'. "
                f"Evaluator feedback: {evaluator_feedback}"
            ),
            "evidence_from_context": [],
            "reasoning_steps": [
                "Read the question carefully.",
                "Identify what information the question is asking for.",
                "Compare the submitted answer with the expected answer.",
            ],
            "why_user_answer_is_wrong": why_wrong,
            "why_correct_answer_is_correct": (
                f"The correct answer is '{reference_answer}'."
            ),
            "option_analysis": [],
            "hint_used_comment": hint_comment,
            "hint_if_retried": (
                "Focus on the key concept in the question and eliminate answers that do not directly address it."
            ),
            "error_type": "fallback_explanation",
            "learning_feedback": (
                "Review the question and the expected answer carefully."
            ),
        }

    def _parse_json_from_text(
        self,
        text: str,
    ) -> dict:
        """
        Parse JSON from LLM output.
        """

        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```json", "", text)
            text = re.sub(r"^```", "", text)
            text = re.sub(r"```$", "", text)
            text = text.strip()

        return json.loads(text)

    def _stringify_context(
        self,
        context: Any,
    ) -> str:
        """
        Convert context to string.
        """

        if context is None:
            return ""

        if isinstance(context, str):
            return context

        return json.dumps(context, ensure_ascii=False)

    def _load_env(
        self,
    ) -> None:
        """
        Load .env if available.
        """

        if load_dotenv is None:
            return

        possible_paths = [
            Path.cwd() / ".env",
            Path.cwd() / "src" / ".env",
        ]

        for path in possible_paths:
            if path.exists():
                load_dotenv(path)