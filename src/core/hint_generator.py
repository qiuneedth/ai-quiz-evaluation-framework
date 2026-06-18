# src/core/hint_generator.py

"""
Hint Generator.

This module generates one hint before the user submits an answer.

Important design:
- Hint happens BEFORE evaluation.
- Explanation happens AFTER evaluation.
- The hint should guide the student toward the correct reasoning direction.
- The hint must NOT reveal the final answer directly.
- Each question should only allow one hint.

Example:
Question:
    Which of these explains why many plants and animals died out at the end
    of the Mesozoic era?

Bad hint:
    The answer is D.

Good hint:
    Think about a sudden event that could block sunlight and change the
    environment globally.
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


class HintGenerator:
    """
    Generate one learner-facing hint for a quiz question.
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

    def generate_hint(
        self,
        question: dict,
    ) -> dict:
        """
        Generate a hint object.

        Returns:
            {
                "hint_text": "...",
                "hint_type": "...",
                "does_reveal_answer": false,
                "generation_method": "llm" | "fallback",
                "raw_output": {...}
            }
        """

        if self.client is None:
            return self._fallback_hint(question)

        question_text = question.get("question", "")
        context = self._stringify_context(question.get("context", ""))
        options = question.get("options") or question.get("choices") or []
        reference_answer = self._get_reference_answer(question)
        question_type = self._get_question_type(question)

        prompt = self._build_prompt(
            question_text=question_text,
            context=context,
            options=options,
            reference_answer=reference_answer,
            question_type=question_type,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful teaching assistant. "
                            "You generate hints for students, but you never reveal the final answer. "
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

            return self._normalize_hint(parsed, generation_method="llm")

        except Exception as error:
            fallback = self._fallback_hint(question)
            fallback["generation_error"] = str(error)
            return fallback

    def _build_prompt(
        self,
        question_text: str,
        context: str,
        options: Any,
        reference_answer: str,
        question_type: str,
    ) -> str:
        """
        Build hint generation prompt.
        """

        options_text = json.dumps(options, ensure_ascii=False, indent=2)

        return f"""
You are generating a hint for a student before they answer a quiz question.

Return ONLY valid JSON with this exact structure:

{{
  "hint_text": "...",
  "hint_type": "conceptual_hint",
  "does_reveal_answer": false
}}

Rules:
- The hint must help the student move toward the correct reasoning direction.
- The hint must NOT reveal the final answer.
- The hint must NOT say the correct option label.
- The hint must NOT quote the exact correct option text.
- The hint should be short, clear, and useful.
- The hint should sound like a teacher guiding a student.
- If this is multiple-choice, guide the student on how to compare the options.
- If this is context-based QA, guide the student to find the relevant evidence in the context.
- If this is short-answer QA, guide the student to identify the key entity or concept.
- Do not evaluate any user answer because the user has not answered yet.

Question type:
{question_type}

Question:
{question_text}

Context:
{context[:4000]}

Options:
{options_text}

Correct/reference answer, for internal use only.
Do NOT reveal it directly:
{reference_answer}
"""

    def _normalize_hint(
        self,
        parsed: dict,
        generation_method: str,
    ) -> dict:
        """
        Normalize hint object.
        """

        hint_text = str(parsed.get("hint_text", "")).strip()

        return {
            "hint_text": hint_text,
            "hint_type": parsed.get("hint_type", "conceptual_hint"),
            "does_reveal_answer": bool(parsed.get("does_reveal_answer", False)),
            "generation_method": generation_method,
            "raw_output": parsed,
        }

    def _fallback_hint(
        self,
        question: dict,
    ) -> dict:
        """
        Fallback hint if LLM is unavailable.
        """

        options = question.get("options") or question.get("choices")
        context = question.get("context")
        question_type = self._get_question_type(question)

        if options:
            hint_text = (
                "Compare each option with the question carefully. "
                "Eliminate options that do not directly explain what the question is asking."
            )
            hint_type = "multiple_choice_strategy_hint"

        elif context:
            hint_text = (
                "Look for the sentence or passage in the context that directly supports "
                "the answer. Focus on the key entity mentioned in the question."
            )
            hint_type = "context_search_hint"

        elif question_type in ["short_text", "factoid_qa", "short_answer"]:
            hint_text = (
                "Identify what type of answer the question expects, such as a person, "
                "place, date, object, or concept."
            )
            hint_type = "answer_type_hint"

        else:
            hint_text = (
                "Read the question carefully and focus on the main concept being tested."
            )
            hint_type = "general_hint"

        return {
            "hint_text": hint_text,
            "hint_type": hint_type,
            "does_reveal_answer": False,
            "generation_method": "fallback",
            "raw_output": {
                "hint_text": hint_text,
                "hint_type": hint_type,
                "does_reveal_answer": False,
            },
        }

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

    def _stringify_context(
        self,
        context: Any,
    ) -> str:
        if context is None:
            return ""

        if isinstance(context, str):
            return context

        return json.dumps(context, ensure_ascii=False)

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