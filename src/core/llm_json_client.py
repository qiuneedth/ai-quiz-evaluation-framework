# src/core/llm_json_client.py

"""
LLM JSON client.

This wrapper calls an LLM and parses JSON safely.
It is intentionally small so the thesis can explain it clearly.
"""

from __future__ import annotations

import json
import re
from typing import Any


class LLMJsonClient:
    def __init__(self, openai_client=None, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    def call_json(
        self,
        prompt: str,
        fallback: dict,
        temperature: float = 0.0,
    ) -> dict:
        """
        Call LLM and parse JSON.

        If the LLM call fails or JSON parsing fails, return fallback.
        """

        if self.client is None:
            return {
                **fallback,
                "_generation_method": "fallback_no_llm_client",
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON. Do not include markdown or extra text.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response.choices[0].message.content or ""
            parsed = self._parse_json(content)

            return {
                **parsed,
                "_generation_method": "llm",
            }

        except Exception as exc:
            return {
                **fallback,
                "_generation_method": "fallback_error",
                "_error": str(exc),
            }

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Parse JSON even if the model accidentally wraps it in markdown.
        """
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        return json.loads(text)