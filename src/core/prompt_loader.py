# src/core/prompt_loader.py

"""
Prompt loader for quiz system prompt templates.

The teacher-provided prompt templates are stored as markdown files.
This loader reads them and replaces {{placeholder}} values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PromptLoader:
    def __init__(self, prompt_dir: str | Path | None = None):
        self.prompt_dir = Path(prompt_dir) if prompt_dir else DEFAULT_PROMPT_DIR

    def load(self, filename: str) -> str:
        path = self.prompt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def render(self, filename: str, **kwargs: Any) -> str:
        template = self.load(filename)

        for key, value in kwargs.items():
            placeholder = "{{" + key + "}}"

            if isinstance(value, (dict, list)):
                replacement = json.dumps(value, ensure_ascii=False, indent=2)
            elif value is None:
                replacement = "null"
            else:
                replacement = str(value)

            template = template.replace(placeholder, replacement)

        return template