# src/llm/prompt_loader.py

"""
This file loads prompt templates from the prompts folder.

Example:
load_prompt("prompts/rag_rubric_judge_prompt.txt")
"""


def load_prompt(file_path: str) -> str:
    """
    Read a prompt template from a text file.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()