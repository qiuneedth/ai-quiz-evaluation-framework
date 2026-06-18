# src/core/env_loader.py

"""
Environment Loader.

This module loads environment variables from .env files.

Why this file is needed:
The project may store API keys in:
- .env
- src/.env

Python does not automatically load these files.
Therefore, we explicitly load them with python-dotenv.

Important:
Never print or log API keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    """
    Load project environment variables.

    It tries both:
    1. project root .env
    2. src/.env

    This makes the project work even if the .env file is placed inside src/.
    """

    # Current file:
    # src/core/env_loader.py
    current_file = Path(__file__).resolve()

    # Project root:
    # rag_base/
    project_root = current_file.parents[2]

    # Possible .env locations.
    root_env = project_root / ".env"
    src_env = project_root / "src" / ".env"

    # Load root .env first if it exists.
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)

    # Load src/.env if it exists.
    if src_env.exists():
        load_dotenv(dotenv_path=src_env, override=False)


def get_openai_api_key() -> str | None:
    """
    Return OpenAI API key from environment variables.
    """

    load_project_env()
    return os.getenv("OPENAI_API_KEY")


def get_openai_chat_model() -> str:
    """
    Return OpenAI chat model name.

    Supports both:
    - OPENAI_CHAT_MODEL
    - OPENAI_MODEL

    OPENAI_CHAT_MODEL is preferred because your .env uses this name.
    """

    load_project_env()

    return (
        os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


def get_openai_embed_model() -> str:
    """
    Return OpenAI embedding model name.
    """

    load_project_env()

    return (
        os.getenv("OPENAI_EMBED_MODEL")
        or "text-embedding-3-small"
    )