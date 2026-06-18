# src/core/request_loader.py

"""
Request Loader.

Loads user_request JSON from file.

This makes the runtime entry closer to the supervisor's expected design:
the system receives a user_request dictionary, not only command-line dataset arguments.
"""


import json
from pathlib import Path

from src.core.user_request import build_user_request_from_dict


def load_user_request_from_json(
    request_file: str,
) -> dict:
    """
    Load and normalize user request JSON.

    Args:
        request_file:
            Path to JSON file.

    Returns:
        Normalized user request dictionary.
    """

    path = Path(request_file)

    if not path.exists():
        raise FileNotFoundError(f"User request file not found: {request_file}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))

    return build_user_request_from_dict(raw_data).to_dict()