# src/storage/user_activity_logger.py

"""
User Activity Logger.

This module records user activity during evaluation.

The supervisor requires a user activity/history table.

This logger stores every event as one JSON line.

Examples of events:
- evaluator_instance_started
- question_answered
- question_evaluated
- evaluator_instance_finished
- dataset_selected
- evaluation_plan_selected

The output file can later be used as a user activity table.
"""


import json
import uuid
from datetime import datetime
from pathlib import Path


class UserActivityLogger:
    """
    Store user activity records in JSONL format.

    JSONL means:
    - one JSON object per line
    - easy to append
    - easy to read later as a table
    """

    def __init__(self, log_path: str):
        """
        Initialize logger.

        log_path:
            The path where user activity records are stored.
            Example:
            data/results/user_activity.jsonl
        """

        self.log_path = Path(log_path)

        # Make sure the parent directory exists.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        user_id: str,
        event_type: str,
        dataset_id: str | None = None,
        profile_id: str | None = None,
        plan_id: str | None = None,
        evaluator_instance_id: str | None = None,
        question_id: str | None = None,
        question_text: str | None = None,
        user_answer: str | None = None,
        reference_answer: str | None = None,
        selected_evaluators: list[str] | None = None,
        metric_scores: dict | None = None,
        final_score: float | None = None,
        status: str = "created",
        metadata: dict | None = None,
    ) -> dict:
        """
        Create and save one user activity record.

        This record is designed to support the user activity table.

        It stores:
        - selected dataset
        - selected profile
        - selected plan
        - evaluator instance id
        - answered question
        - user answer
        - scores
        - timestamp
        """

        if selected_evaluators is None:
            selected_evaluators = []

        if metric_scores is None:
            metric_scores = {}

        if metadata is None:
            metadata = {}

        record = {
            "activity_id": str(uuid.uuid4()),
            "user_id": user_id,
            "event_type": event_type,
            "dataset_id": dataset_id,
            "profile_id": profile_id,
            "plan_id": plan_id,
            "evaluator_instance_id": evaluator_instance_id,
            "question_id": question_id,
            "question_text": question_text,
            "user_answer": user_answer,
            "reference_answer": reference_answer,
            "selected_evaluators": selected_evaluators,
            "metric_scores": metric_scores,
            "final_score": final_score,
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
        }

        self._append_record(record)

        return record

    def _append_record(self, record: dict) -> None:
        """
        Append one record to the JSONL file.
        """

        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")