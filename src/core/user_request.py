# src/core/user_request.py

"""
User Request Schema.

The user request is the entry point of the framework.

It should not only specify a dataset.
It should describe the whole quiz/evaluation session:

- What topic or dataset the user wants
- What question types are requested
- How many questions are requested
- How questions should be collected
- Whether questions should be unseen-only or random
- Whether topic matching should be strict or soft
- Whether progressive/final reports are needed
- Whether answers come from a human user or from an LLM experiment

This module normalizes both flat and nested JSON request formats.
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserRequest:
    request_id: str | None = None
    user_id: str = "demo_user"

    dataset_key: str | None = None
    topic_query: str | None = None
    topic: str | None = None
    domain: str | None = None
    question_types: list[str] = field(default_factory=list)
    allow_profile_suggestion: bool = True
    selection_scope: str = "single_profile"

    total_questions: int = 5
    questions_per_round: int = 1

    collection_plan: str = "1b1-ordered"
    progress_plan: str = "partial_eval"
    question_selection_plan: str = "unseen_only"
    topic_match_mode: str = "soft"
    insufficient_question_policy: str = "allow_fallback"
    repeat_probability: float = 0.0
    random_seed: int | None = 42
    seen_question_ids: list[str] = field(default_factory=list)

    delivery_mode: str = "one_by_one"
    report_mode: str = "progressive_and_final"

    answer_source: str = "manual"
    evaluator_mode: str = "real"
    allow_mock_fallback: bool = False

    extra_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,

            "dataset_key": self.dataset_key,
            "topic_query": self.topic_query,
            "topic": self.topic,
            "domain": self.domain,
            "question_types": self.question_types,
            "allow_profile_suggestion": self.allow_profile_suggestion,
            "selection_scope": self.selection_scope,

            "total_questions": self.total_questions,
            "questions_per_round": self.questions_per_round,

            "collection_plan": self.collection_plan,
            "progress_plan": self.progress_plan,
            "question_selection_plan": self.question_selection_plan,
            "topic_match_mode": self.topic_match_mode,
            "insufficient_question_policy": self.insufficient_question_policy,
            "repeat_probability": self.repeat_probability,
            "random_seed": self.random_seed,
            "seen_question_ids": self.seen_question_ids,

            "delivery_mode": self.delivery_mode,
            "report_mode": self.report_mode,

            "answer_source": self.answer_source,
            "evaluator_mode": self.evaluator_mode,
            "allow_mock_fallback": self.allow_mock_fallback,

            "extra_constraints": self.extra_constraints,
        }


def build_user_request_from_dict(data: dict) -> UserRequest:
    dataset_request = data.get("dataset_request", {})
    quiz_request = data.get("quiz_request", {})
    planner = data.get("planner", {})
    answer_request = data.get("answer_request", {})
    report_request = data.get("report_request", {})

    dataset_key = dataset_request.get("dataset_key", data.get("dataset_key"))
    topic_query = dataset_request.get(
        "topic_query",
        data.get("topic_query", data.get("topic")),
    )
    topic = dataset_request.get("topic", data.get("topic"))
    domain = dataset_request.get("domain", data.get("domain"))

    question_types = dataset_request.get(
        "question_types",
        data.get("question_types", []),
    ) or []

    allow_profile_suggestion = bool(
        dataset_request.get(
            "allow_profile_suggestion",
            data.get("allow_profile_suggestion", True),
        )
    )

    selection_scope = dataset_request.get(
        "selection_scope",
        data.get("selection_scope", "single_profile"),
    )

    total_questions = int(
        quiz_request.get(
            "total_questions",
            data.get("total_questions", data.get("num_questions", 5)),
        )
    )

    questions_per_round = int(
        quiz_request.get(
            "questions_per_round",
            data.get("questions_per_round", 1),
        )
    )

    collection_plan = planner.get(
        "collection_plan",
        data.get("collection_plan", "1b1-ordered"),
    )

    progress_plan = planner.get(
        "progress_plan",
        data.get("progress_plan", "partial_eval"),
    )

    question_selection_plan = planner.get(
        "question_selection_plan",
        data.get("question_selection_plan", "unseen_only"),
    )

    topic_match_mode = planner.get(
        "topic_match_mode",
        data.get("topic_match_mode", "soft"),
    )

    insufficient_question_policy = planner.get(
        "insufficient_question_policy",
        data.get("insufficient_question_policy", "allow_fallback"),
    )

    repeat_probability = float(
        planner.get(
            "repeat_probability",
            data.get("repeat_probability", 0.0),
        )
    )

    random_seed = planner.get(
        "random_seed",
        data.get("random_seed", 42),
    )

    seen_question_ids = planner.get(
        "seen_question_ids",
        data.get("seen_question_ids", []),
    )

    answer_source = answer_request.get(
        "answer_source",
        data.get("answer_source", "manual"),
    )

    evaluator_mode = answer_request.get(
        "evaluator_mode",
        data.get("evaluator_mode", "real"),
    )

    allow_mock_fallback = bool(
        answer_request.get(
            "allow_mock_fallback",
            data.get("allow_mock_fallback", False),
        )
    )

    report_mode = data.get("report_mode")

    if report_mode is None:
        progressive_report = report_request.get("progressive_report", True)
        final_report = report_request.get("final_report", True)

        if progressive_report and final_report:
            report_mode = "progressive_and_final"
        elif progressive_report:
            report_mode = "progressive"
        elif final_report:
            report_mode = "final_only"
        else:
            report_mode = "none"

    delivery_mode = data.get("delivery_mode")

    if delivery_mode is None:
        delivery_mode = _infer_delivery_mode(collection_plan, questions_per_round)

    return UserRequest(
        request_id=data.get("request_id"),
        user_id=data.get("user_id", "demo_user"),

        dataset_key=dataset_key,
        topic_query=topic_query,
        topic=topic,
        domain=domain,
        question_types=question_types,
        allow_profile_suggestion=allow_profile_suggestion,
        selection_scope=selection_scope,

        total_questions=total_questions,
        questions_per_round=questions_per_round,

        collection_plan=collection_plan,
        progress_plan=progress_plan,
        question_selection_plan=question_selection_plan,
        topic_match_mode=topic_match_mode,
        insufficient_question_policy=insufficient_question_policy,
        repeat_probability=repeat_probability,
        random_seed=random_seed,
        seen_question_ids=seen_question_ids,

        delivery_mode=delivery_mode,
        report_mode=report_mode,

        answer_source=answer_source,
        evaluator_mode=evaluator_mode,
        allow_mock_fallback=allow_mock_fallback,

        extra_constraints=data.get("extra_constraints", {}),
    )


def build_user_request_from_args(args) -> UserRequest:
    question_types = getattr(args, "question_types", None)

    if isinstance(question_types, str):
        question_types = [
            item.strip()
            for item in question_types.split(",")
            if item.strip()
        ]

    return UserRequest(
        dataset_key=getattr(args, "dataset", None),
        topic_query=getattr(args, "topic", None),
        topic=getattr(args, "topic", None),
        domain=getattr(args, "domain", None),
        question_types=question_types or [],
        total_questions=getattr(args, "total_questions", 5),
        questions_per_round=getattr(args, "questions_per_round", 1),
        collection_plan=getattr(args, "collection_plan", "1b1-ordered"),
        progress_plan=getattr(args, "progress_plan", "partial_eval"),
        question_selection_plan=getattr(args, "question_selection_plan", "unseen_only"),
        topic_match_mode=getattr(args, "topic_match_mode", "soft"),
        insufficient_question_policy=getattr(args, "insufficient_question_policy", "allow_fallback"),
        repeat_probability=getattr(args, "repeat_probability", 0.0),
        random_seed=getattr(args, "random_seed", 42),
        delivery_mode=getattr(args, "delivery_mode", "one_by_one"),
        report_mode=getattr(args, "report_mode", "progressive_and_final"),
        answer_source=getattr(args, "answer_source", "manual"),
        evaluator_mode=getattr(args, "evaluator_mode", "real"),
        allow_mock_fallback=getattr(args, "allow_mock_fallback", False),
    )


def _infer_delivery_mode(collection_plan: str, questions_per_round: int) -> str:
    if collection_plan in [
        "1b1-ordered",
        "1b1-random",
        "one_by_one_ordered",
        "one_by_one_random",
    ]:
        return "one_by_one"

    if collection_plan in [
        "bBb-random",
        "bBb-ordered",
        "batch_random",
        "batch_ordered",
    ]:
        return "batch"

    if questions_per_round > 1:
        return "batch"

    return "one_by_one"