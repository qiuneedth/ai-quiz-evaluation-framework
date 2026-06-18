# src/core/quiz_plan_generator.py

"""
Quiz Plan Generator.

This module generates a quiz plan from:
- selected dataset profile
- completed user request

Main responsibilities:
1. Extract candidate questions from selected dataset profile.
2. Filter questions by seen_question_ids.
3. Match topic_query using hybrid topic relevance matching.
4. Select questions according to collection plan.
5. Split selected questions into rounds.
6. Assign evaluators to each question.
7. Return a structured quiz plan.

Important design:
Topic matching should NOT be simple substring matching.

Bad example:
    topic_query = "cat"
    "cat" should NOT match "character".

But topic matching should also NOT be only exact word matching.

Bad example:
    topic_query = "science"
    A science question may not explicitly contain the word "science".

Therefore, this generator uses hybrid topic relevance matching:

Layer 1:
    Profile-level topic relevance.
    If the selected dataset profile is clearly aligned with the topic,
    all candidate questions are considered topic-compatible.

Layer 2:
    Question-level exact token matching.
    This prevents false positives such as "cat" matching "character".

Layer 3:
    Optional LLM semantic relevance matching.
    This helps broad topics such as "animal", "history", "science", etc.,
    where the topic word may not appear directly in the question text.

If no relevant questions are found, the generator returns:
    plan_status = "no_topic_match"

In a future web interface, this can be shown as:
    "No matching questions found. Please choose another topic, another dataset,
     or allow fallback."
"""


from __future__ import annotations

import json
import os
import random
import re
import uuid
from datetime import datetime
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


class QuizPlanGenerator:
    """
    Generate quiz plan from selected dataset profile and completed user request.
    """

    def __init__(
        self,
        use_llm_topic_matching: bool = True,
        max_llm_topic_checks: int = 20,
        topic_match_threshold: float = 0.6,
    ):
        self.use_llm_topic_matching = use_llm_topic_matching
        self.max_llm_topic_checks = max_llm_topic_checks
        self.topic_match_threshold = topic_match_threshold

        self._load_env()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.client = None

        if OpenAI is not None and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)

    # ============================================================
    # Public method
    # ============================================================

    def generate_quiz_plan(
        self,
        profile: dict | None = None,
        user_request: dict | None = None,
        selected_profile: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Generate quiz plan.

        Compatible call styles:

            generate_quiz_plan(profile=selected_profile, user_request=...)
            generate_quiz_plan(selected_profile=selected_profile, user_request=...)

        This avoids:
            TypeError: got an unexpected keyword argument 'profile'
        """

        if profile is None:
            profile = selected_profile

        if profile is None:
            profile = kwargs.get("dataset_profile")

        if profile is None:
            raise ValueError(
                "QuizPlanGenerator.generate_quiz_plan() requires a dataset profile."
            )

        if user_request is None:
            user_request = kwargs.get("request")

        if user_request is None:
            raise ValueError(
                "QuizPlanGenerator.generate_quiz_plan() requires a user_request dictionary."
            )

        dataset_id = self._get_dataset_id(profile)
        profile_id = self._get_profile_id(profile)

        total_questions = int(user_request.get("total_questions", 5))
        questions_per_round = int(user_request.get("questions_per_round", 1))

        collection_plan = user_request.get("collection_plan", "1b1-ordered")
        progress_plan = user_request.get("progress_plan", "partial_eval")
        question_selection_plan = user_request.get("question_selection_plan", "unseen_only")

        topic_query = (
            user_request.get("topic_query")
            or user_request.get("topic")
            or None
        )

        topic_match_mode = user_request.get("topic_match_mode", "soft")
        insufficient_question_policy = user_request.get(
            "insufficient_question_policy",
            "allow_fallback",
        )

        random_seed = user_request.get("random_seed", 42)
        repeat_probability = float(user_request.get("repeat_probability", 0.0))

        answer_source = user_request.get("answer_source", "manual")
        evaluator_mode = user_request.get("evaluator_mode", "real")
        delivery_mode = user_request.get("delivery_mode")
        report_mode = user_request.get("report_mode", "progressive_and_final")

        seen_question_ids = set(
            str(qid)
            for qid in user_request.get("seen_question_ids", [])
        )

        requested_question_types = user_request.get("question_types", []) or []

        all_questions = self._extract_questions_from_profile(profile)
        available_question_count = len(all_questions)

        candidate_questions = self._filter_seen_questions(
            questions=all_questions,
            seen_question_ids=seen_question_ids,
            question_selection_plan=question_selection_plan,
        )

        candidate_questions = self._filter_question_types(
            questions=candidate_questions,
            requested_question_types=requested_question_types,
        )

        candidate_question_count = len(candidate_questions)

        topic_match_result = self._apply_hybrid_topic_matching(
            profile=profile,
            questions=candidate_questions,
            topic_query=topic_query,
            topic_match_mode=topic_match_mode,
            insufficient_question_policy=insufficient_question_policy,
            total_questions=total_questions,
        )

        topic_matched_questions = topic_match_result["topic_matched_questions"]
        final_candidate_questions = topic_match_result["final_candidate_questions"]
        topic_match_debug = topic_match_result["topic_match_debug"]
        topic_match_warnings = topic_match_result["warnings"]
        topic_match_source = topic_match_result["topic_match_source"]
        profile_topic_match = topic_match_result["profile_topic_match"]
        topic_matched_count = len(topic_matched_questions)

        selection_warnings = []
        selection_warnings.extend(topic_match_warnings)

        if topic_query and not profile_topic_match and topic_matched_count == 0:
            return self._build_no_topic_match_plan(
                dataset_id=dataset_id,
                profile_id=profile_id,
                user_request=user_request,
                topic_query=topic_query,
                topic_match_debug=topic_match_debug,
                available_question_count=available_question_count,
                candidate_question_count=candidate_question_count,
                selection_warnings=selection_warnings,
                profile_topic_match_debug=topic_match_result.get("profile_topic_match_debug", {}),
            )

        selected_questions = self._select_questions(
            questions=final_candidate_questions,
            total_questions=total_questions,
            collection_plan=collection_plan,
            random_seed=random_seed,
        )

        if len(selected_questions) < total_questions:
            selection_warnings.append(
                f"Requested {total_questions} questions, but only "
                f"{len(selected_questions)} questions could be selected."
            )

        selected_questions = [
            self._attach_question_metadata(
                question=question,
                profile=profile,
            )
            for question in selected_questions
        ]

        rounds = self._build_rounds(
            selected_questions=selected_questions,
            questions_per_round=questions_per_round,
        )

        assigned_evaluators = self._assign_evaluators(
            profile=profile,
            selected_questions=selected_questions,
        )

        if delivery_mode is None:
            delivery_mode = (
                "one_by_one"
                if questions_per_round == 1
                else "batch"
            )

        quiz_plan_id = f"quiz_plan_{dataset_id}_{uuid.uuid4().hex[:8]}"

        selection_summary = {
            "requested_questions": total_questions,
            "available_questions_in_profile": available_question_count,
            "candidate_questions_after_filter": candidate_question_count,
            "candidate_questions_after_filtering": candidate_question_count,
            "topic_query": topic_query,
            "topic_match_mode": topic_match_mode,
            "topic_match_source": topic_match_source,
            "profile_topic_match": profile_topic_match,
            "profile_topic_match_debug": topic_match_result.get("profile_topic_match_debug", {}),
            "topic_matched_questions": topic_matched_count,
            "selected_questions": len(selected_questions),
            "topic_match_debug": topic_match_debug,
            "warnings": selection_warnings,
        }

        quiz_plan = {
            "plan_status": "ready",
            "quiz_plan_id": quiz_plan_id,
            "dataset_id": dataset_id,
            "profile_id": profile_id,
            "created_at": datetime.utcnow().isoformat(),
            "session_config": {
                "total_questions": total_questions,
                "questions_per_round": questions_per_round,
                "delivery_mode": delivery_mode,
                "report_mode": report_mode,
                "answer_source": answer_source,
                "evaluator_mode": evaluator_mode,
            },
            "planner_config": {
                "collection_plan": collection_plan,
                "progress_plan": progress_plan,
                "question_selection_plan": question_selection_plan,
                "topic_match_mode": topic_match_mode,
                "insufficient_question_policy": insufficient_question_policy,
                "repeat_probability": repeat_probability,
                "random_seed": random_seed,
                "topic_query": topic_query,
                "requested_question_types": requested_question_types,
            },
            "selection_summary": selection_summary,
            "selected_questions": selected_questions,
            "rounds": rounds,
            "assigned_evaluators": assigned_evaluators,
        }

        return quiz_plan

    # ============================================================
    # No topic match plan
    # ============================================================

    def _build_no_topic_match_plan(
        self,
        dataset_id: str,
        profile_id: str,
        user_request: dict,
        topic_query: str,
        topic_match_debug: list[dict],
        available_question_count: int,
        candidate_question_count: int,
        selection_warnings: list[str],
        profile_topic_match_debug: dict | None = None,
    ) -> dict:
        """
        Build a safe plan when no topic-relevant questions are found.

        This avoids traceback and avoids silently selecting unrelated questions.
        """

        message = (
            f"No topic-relevant questions were found for topic_query='{topic_query}'. "
            "The quiz plan was not created. Please choose another topic, choose another "
            "dataset, or explicitly allow fallback questions."
        )

        selection_warnings.append(message)

        quiz_plan_id = f"quiz_plan_{dataset_id}_no_topic_match_{uuid.uuid4().hex[:8]}"

        total_questions = int(user_request.get("total_questions", 5))
        questions_per_round = int(user_request.get("questions_per_round", 1))

        return {
            "plan_status": "no_topic_match",
            "quiz_plan_id": quiz_plan_id,
            "dataset_id": dataset_id,
            "profile_id": profile_id,
            "created_at": datetime.utcnow().isoformat(),
            "stop_reason": message,
            "session_config": {
                "total_questions": total_questions,
                "questions_per_round": questions_per_round,
                "delivery_mode": user_request.get("delivery_mode", "one_by_one"),
                "report_mode": user_request.get("report_mode", "progressive_and_final"),
                "answer_source": user_request.get("answer_source", "manual"),
                "evaluator_mode": user_request.get("evaluator_mode", "real"),
            },
            "planner_config": {
                "collection_plan": user_request.get("collection_plan", "1b1-ordered"),
                "progress_plan": user_request.get("progress_plan", "partial_eval"),
                "question_selection_plan": user_request.get("question_selection_plan", "unseen_only"),
                "topic_match_mode": user_request.get("topic_match_mode", "soft"),
                "insufficient_question_policy": user_request.get(
                    "insufficient_question_policy",
                    "allow_fallback",
                ),
                "repeat_probability": float(user_request.get("repeat_probability", 0.0)),
                "random_seed": user_request.get("random_seed", 42),
                "topic_query": topic_query,
                "requested_question_types": user_request.get("question_types", []) or [],
            },
            "selection_summary": {
                "requested_questions": total_questions,
                "available_questions_in_profile": available_question_count,
                "candidate_questions_after_filter": candidate_question_count,
                "candidate_questions_after_filtering": candidate_question_count,
                "topic_query": topic_query,
                "topic_match_source": "no_match",
                "profile_topic_match": False,
                "profile_topic_match_debug": profile_topic_match_debug or {},
                "topic_matched_questions": 0,
                "selected_questions": 0,
                "topic_match_debug": topic_match_debug,
                "warnings": selection_warnings,
            },
            "selected_questions": [],
            "rounds": [],
            "assigned_evaluators": {},
        }

    # ============================================================
    # Hybrid topic relevance matching
    # ============================================================

    def _apply_hybrid_topic_matching(
        self,
        profile: dict,
        questions: list[dict],
        topic_query: str | None,
        topic_match_mode: str,
        insufficient_question_policy: str,
        total_questions: int,
    ) -> dict:
        """
        Apply hybrid topic relevance matching.

        Layer 1:
            Profile-level relevance.

        Layer 2:
            Question-level exact token matching.

        Layer 3:
            Optional LLM semantic relevance matching.
        """

        warnings = []
        topic_match_debug = []

        if not topic_query:
            return {
                "profile_topic_match": False,
                "profile_topic_match_debug": {},
                "topic_match_source": "no_topic_query",
                "topic_matched_questions": [],
                "final_candidate_questions": questions,
                "topic_match_debug": topic_match_debug,
                "warnings": warnings,
            }

        profile_match, profile_debug = self._profile_matches_topic(
            profile=profile,
            topic_query=topic_query,
        )

        if profile_match:
            warnings.append(
                f"Topic query '{topic_query}' matched the selected dataset profile. "
                "Candidate questions are treated as topic-compatible at profile level."
            )

            for question in questions:
                topic_match_debug.append(
                    {
                        "question_id": self._get_question_id(question),
                        "question": question.get("question", ""),
                        "matched": True,
                        "reason": "profile_level_topic_match",
                        "matched_terms": profile_debug.get("matched_terms", []),
                        "llm_checked": False,
                        "llm_relevance": None,
                    }
                )

            return {
                "profile_topic_match": True,
                "profile_topic_match_debug": profile_debug,
                "topic_match_source": "profile_level",
                "topic_matched_questions": questions,
                "final_candidate_questions": questions,
                "topic_match_debug": topic_match_debug,
                "warnings": warnings,
            }

        exact_matched_questions = []
        uncertain_questions = []

        for question in questions:
            matched, debug_info = self._question_exact_matches_topic(
                question=question,
                topic_query=topic_query,
            )

            if matched:
                exact_matched_questions.append(question)
                topic_match_debug.append(
                    {
                        "question_id": self._get_question_id(question),
                        "question": question.get("question", ""),
                        **debug_info,
                        "llm_checked": False,
                        "llm_relevance": None,
                    }
                )
            else:
                uncertain_questions.append(question)
                topic_match_debug.append(
                    {
                        "question_id": self._get_question_id(question),
                        "question": question.get("question", ""),
                        **debug_info,
                        "llm_checked": False,
                        "llm_relevance": None,
                    }
                )

        semantic_matched_questions = []

        if self.use_llm_topic_matching and self.client is not None:
            questions_to_check = uncertain_questions[: self.max_llm_topic_checks]

            for question in questions_to_check:
                llm_result = self._llm_question_matches_topic(
                    question=question,
                    topic_query=topic_query,
                )

                if llm_result.get("is_relevant"):
                    semantic_matched_questions.append(question)

                question_id = self._get_question_id(question)

                for debug_item in topic_match_debug:
                    if debug_item.get("question_id") == question_id:
                        debug_item["llm_checked"] = True
                        debug_item["llm_relevance"] = llm_result

                        if llm_result.get("is_relevant"):
                            debug_item["matched"] = True
                            debug_item["reason"] = "llm_semantic_topic_match"
                            debug_item["matched_terms"] = llm_result.get("matched_concepts", [])

                        break

        elif self.use_llm_topic_matching and self.client is None:
            warnings.append(
                "LLM topic matching was enabled, but OpenAI client is unavailable. "
                "Only profile-level and exact token topic matching were used."
            )

        topic_matched_questions = self._deduplicate_questions(
            exact_matched_questions + semantic_matched_questions
        )

        if not topic_matched_questions:
            warnings.append(
                f"No topic-relevant questions were found for topic_query='{topic_query}'."
            )

            return {
                "profile_topic_match": False,
                "profile_topic_match_debug": profile_debug,
                "topic_match_source": "no_match",
                "topic_matched_questions": [],
                "final_candidate_questions": [],
                "topic_match_debug": topic_match_debug,
                "warnings": warnings,
            }

        if topic_match_mode == "strict":
            final_candidate_questions = topic_matched_questions
            topic_match_source = "question_level_strict"

        elif topic_match_mode == "soft":
            if (
                insufficient_question_policy == "allow_fallback"
                and len(topic_matched_questions) < total_questions
            ):
                matched_ids = {
                    self._get_question_id(question)
                    for question in topic_matched_questions
                }

                fallback_questions = [
                    question
                    for question in questions
                    if self._get_question_id(question) not in matched_ids
                ]

                final_candidate_questions = topic_matched_questions + fallback_questions

                warnings.append(
                    f"Topic query matched {len(topic_matched_questions)} questions. "
                    "Fallback questions may be used because requested total_questions "
                    "exceeds the number of topic-matched questions."
                )

                topic_match_source = "question_level_with_fallback"

            else:
                final_candidate_questions = topic_matched_questions
                topic_match_source = "question_level"

        else:
            final_candidate_questions = questions
            topic_match_source = "topic_matching_disabled"

        return {
            "profile_topic_match": False,
            "profile_topic_match_debug": profile_debug,
            "topic_match_source": topic_match_source,
            "topic_matched_questions": topic_matched_questions,
            "final_candidate_questions": final_candidate_questions,
            "topic_match_debug": topic_match_debug,
            "warnings": warnings,
        }

    # ============================================================
    # Profile-level topic matching
    # ============================================================

    def _profile_matches_topic(
        self,
        profile: dict,
        topic_query: str | None,
    ) -> tuple[bool, dict]:
        """
        Check whether the selected dataset profile is relevant to topic_query.

        Example:
            topic_query = "science"
            profile domain = "science"
            profile category = "multiple_choice_science_qa"
            → match

        Example:
            topic_query = "cat"
            profile topic = "multi_hop_wikipedia_qa"
            → no match
        """

        if not topic_query:
            return False, {
                "matched": False,
                "reason": "no_topic_query",
                "matched_terms": [],
            }

        query = str(topic_query).strip().lower()
        query_tokens = self._tokenize(query)

        profile_text = self._profile_to_search_text(profile)
        profile_text_lower = profile_text.lower()
        profile_tokens = self._tokenize(profile_text_lower)

        if not query_tokens:
            return False, {
                "matched": False,
                "reason": "empty_query_tokens",
                "matched_terms": [],
                "profile_text": profile_text,
            }

        if len(query_tokens) == 1:
            token = list(query_tokens)[0]
            matched = token in profile_tokens

            return matched, {
                "matched": matched,
                "reason": "profile_exact_token_match",
                "matched_terms": [token] if matched else [],
                "profile_text": profile_text,
            }

        phrase_pattern = r"\b" + re.escape(query) + r"\b"
        phrase_match = re.search(phrase_pattern, profile_text_lower) is not None
        all_tokens_match = all(token in profile_tokens for token in query_tokens)

        matched_terms = [
            token
            for token in query_tokens
            if token in profile_tokens
        ]

        matched = phrase_match or all_tokens_match

        return matched, {
            "matched": matched,
            "reason": "profile_phrase_or_all_tokens_match",
            "matched_terms": matched_terms,
            "profile_text": profile_text,
        }

    def _profile_to_search_text(
        self,
        profile: dict,
    ) -> str:
        """
        Build searchable profile text from nested profile fields.
        """

        identity = profile.get("dataset_identity", {})
        structure = profile.get("dataset_structure", {})
        semantic = profile.get("semantic_profile", {})
        qa = profile.get("question_answer_profile", {})
        basic = profile.get("basic_features", {})
        dataset_metadata = profile.get("dataset_metadata", {})

        parts = [
            profile.get("dataset_id", ""),
            profile.get("dataset_key", ""),
            profile.get("name", ""),
            profile.get("profile_id", ""),
            profile.get("dataset_category", ""),
            profile.get("category", ""),
            profile.get("domain", ""),
            profile.get("topic", ""),
            profile.get("description", ""),
            identity.get("dataset_name", ""),
            identity.get("dataset_key", ""),
            identity.get("dataset_description", ""),
            structure.get("dataset_category", ""),
            semantic.get("domain", ""),
            semantic.get("topic", ""),
            semantic.get("subdomain", ""),
            qa.get("question_type", ""),
            qa.get("answer_type", ""),
            qa.get("reasoning_type", ""),
            qa.get("context_dependency", ""),
            basic.get("domain", ""),
            dataset_metadata.get("description", ""),
        ]

        return " ".join(
            str(part)
            for part in parts
            if part is not None and str(part).strip()
        )

    # ============================================================
    # Question-level topic matching
    # ============================================================

    def _question_exact_matches_topic(
        self,
        question: dict,
        topic_query: str,
    ) -> tuple[bool, dict]:
        """
        Exact token topic matching at question level.

        This avoids:
            cat -> character
            cat -> category

        It only matches:
            cat -> cat
        """

        query = str(topic_query or "").strip().lower()

        if not query:
            return False, {
                "topic_query": topic_query,
                "matched": False,
                "reason": "empty_topic_query",
                "matched_terms": [],
            }

        search_text = self._question_to_search_text(question)
        search_text_lower = search_text.lower()

        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(search_text_lower)

        if not query_tokens:
            return False, {
                "topic_query": topic_query,
                "matched": False,
                "reason": "empty_query_tokens",
                "matched_terms": [],
            }

        if len(query_tokens) == 1:
            token = list(query_tokens)[0]
            matched = token in text_tokens

            return matched, {
                "topic_query": topic_query,
                "matched": matched,
                "reason": "exact_single_token_match",
                "matched_terms": [token] if matched else [],
            }

        phrase_pattern = r"\b" + re.escape(query) + r"\b"
        phrase_match = re.search(phrase_pattern, search_text_lower) is not None
        all_tokens_match = all(token in text_tokens for token in query_tokens)

        matched_terms = [
            token
            for token in query_tokens
            if token in text_tokens
        ]

        matched = phrase_match or all_tokens_match

        return matched, {
            "topic_query": topic_query,
            "matched": matched,
            "reason": "phrase_match_or_all_tokens_match",
            "matched_terms": matched_terms,
        }

    def _llm_question_matches_topic(
        self,
        question: dict,
        topic_query: str,
    ) -> dict:
        """
        Use LLM to judge whether a question is semantically relevant to topic_query.

        This is useful for broad topics:
            science, animal, history, medicine, physics, etc.

        The LLM should not judge answer correctness here.
        It only judges topic relevance.
        """

        if self.client is None:
            return {
                "is_relevant": False,
                "confidence": 0.0,
                "reason": "OpenAI client unavailable.",
                "matched_concepts": [],
            }

        question_text = question.get("question", "")
        context = self._stringify_context(question.get("context", ""))
        reference_answer = question.get("reference_answer", "")

        prompt = f"""
You are a topic relevance classifier for a quiz system.

Your task:
Decide whether the question is relevant to the user's requested topic.

Important:
- Do NOT answer the quiz question.
- Do NOT evaluate a user's answer.
- Only judge topic relevance.
- A question can be relevant even if the exact topic word does not appear.
- For example:
  topic "science" can include physics, chemistry, biology, astronomy, space,
  electricity, force, energy, materials, planets, animals, medicine, etc.
- topic "animal" can include mammals, birds, fish, cats, dogs, species,
  habitats, behavior, zoology, etc.
- But avoid false positives:
  topic "cat" should not match the word "character" unless the question is
  truly about cats.

Return ONLY valid JSON:
{{
  "is_relevant": true,
  "confidence": 0.0,
  "reason": "...",
  "matched_concepts": ["..."]
}}

User requested topic:
{topic_query}

Question:
{question_text}

Context:
{context[:4000]}

Reference answer:
{reference_answer}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.openai_chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON topic relevance classifier. Return only valid JSON.",
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

            confidence = self._safe_float(parsed.get("confidence", 0.0))
            is_relevant = bool(parsed.get("is_relevant", False))

            if confidence < self.topic_match_threshold:
                is_relevant = False

            return {
                "is_relevant": is_relevant,
                "confidence": round(confidence, 4),
                "reason": parsed.get("reason", ""),
                "matched_concepts": parsed.get("matched_concepts", []),
            }

        except Exception as error:
            return {
                "is_relevant": False,
                "confidence": 0.0,
                "reason": f"LLM topic relevance check failed: {error}",
                "matched_concepts": [],
            }

    def _question_to_search_text(
        self,
        question: dict,
    ) -> str:
        """
        Build searchable text from question fields.
        """

        parts = [
            question.get("question", ""),
            question.get("reference_answer", ""),
            question.get("answer", ""),
            question.get("context", ""),
        ]

        options = question.get("options") or question.get("choices")

        if options:
            parts.append(str(options))

        return " ".join(
            str(part)
            for part in parts
            if part is not None
        )

    # ============================================================
    # Profile and question extraction
    # ============================================================

    def _get_dataset_id(
        self,
        profile: dict,
    ) -> str:
        identity = profile.get("dataset_identity", {})

        return str(
            profile.get("dataset_id")
            or profile.get("dataset_key")
            or profile.get("name")
            or profile.get("profile_id")
            or identity.get("dataset_name")
            or identity.get("dataset_key")
            or "unknown_dataset"
        )

    def _get_profile_id(
        self,
        profile: dict,
    ) -> str:
        identity = profile.get("dataset_identity", {})

        return str(
            profile.get("profile_id")
            or profile.get("dataset_id")
            or profile.get("dataset_key")
            or profile.get("name")
            or identity.get("dataset_name")
            or identity.get("dataset_key")
            or "unknown_profile"
        )

    def _extract_questions_from_profile(
        self,
        profile: dict,
    ) -> list[dict]:
        """
        Extract questions from possible profile fields.

        This function is intentionally defensive because different versions
        of DatasetAnalyzer / MetadataPostprocessor may store question samples
        under different keys.

        It first checks common top-level keys.
        Then it checks common nested keys.
        Finally, it recursively searches for a list of dictionaries that look
        like question samples.
        """

        possible_keys = [
            "questions",
            "sample_questions",
            "samples",
            "examples",
            "data_samples",
            "normalized_samples",
            "dataset_samples",
            "sample_data",
            "adapted_samples",
        ]

        for key in possible_keys:
            value = profile.get(key)

            if self._looks_like_question_list(value):
                return [
                    self._normalize_question(raw_question, index)
                    for index, raw_question in enumerate(value)
                ]

        for nested_key in [
            "metadata",
            "dataset_metadata",
            "profile_metadata",
            "sample_metadata",
            "dataset_examples",
            "examples_metadata",
        ]:
            nested = profile.get(nested_key, {})

            if isinstance(nested, dict):
                for key in possible_keys:
                    value = nested.get(key)

                    if self._looks_like_question_list(value):
                        return [
                            self._normalize_question(raw_question, index)
                            for index, raw_question in enumerate(value)
                        ]

        found = self._recursive_find_question_list(profile)

        if found:
            return [
                self._normalize_question(raw_question, index)
                for index, raw_question in enumerate(found)
            ]

        return []

    def _looks_like_question_list(
        self,
        value,
    ) -> bool:
        """
        Check whether a value looks like a list of question samples.
        """

        if not isinstance(value, list) or not value:
            return False

        first_items = value[:3]

        for item in first_items:
            if not isinstance(item, dict):
                continue

            possible_question_keys = {
                "question",
                "query",
                "prompt",
                "input",
            }

            possible_answer_keys = {
                "answer",
                "reference_answer",
                "label",
                "target",
            }

            has_question = any(key in item for key in possible_question_keys)
            has_answer = any(key in item for key in possible_answer_keys)

            if has_question or has_answer:
                return True

        return False

    def _recursive_find_question_list(
        self,
        obj,
    ):
        """
        Recursively search for a list that looks like question samples.
        """

        if self._looks_like_question_list(obj):
            return obj

        if isinstance(obj, dict):
            for value in obj.values():
                found = self._recursive_find_question_list(value)

                if found:
                    return found

        if isinstance(obj, list):
            for value in obj:
                found = self._recursive_find_question_list(value)

                if found:
                    return found

        return None

    def _normalize_question(
        self,
        raw_question: Any,
        index: int,
    ) -> dict:
        """
        Normalize one raw question sample.
        """

        if not isinstance(raw_question, dict):
            return {
                "question_id": f"question_{index}",
                "question": str(raw_question),
                "answer": "",
                "reference_answer": "",
                "context": "",
                "metadata": {},
            }

        question = dict(raw_question)

        question_id = (
            question.get("question_id")
            or question.get("id")
            or question.get("qid")
            or f"question_{index}"
        )

        question_text = (
            question.get("question")
            or question.get("query")
            or question.get("prompt")
            or question.get("input")
            or ""
        )

        reference_answer = (
            question.get("reference_answer")
            or question.get("answer")
            or question.get("label")
            or question.get("target")
            or ""
        )

        question["question_id"] = str(question_id)
        question["question"] = str(question_text)
        question["reference_answer"] = str(reference_answer)

        if "metadata" not in question or not isinstance(question["metadata"], dict):
            question["metadata"] = {}

        return question

    def _attach_question_metadata(
        self,
        question: dict,
        profile: dict,
    ) -> dict:
        """
        Attach profile-level metadata to the question.
        """

        question = dict(question)

        metadata = question.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        question_type = self._get_profile_question_type(profile)
        answer_type = self._get_profile_answer_type(profile)

        metadata.setdefault("question_type", question_type)
        metadata.setdefault("answer_type", answer_type)

        question["metadata"] = metadata

        return question

    def _get_profile_question_type(
        self,
        profile: dict,
    ) -> str:
        qa = profile.get("question_answer_profile", {})

        return str(
            profile.get("question_type")
            or profile.get("q_type")
            or qa.get("question_type")
            or profile.get("semantic_profile", {}).get("question_type")
            or "unknown"
        )

    def _get_profile_answer_type(
        self,
        profile: dict,
    ) -> str:
        qa = profile.get("question_answer_profile", {})

        return str(
            profile.get("answer_type")
            or profile.get("a_type")
            or qa.get("answer_type")
            or profile.get("semantic_profile", {}).get("answer_type")
            or "unknown"
        )

    # ============================================================
    # Filtering
    # ============================================================

    def _filter_seen_questions(
        self,
        questions: list[dict],
        seen_question_ids: set[str],
        question_selection_plan: str,
    ) -> list[dict]:
        if question_selection_plan != "unseen_only":
            return questions

        if not seen_question_ids:
            return questions

        return [
            question
            for question in questions
            if self._get_question_id(question) not in seen_question_ids
        ]

    def _filter_question_types(
        self,
        questions: list[dict],
        requested_question_types: list[str],
    ) -> list[dict]:
        if not requested_question_types:
            return questions

        requested = {
            str(question_type).strip().lower()
            for question_type in requested_question_types
            if str(question_type).strip()
        }

        if not requested:
            return questions

        filtered = []

        for question in questions:
            metadata = question.get("metadata", {})
            question_type = (
                metadata.get("question_type")
                or question.get("question_type")
                or ""
            )

            if str(question_type).lower() in requested:
                filtered.append(question)

        return filtered

    # ============================================================
    # Selection and rounds
    # ============================================================

    def _select_questions(
        self,
        questions: list[dict],
        total_questions: int,
        collection_plan: str,
        random_seed: int,
    ) -> list[dict]:
        if not questions:
            return []

        total_questions = max(0, int(total_questions))

        if total_questions == 0:
            return []

        if "random" in collection_plan:
            random.seed(random_seed)
            questions_copy = list(questions)
            random.shuffle(questions_copy)
            return questions_copy[:total_questions]

        return list(questions)[:total_questions]

    def _build_rounds(
        self,
        selected_questions: list[dict],
        questions_per_round: int,
    ) -> list[dict]:
        rounds = []

        if not selected_questions:
            return rounds

        questions_per_round = max(1, int(questions_per_round))

        for start_index in range(0, len(selected_questions), questions_per_round):
            round_questions = selected_questions[
                start_index : start_index + questions_per_round
            ]

            rounds.append(
                {
                    "round_index": len(rounds) + 1,
                    "question_ids": [
                        self._get_question_id(question)
                        for question in round_questions
                    ],
                }
            )

        return rounds

    # ============================================================
    # Evaluator assignment
    # ============================================================

    def _assign_evaluators(
        self,
        profile: dict,
        selected_questions: list[dict],
    ) -> dict:
        evaluator_name = self._choose_evaluator_for_profile(profile)

        return {
            self._get_question_id(question): [evaluator_name]
            for question in selected_questions
        }

    def _choose_evaluator_for_profile(
        self,
        profile: dict,
    ) -> str:
        structure = profile.get("dataset_structure", {})
        qa = profile.get("question_answer_profile", {})
        semantic = profile.get("semantic_profile", {})
        basic = profile.get("basic_features", {})

        dataset_category = str(
            profile.get("dataset_category")
            or profile.get("category")
            or structure.get("dataset_category")
            or ""
        ).lower()

        question_type = str(
            profile.get("question_type")
            or qa.get("question_type")
            or ""
        ).lower()

        answer_type = str(
            profile.get("answer_type")
            or qa.get("answer_type")
            or ""
        ).lower()

        context_dependency = str(
            profile.get("context_dependency")
            or qa.get("context_dependency")
            or semantic.get("context_dependency")
            or ""
        ).lower()

        has_context = bool(
            profile.get("has_context")
            or basic.get("has_context")
            or profile.get("dataset_metadata", {}).get("has_context")
        )

        if "multiple_choice" in question_type or "multiple_choice" in answer_type:
            return "script_multiple_choice"

        if "boolean" in question_type or "true_false" in answer_type:
            return "script_true_false"

        if has_context or "context" in context_dependency or "hotpot" in dataset_category:
            return "context_llm_semantic"

        return "llm_semantic"

    # ============================================================
    # Utility
    # ============================================================

    def _deduplicate_questions(
        self,
        questions: list[dict],
    ) -> list[dict]:
        seen_ids = set()
        deduplicated = []

        for question in questions:
            question_id = self._get_question_id(question)

            if question_id not in seen_ids:
                seen_ids.add(question_id)
                deduplicated.append(question)

        return deduplicated

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