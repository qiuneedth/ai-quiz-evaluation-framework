# src/core/request_selector.py

"""
Request Selector.

This module selects a dataset profile based on a structured user request.

Important:
The selector does NOT only receive a dataset name.

It receives a user_request dictionary, for example:

{
    "dataset_key": null,
    "topic_query": "science",
    "question_types": ["multiple_choice"],
    "total_questions": 10,
    "planner": {...}
}

Selector responsibility:
1. If dataset_key is provided, select that dataset profile directly.
2. If dataset_key is missing, search available dataset profiles by:
   - topic_query
   - domain
   - question_types
   - sample question text / context
3. If multiple profiles match, show candidates to the user.
4. Return selected_profile and completed_user_request.

This class fixes the ImportError:
    from src.core.request_selector import RequestSelector
"""


class RequestSelector:
    """
    Select a dataset profile according to a user request dictionary.
    """

    def select(
        self,
        user_request: dict,
        available_profiles: list[dict],
        interactive: bool = True,
    ) -> dict:
        """
        Select a profile based on user request.

        Args:
            user_request:
                Normalized user request dictionary.

            available_profiles:
                List of dataset profiles.

            interactive:
                If True, ask user to choose when multiple profiles match.

        Returns:
            {
                "completed_user_request": {...},
                "selected_profile": {...},
                "candidate_profiles": [...],
                "selection_reason": [...]
            }
        """

        dataset_key = user_request.get("dataset_key")

        # Case 1: user explicitly selected a dataset.
        if dataset_key:
            return self._select_by_dataset_key(
                user_request=user_request,
                available_profiles=available_profiles,
                dataset_key=dataset_key,
            )

        # Case 2: user did not select dataset.
        # Search candidates by topic/domain/question type/sample content.
        candidates = self._find_candidate_profiles(
            user_request=user_request,
            available_profiles=available_profiles,
        )

        if not candidates:
            raise ValueError(
                "No matching dataset/profile found for this user request. "
                f"Request: {user_request}. "
                "Try increasing sample_size, adding a matching dataset, or using a broader topic."
            )

        if len(candidates) == 1:
            selected_profile = candidates[0]
        else:
            if interactive:
                selected_profile = self._ask_user_to_select_profile(candidates)
            else:
                selected_profile = candidates[0]

        completed_user_request = dict(user_request)
        completed_user_request["dataset_key"] = selected_profile["dataset_identity"]["dataset_name"]

        return {
            "completed_user_request": completed_user_request,
            "selected_profile": selected_profile,
            "candidate_profiles": candidates,
            "selection_reason": [
                "User did not provide dataset_key.",
                "Selector searched available profiles using topic/domain/question type/sample content.",
                f"Selected dataset_key='{completed_user_request['dataset_key']}'.",
            ],
        }

    def _select_by_dataset_key(
        self,
        user_request: dict,
        available_profiles: list[dict],
        dataset_key: str,
    ) -> dict:
        """
        Select profile directly by dataset_key.
        """

        for profile in available_profiles:
            profile_name = profile.get("dataset_identity", {}).get("dataset_name")

            if profile_name == dataset_key:
                completed_user_request = dict(user_request)
                completed_user_request["dataset_key"] = dataset_key

                return {
                    "completed_user_request": completed_user_request,
                    "selected_profile": profile,
                    "candidate_profiles": [profile],
                    "selection_reason": [
                        f"User explicitly selected dataset_key='{dataset_key}'."
                    ],
                }

        raise ValueError(
            f"dataset_key='{dataset_key}' was not found in available profiles."
        )

    def _find_candidate_profiles(
        self,
        user_request: dict,
        available_profiles: list[dict],
    ) -> list[dict]:
        """
        Find candidate profiles matching user request.

        Important:
        The selector score is only an internal ranking value.
        It is NOT an evaluation score.
        """

        topic_query = self._normalize(user_request.get("topic_query") or user_request.get("topic"))
        domain = self._normalize(user_request.get("domain"))
        requested_question_types = user_request.get("question_types", [])

        candidates = []

        for profile in available_profiles:
            score = self._profile_match_score(
                profile=profile,
                topic_query=topic_query,
                domain=domain,
                requested_question_types=requested_question_types,
            )

            if score > 0:
                copied_profile = dict(profile)
                copied_profile["_selector_score"] = score
                candidates.append(copied_profile)

        candidates.sort(
            key=lambda item: item.get("_selector_score", 0.0),
            reverse=True,
        )

        return candidates

    def _profile_match_score(
        self,
        profile: dict,
        topic_query: str | None,
        domain: str | None,
        requested_question_types: list[str] | None,
    ) -> float:
        """
        Calculate profile match score.

        This is rule-based in the first prototype.

        Matching sources:
        - dataset name
        - dataset category
        - semantic profile
        - question type
        - reasoning type
        - sample question text
        - sample context text

        The score is only for candidate ranking.
        """

        score = 0.0

        requested_question_types = requested_question_types or []

        identity = profile.get("dataset_identity", {})
        structure = profile.get("dataset_structure", {})
        qa = profile.get("question_answer_profile", {})
        semantic = profile.get("semantic_profile", {})
        evaluation = profile.get("evaluation_profile", {})
        sample_profile = profile.get("sample_profile", {})
        notes = profile.get("notes", "")

        sample_questions = sample_profile.get("sample_questions", [])

        sample_text_parts = []

        for sample in sample_questions:
            sample_text_parts.append(str(sample.get("question", "")))
            sample_text_parts.append(str(sample.get("context", "")))
            sample_text_parts.append(str(sample.get("reference_answer", "")))

        searchable_parts = [
            identity.get("dataset_name", ""),
            structure.get("dataset_category", ""),
            qa.get("question_type", ""),
            qa.get("answer_type", ""),
            qa.get("reasoning_type", ""),
            semantic.get("domain", ""),
            semantic.get("topic", ""),
            " ".join(evaluation.get("possible_metrics", [])),
            notes,
            " ".join(sample_text_parts),
        ]

        searchable_text = " ".join(str(part) for part in searchable_parts).lower()

        profile_question_type = str(qa.get("question_type", "")).lower()

        # Topic / keyword match.
        if topic_query and topic_query in searchable_text:
            score += 3.0

        # Domain match.
        if domain and domain in searchable_text:
            score += 2.0

        # Requested question type match.
        if requested_question_types:
            normalized_requested = [
                item.lower()
                for item in requested_question_types
            ]

            if profile_question_type in normalized_requested:
                score += 3.0

        # Simple synonym matching.
        if topic_query == "mathematics":
            math_words = [
                "math",
                "mathematics",
                "arithmetic",
                "algebra",
                "geometry",
                "calculation",
                "quantitative",
            ]
            if any(word in searchable_text for word in math_words):
                score += 2.0

        if topic_query == "science":
            science_words = [
                "science",
                "physics",
                "chemistry",
                "biology",
                "scientific",
            ]
            if any(word in searchable_text for word in science_words):
                score += 2.0

        if topic_query == "history":
            history_words = [
                "history",
                "historical",
                "war",
                "president",
                "ancient",
                "empire",
            ]
            if any(word in searchable_text for word in history_words):
                score += 2.0

        if topic_query == "moon":
            moon_words = [
                "moon",
                "lunar",
                "apollo",
                "satellite",
            ]
            if any(word in searchable_text for word in moon_words):
                score += 2.0

        if topic_query == "medicine":
            medical_words = [
                "medical",
                "medicine",
                "biomedical",
                "covid",
                "health",
                "clinical",
            ]
            if any(word in searchable_text for word in medical_words):
                score += 2.0

        return score

    def _ask_user_to_select_profile(
        self,
        candidates: list[dict],
    ) -> dict:
        """
        Show candidate profiles and let the user select one.

        This implements the supervisor's idea:
        If the user asks for a topic, bring matching profiles to the user,
        and the user selects one.
        """

        print("\nMatching dataset profiles found:")
        print("=" * 60)

        for index, profile in enumerate(candidates, start=1):
            identity = profile.get("dataset_identity", {})
            structure = profile.get("dataset_structure", {})
            semantic = profile.get("semantic_profile", {})
            qa = profile.get("question_answer_profile", {})

            print(f"{index}. {identity.get('dataset_name')}")
            print(f"   category:      {structure.get('dataset_category')}")
            print(f"   domain:        {semantic.get('domain')}")
            print(f"   topic:         {semantic.get('topic')}")
            print(f"   question type: {qa.get('question_type')}")
            print(f"   reasoning:     {qa.get('reasoning_type')}")
            print()

        while True:
            choice = input("Select dataset/profile number: ").strip()

            if choice.isdigit():
                index = int(choice)

                if 1 <= index <= len(candidates):
                    return candidates[index - 1]

            print("Invalid choice. Please enter a valid number.")

    def _normalize(
        self,
        value,
    ) -> str | None:
        """
        Normalize request value.
        """

        if value is None:
            return None

        value = str(value).strip().lower()

        if not value:
            return None

        return value

