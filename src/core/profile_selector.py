# src/core/profile_selector.py

"""
Profile Selector.

This module selects the most suitable Dataset Profile according to user request.

This corresponds to the supervisor's diagram:

User Request
    ↓
Profile Selector
    ↓
Selected Profile(s)
    ↓
Evaluator Manager

Important:
Profile Selector does NOT select evaluators.
Profile Selector only selects dataset profiles.

Evaluator Manager will later use the selected profile to choose evaluators.

This module supports two selection modes:

1. Direct dataset selection
   If the user explicitly provides dataset_key, for example "hotpotqa",
   the selector directly selects that dataset profile.

2. Metadata-based selection
   If the user only provides requirements, for example:
   - context-based questions
   - multiple-choice questions
   - open-ended questions
   then the selector ranks available profiles by metadata matching.
"""


class ProfileSelector:
    """
    Select dataset profiles based on user request.

    A user request can contain:
    - dataset_key
    - question_type
    - requires_context
    - answer_type
    - reasoning_type
    - difficulty_level
    - domain
    """

    def select_profile(
        self,
        user_request: dict,
        available_profiles: list[dict],
    ) -> dict:
        """
        Select the best profile from available profiles.

        Parameters:
            user_request:
                A structured request dictionary.

                Example 1:
                {
                    "dataset_key": "hotpotqa"
                }

                Example 2:
                {
                    "question_type": "multiple_choice",
                    "requires_context": false
                }

                Example 3:
                {
                    "requires_context": true,
                    "reasoning_type": "multi_hop_reasoning"
                }

            available_profiles:
                A list of Dataset Profile dictionaries.

        Returns:
            A dictionary containing:
            - selected_profile
            - selection_reason
            - selection_score
            - all_candidates
        """

        # ------------------------------------------------------------
        # Case 1:
        # User explicitly selects a dataset.
        # This is the simplest and most reliable case.
        # ------------------------------------------------------------
        dataset_key = user_request.get("dataset_key")

        if dataset_key:
            return self._select_by_dataset_key(
                dataset_key=dataset_key,
                available_profiles=available_profiles,
            )

        # ------------------------------------------------------------
        # Case 2:
        # User provides requirements instead of a dataset name.
        # In this case, we rank profiles by metadata matching score.
        # ------------------------------------------------------------
        ranked_profiles = []

        for profile in available_profiles:
            score, reasons = self._score_profile_match(
                user_request=user_request,
                profile=profile,
            )

            ranked_profiles.append(
                {
                    "profile": profile,
                    "score": score,
                    "reasons": reasons,
                }
            )

        ranked_profiles.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        if not ranked_profiles:
            raise ValueError("No available profiles provided.")

        best = ranked_profiles[0]

        return {
            "selected_profile": best["profile"],
            "selection_reason": best["reasons"],
            "selection_score": best["score"],
            "all_candidates": [
                {
                    "dataset_name": candidate["profile"]["dataset_identity"]["dataset_name"],
                    "score": candidate["score"],
                    "reasons": candidate["reasons"],
                }
                for candidate in ranked_profiles
            ],
        }

    def _select_by_dataset_key(
        self,
        dataset_key: str,
        available_profiles: list[dict],
    ) -> dict:
        """
        Select profile directly by dataset name.

        This corresponds to the user request:
        "I want to work on this dataset: A"
        """

        for profile in available_profiles:
            dataset_name = profile.get(
                "dataset_identity",
                {},
            ).get(
                "dataset_name",
            )

            if dataset_name == dataset_key:
                return {
                    "selected_profile": profile,
                    "selection_reason": [
                        f"User explicitly selected dataset_key='{dataset_key}'."
                    ],
                    "selection_score": 1.0,
                    "all_candidates": [
                        {
                            "dataset_name": dataset_name,
                            "score": 1.0,
                            "reasons": [
                                "Exact dataset match."
                            ],
                        }
                    ],
                }

        raise ValueError(
            f"No profile found for dataset_key='{dataset_key}'."
        )

    def _score_profile_match(
        self,
        user_request: dict,
        profile: dict,
    ) -> tuple[float, list[str]]:
        """
        Score how well one profile matches the user request.

        This is a manually designed rule-based matching method.

        It does not use AI.
        It compares user requirements with profile metadata.
        """

        score = 0.0
        reasons = []

        dataset_identity = profile.get("dataset_identity", {})
        dataset_structure = profile.get("dataset_structure", {})
        qa_profile = profile.get("question_answer_profile", {})
        semantic_profile = profile.get("semantic_profile", {})

        dataset_name = dataset_identity.get("dataset_name", "unknown")

        # ------------------------------------------------------------
        # Match question type.
        # Example:
        # User wants multiple_choice.
        # Profile question_type is multiple_choice.
        # ------------------------------------------------------------
        requested_question_type = user_request.get("question_type")

        if requested_question_type:
            profile_question_type = qa_profile.get("question_type")

            if requested_question_type == profile_question_type:
                score += 3.0
                reasons.append(
                    f"Question type matches: {requested_question_type}."
                )

        # ------------------------------------------------------------
        # Match context requirement.
        # Example:
        # User wants context-based dataset.
        # Profile has_context = True.
        # ------------------------------------------------------------
        if "requires_context" in user_request:
            requested_context = user_request.get("requires_context")
            profile_has_context = dataset_structure.get("has_context", False)

            if requested_context == profile_has_context:
                score += 2.0
                reasons.append(
                    f"Context requirement matches: requires_context={requested_context}."
                )

        # ------------------------------------------------------------
        # Match answer type.
        # Example:
        # User wants boolean answer.
        # Profile answer_type is boolean.
        # ------------------------------------------------------------
        requested_answer_type = user_request.get("answer_type")

        if requested_answer_type:
            profile_answer_type = qa_profile.get("answer_type")

            if requested_answer_type == profile_answer_type:
                score += 2.0
                reasons.append(
                    f"Answer type matches: {requested_answer_type}."
                )

        # ------------------------------------------------------------
        # Match reasoning type.
        # Example:
        # User wants multi-hop reasoning.
        # Profile reasoning_type is multi_hop_reasoning.
        # ------------------------------------------------------------
        requested_reasoning_type = user_request.get("reasoning_type")

        if requested_reasoning_type:
            profile_reasoning_type = qa_profile.get("reasoning_type")

            if requested_reasoning_type == profile_reasoning_type:
                score += 2.0
                reasons.append(
                    f"Reasoning type matches: {requested_reasoning_type}."
                )

        # ------------------------------------------------------------
        # Match difficulty.
        # Example:
        # User wants medium questions.
        # Profile difficulty_level is medium.
        # ------------------------------------------------------------
        requested_difficulty = user_request.get("difficulty_level")

        if requested_difficulty:
            profile_difficulty = qa_profile.get("difficulty_level")

            if requested_difficulty == profile_difficulty:
                score += 1.0
                reasons.append(
                    f"Difficulty level matches: {requested_difficulty}."
                )

        # ------------------------------------------------------------
        # Match domain.
        # Example:
        # User wants science questions.
        # Profile domain is science.
        # ------------------------------------------------------------
        requested_domain = user_request.get("domain")

        if requested_domain:
            profile_domain = semantic_profile.get("domain")

            if requested_domain == profile_domain:
                score += 1.0
                reasons.append(
                    f"Domain matches: {requested_domain}."
                )

        # ------------------------------------------------------------
        # If nothing matches, still add a reason for transparency.
        # ------------------------------------------------------------
        if not reasons:
            reasons.append(
                f"No strong metadata match found for profile '{dataset_name}'."
            )

        return score, reasons