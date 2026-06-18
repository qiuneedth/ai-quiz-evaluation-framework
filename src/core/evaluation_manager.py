# src/core/evaluation_manager.py

"""
Evaluation Manager.

This module selects suitable evaluators based on the selected Dataset Profile.

It corresponds to this part of the supervisor's framework:

Selected Profile
    ↓
Evaluator Manager
    ↓
Selected Evaluators
    ↓
Evaluation Plan Generator

Important distinction:

1. Profile Selector
   Selects which dataset/question profile should be used.

2. Evaluation Manager
   Selects which evaluators should be used for the selected profile.

3. Evaluation Plan Generator
   Converts selected evaluators into a detailed workflow with:
   - evaluator sequence
   - prompt templates
   - metrics
   - evaluator rules
   - scoring rubrics
   - aggregation strategy

The Evaluation Manager is not a free AI decision module.

AI can help in earlier stages:
- generating profile metadata
- suggesting candidate evaluators

But the final evaluator selection is controlled by manually defined
framework rules.

Example:
- If question_type is multiple_choice, select rule_based_evaluator.
- If context is required, select llm_context_evaluator and groundedness_evaluator.
- If reasoning is required, select completeness_evaluator and reasoning_quality_evaluator.
- If generated response exists, select relevance, groundedness, and completeness evaluators.
"""


class EvaluationManager:
    """
    Rule-controlled evaluator selector.

    It reads the Dataset Profile and decides which evaluators should be used.
    """

    def select_evaluators(self, profile: dict) -> list[str]:
        """
        Backward-compatible method.

        Returns only selected evaluator names.

        This keeps older code working.
        """

        result = self.select_evaluators_with_reasons(profile)
        return result["selected_evaluators"]

    def select_evaluators_with_reasons(self, profile: dict) -> dict:
        """
        Select evaluators and provide selection reasons.

        This method is useful for:
        - demo
        - report
        - explaining the manager logic to the supervisor

        Returns:
            {
                "selected_evaluators": [...],
                "selection_reasons": {...},
                "profile_candidates": [...],
                "rule_based_candidates": [...]
            }
        """

        selected = []
        selection_reasons = {}
        rule_based_candidates = []

        profile_candidates = self._get_profile_candidates(profile)

        qa_profile = profile.get("question_answer_profile", {})
        structure = profile.get("dataset_structure", {})

        question_type = qa_profile.get("question_type", "unknown")
        answer_type = qa_profile.get("answer_type", "unknown")
        reasoning_type = qa_profile.get("reasoning_type", "unknown")
        context_dependency = qa_profile.get("context_dependency", "unknown")

        has_context = structure.get("has_context", False)
        has_options = structure.get("has_options", False)
        has_generated_response = structure.get("has_generated_response", False)

        # ------------------------------------------------------------
        # Rule 1:
        # Multiple-choice or option-based questions should use
        # deterministic rule-based evaluation.
        # ------------------------------------------------------------
        if question_type == "multiple_choice" or has_options:
            self._add_evaluator(
                selected=selected,
                selection_reasons=selection_reasons,
                rule_based_candidates=rule_based_candidates,
                evaluator_name="rule_based_evaluator",
                reason=(
                    "Selected because the profile indicates multiple-choice "
                    "or option-based answers."
                ),
            )

        # ------------------------------------------------------------
        # Rule 2:
        # Boolean questions can use normalized exact matching.
        # They may also use semantic correctness evaluation if explanation
        # quality matters.
        # ------------------------------------------------------------
        if question_type == "boolean_reasoning" or answer_type == "boolean":
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "rule_based_evaluator",
                "Selected because boolean answers can be checked by normalized exact matching.",
            )

            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "llm_evaluator",
                "Selected because boolean reasoning may require semantic correctness evaluation.",
            )

        # ------------------------------------------------------------
        # Rule 3:
        # Factoid QA can use keyword coverage and semantic correctness.
        # ------------------------------------------------------------
        if question_type == "factoid_qa":
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "keyword_evaluator",
                "Selected because factoid QA can be evaluated with keyword coverage as a baseline.",
            )

            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "llm_evaluator",
                "Selected because semantically correct answers may use different wording from the reference answer.",
            )

        # ------------------------------------------------------------
        # Rule 4:
        # Context-grounded QA requires context correctness and groundedness.
        # ------------------------------------------------------------
        if (
            question_type == "context_grounded_qa"
            or has_context
            or context_dependency in [
                "provided_context_required",
                "retrieved_context_required",
            ]
        ):
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "llm_context_evaluator",
                "Selected because the answer must be evaluated against provided or retrieved context.",
            )

            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "groundedness_evaluator",
                "Selected because factual claims in the answer must be supported by context.",
            )

        # ------------------------------------------------------------
        # Rule 5:
        # Reasoning or open-ended QA should include completeness and
        # reasoning quality.
        # ------------------------------------------------------------
        if question_type in [
            "multi_hop_reasoning",
            "open_ended_reasoning",
        ] or reasoning_type in [
            "multi_hop_reasoning",
            "open_ended_reasoning",
            "contextual_reasoning",
            "logical_reasoning",
            "commonsense_reasoning",
        ]:
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "completeness_evaluator",
                "Selected because reasoning or open-ended answers may require multiple required points.",
            )

            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "reasoning_quality_evaluator",
                "Selected because the profile indicates reasoning is required.",
            )

        # ------------------------------------------------------------
        # Rule 6:
        # RAG response evaluation or generated answers should include
        # relevance, groundedness, and completeness.
        # ------------------------------------------------------------
        if question_type == "rag_response_evaluation" or has_generated_response:
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "relevance_evaluator",
                "Selected because generated responses should directly address the user question.",
            )

            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "groundedness_evaluator",
                "Selected because generated responses must be grounded in retrieved documents.",
            )

            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "completeness_evaluator",
                "Selected because generated responses should cover required information.",
            )

        # ------------------------------------------------------------
        # Rule 7:
        # The Dataset Profile may contain candidate evaluators suggested
        # by an analyzer or an LLM-assisted profile generator.
        #
        # These are considered as candidates, but the manager still
        # controls final selection.
        # ------------------------------------------------------------
        for candidate in profile_candidates:
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                candidate,
                (
                    "Selected because it was included in the Dataset Profile "
                    "as a possible evaluator candidate."
                ),
            )

        # ------------------------------------------------------------
        # Rule 8:
        # Fallback evaluator.
        # ------------------------------------------------------------
        if not selected:
            self._add_evaluator(
                selected,
                selection_reasons,
                rule_based_candidates,
                "llm_evaluator",
                "Selected as fallback because no more specific evaluator rule matched.",
            )

        return {
            "selected_evaluators": selected,
            "selection_reasons": selection_reasons,
            "profile_candidates": profile_candidates,
            "rule_based_candidates": rule_based_candidates,
        }

    def _get_profile_candidates(self, profile: dict) -> list[str]:
        """
        Read evaluator candidates stored in Dataset Profile.

        This supports the hybrid idea:
        - AI or analyzer can suggest candidate evaluators in the profile.
        - Manager rules still make the final controlled selection.
        """

        evaluation_profile = profile.get("evaluation_profile", {})
        candidates = evaluation_profile.get("evaluation_candidates", [])

        if candidates is None:
            return []

        return candidates

    def _add_evaluator(
        self,
        selected: list[str],
        selection_reasons: dict,
        rule_based_candidates: list[str],
        evaluator_name: str,
        reason: str,
    ) -> None:
        """
        Add one evaluator and store the reason.

        This method prevents duplicate evaluators while preserving reasons.
        """

        if evaluator_name not in selected:
            selected.append(evaluator_name)

        if evaluator_name not in rule_based_candidates:
            rule_based_candidates.append(evaluator_name)

        if evaluator_name not in selection_reasons:
            selection_reasons[evaluator_name] = []

        if reason not in selection_reasons[evaluator_name]:
            selection_reasons[evaluator_name].append(reason)