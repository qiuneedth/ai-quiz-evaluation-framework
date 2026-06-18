# src/core/evaluator_planner.py

"""
Evaluator Planner.

This module controls how questions are selected and delivered to the user.

Important:
Evaluator Planner does NOT select evaluators.
Evaluator Planner does NOT evaluate answers.
Evaluator Planner does NOT use AI.

It only controls the question delivery process.

It defines three types of plans:

1. Collection Plan
   How questions are collected and presented.
   Examples:
   - one_by_one_ordered
   - one_by_one_random
   - batch_ordered
   - batch_random
   - incremental
   - totally_random

2. Progress Plan
   How long the evaluation continues.
   Examples:
   - full_eval
   - partial_eval

3. Question Selection Plan
   Where questions are selected from.
   Examples:
   - unseen_only
   - all_questions
   - mixed_seen_unseen

This corresponds to the supervisor's diagram:

Generate a plan:
- collection plan
- progress plan
- q-selection plan
"""


import random


class EvaluatorPlanner:
    """
    EvaluatorPlanner provides deterministic question selection methods.

    The purpose of this class is to manage the interaction process
    between the system and the user.

    Example:
    - Select top 3 unseen questions.
    - Ask questions one by one.
    - Wait for user answers.
    - Stop after partial evaluation.
    """

    def build_planner_config(
        self,
        collection_strategy: str = "one_by_one_ordered",
        question_selection_strategy: str = "unseen_only",
        progress_strategy: str = "partial_eval",
        batch_size: int = 1,
        max_questions: int | None = None,
        repeat_probability: float = 0.0,
        min_incremental_batch_size: int = 1,
        max_incremental_batch_size: int = 5,
        random_seed: int | None = None,
    ) -> dict:
        """
        Build a planner configuration.

        Parameters:
        collection_strategy:
            Defines how questions are delivered.
            Supported values:
            - one_by_one_ordered
            - one_by_one_random
            - batch_ordered
            - batch_random
            - incremental
            - totally_random

        question_selection_strategy:
            Defines the question pool.
            Supported values:
            - unseen_only
            - all_questions
            - mixed_seen_unseen

        progress_strategy:
            Defines how much of the dataset should be evaluated.
            Supported values:
            - full_eval
            - partial_eval

        batch_size:
            Number of questions per batch for batch strategies.

        max_questions:
            Maximum number of questions for partial evaluation.

        repeat_probability:
            Used by mixed_seen_unseen.
            Example:
            0.2 means 20% chance to select from previously seen questions.

        min_incremental_batch_size / max_incremental_batch_size:
            Used by incremental strategy.
            Each round can randomly select a batch size in this range.

        random_seed:
            Used to make random selection reproducible.
        """

        return {
            "collection_strategy": collection_strategy,
            "question_selection_strategy": question_selection_strategy,
            "progress_strategy": progress_strategy,
            "batch_size": batch_size,
            "max_questions": max_questions,
            "repeat_probability": repeat_probability,
            "min_incremental_batch_size": min_incremental_batch_size,
            "max_incremental_batch_size": max_incremental_batch_size,
            "random_seed": random_seed,
        }

    def select_questions(
        self,
        questions: list[dict],
        answered_question_ids: set[str] | None,
        planner_config: dict,
    ) -> list[dict]:
        """
        Select questions according to the planner configuration.

        This is the main method used by Evaluator Instance.

        Steps:
        1. Build the candidate question pool.
        2. Apply collection strategy.
        3. Apply progress strategy.
        """

        if answered_question_ids is None:
            answered_question_ids = set()

        random_seed = planner_config.get("random_seed")

        if random_seed is not None:
            random.seed(random_seed)

        # Step 1: select question pool.
        candidate_questions = self._select_question_pool(
            questions=questions,
            answered_question_ids=answered_question_ids,
            question_selection_strategy=planner_config.get(
                "question_selection_strategy",
                "unseen_only",
            ),
            repeat_probability=planner_config.get("repeat_probability", 0.0),
        )

        # Step 2: apply ordering/random strategy.
        ordered_questions = self._apply_collection_strategy(
            questions=candidate_questions,
            collection_strategy=planner_config.get(
                "collection_strategy",
                "one_by_one_ordered",
            ),
        )

        # Step 3: apply full or partial progress strategy.
        final_questions = self._apply_progress_strategy(
            questions=ordered_questions,
            progress_strategy=planner_config.get(
                "progress_strategy",
                "partial_eval",
            ),
            max_questions=planner_config.get("max_questions"),
        )

        return final_questions

    def get_next_batch(
        self,
        remaining_questions: list[dict],
        planner_config: dict,
    ) -> list[dict]:
        """
        Get the next batch of questions.

        This method is useful when the evaluator instance runs interactively.

        For example:
        - one_by_one_ordered returns 1 question
        - batch_ordered returns batch_size questions
        - incremental returns a random number of questions between min and max
        """

        if not remaining_questions:
            return []

        collection_strategy = planner_config.get(
            "collection_strategy",
            "one_by_one_ordered",
        )

        batch_size = planner_config.get("batch_size", 1)

        if collection_strategy in ["one_by_one_ordered", "one_by_one_random"]:
            return remaining_questions[:1]

        if collection_strategy in ["batch_ordered", "batch_random"]:
            return remaining_questions[:batch_size]

        if collection_strategy == "incremental":
            min_size = planner_config.get("min_incremental_batch_size", 1)
            max_size = planner_config.get("max_incremental_batch_size", 5)

            dynamic_batch_size = random.randint(min_size, max_size)

            return remaining_questions[:dynamic_batch_size]

        if collection_strategy == "totally_random":
            dynamic_batch_size = random.randint(1, min(5, len(remaining_questions)))

            copied_questions = remaining_questions[:]
            random.shuffle(copied_questions)

            return copied_questions[:dynamic_batch_size]

        raise ValueError(f"Unknown collection_strategy: {collection_strategy}")

    def should_continue(
        self,
        answered_count: int,
        total_selected_count: int,
        planner_config: dict,
    ) -> bool:
        """
        Decide whether the evaluator instance should continue asking questions.

        full_eval:
            Continue until all selected questions are answered.

        partial_eval:
            Continue until max_questions is reached.
        """

        progress_strategy = planner_config.get("progress_strategy", "partial_eval")
        max_questions = planner_config.get("max_questions")

        if progress_strategy == "full_eval":
            return answered_count < total_selected_count

        if progress_strategy == "partial_eval":
            if max_questions is None:
                return answered_count < total_selected_count

            return answered_count < min(max_questions, total_selected_count)

        raise ValueError(f"Unknown progress_strategy: {progress_strategy}")

    def _select_question_pool(
        self,
        questions: list[dict],
        answered_question_ids: set[str],
        question_selection_strategy: str,
        repeat_probability: float,
    ) -> list[dict]:
        """
        Select the question pool.

        unseen_only:
            Only select questions that the user has not answered before.

        all_questions:
            Select from all questions.

        mixed_seen_unseen:
            Usually select unseen questions, but sometimes select seen
            questions according to repeat_probability.

            Example:
            repeat_probability = 0.2
            means 20% chance to include previous questions.
        """

        if question_selection_strategy == "unseen_only":
            return [
                question for question in questions
                if str(question.get("question_id")) not in answered_question_ids
            ]

        if question_selection_strategy == "all_questions":
            return questions

        if question_selection_strategy == "mixed_seen_unseen":
            unseen_questions = [
                question for question in questions
                if str(question.get("question_id")) not in answered_question_ids
            ]

            seen_questions = [
                question for question in questions
                if str(question.get("question_id")) in answered_question_ids
            ]

            # If there are no seen questions, fallback to unseen only.
            if not seen_questions:
                return unseen_questions

            selected_pool = []

            for question in unseen_questions:
                selected_pool.append(question)

            # Add seen questions with a certain probability.
            for question in seen_questions:
                if random.random() < repeat_probability:
                    selected_pool.append(question)

            return selected_pool

        raise ValueError(
            f"Unknown question_selection_strategy: {question_selection_strategy}"
        )

    def _apply_collection_strategy(
        self,
        questions: list[dict],
        collection_strategy: str,
    ) -> list[dict]:
        """
        Apply ordering or randomization.

        This method does not limit the number of questions.
        It only changes the order.
        """

        if collection_strategy in [
            "one_by_one_ordered",
            "batch_ordered",
            "incremental",
        ]:
            return questions

        if collection_strategy in [
            "one_by_one_random",
            "batch_random",
            "totally_random",
        ]:
            copied_questions = questions[:]
            random.shuffle(copied_questions)
            return copied_questions

        raise ValueError(f"Unknown collection_strategy: {collection_strategy}")

    def _apply_progress_strategy(
        self,
        questions: list[dict],
        progress_strategy: str,
        max_questions: int | None,
    ) -> list[dict]:
        """
        Apply full or partial evaluation strategy.

        full_eval:
            Use all candidate questions.

        partial_eval:
            Use max_questions if it is provided.
        """

        if progress_strategy == "full_eval":
            return questions

        if progress_strategy == "partial_eval":
            if max_questions is None:
                return questions

            return questions[:max_questions]

        raise ValueError(f"Unknown progress_strategy: {progress_strategy}")