# src/core/evaluator_instance.py

"""
Evaluator Instance.

This module defines the runtime process of one evaluation session.

An Evaluator Instance connects:
- one user
- one selected dataset
- one dataset profile
- one evaluation plan
- one evaluator planner config
- user answers
- evaluation results
- user activity logs

Important:
Evaluator Instance is not the same as a single evaluator class.

A single evaluator class may check correctness or groundedness.
Evaluator Instance manages the whole session process.
"""


from datetime import datetime


class EvaluatorInstance:
    """
    Represents one running evaluation session.

    Main responsibilities:
    1. Store session state.
    2. Select questions using EvaluatorPlanner.
    3. Provide questions to the user.
    4. Receive user answers.
    5. Send answers to EvaluatorExecutor.
    6. Store evaluation results.
    7. Write user activity logs.
    """

    def __init__(
        self,
        instance_id: str,
        user_id: str,
        dataset_profile: dict,
        evaluation_plan: dict,
        planner_config: dict,
        evaluator_planner,
        evaluator_executor=None,
        activity_logger=None,
    ):
        """
        Initialize one evaluator instance.

        Parameters:
        instance_id:
            Unique id of this evaluation session.

        user_id:
            The user who is taking the evaluation.

        dataset_profile:
            The selected dataset profile.

        evaluation_plan:
            The generated evaluation plan.

        planner_config:
            The collection/progress/question-selection configuration.

        evaluator_planner:
            Object responsible for selecting questions.

        evaluator_executor:
            Object responsible for running evaluators.
            It can be None in early testing.

        activity_logger:
            Object responsible for recording user activities.
            It can be None in early testing.
        """

        self.instance_id = instance_id
        self.user_id = user_id
        self.dataset_profile = dataset_profile
        self.evaluation_plan = evaluation_plan
        self.planner_config = planner_config
        self.evaluator_planner = evaluator_planner
        self.evaluator_executor = evaluator_executor
        self.activity_logger = activity_logger

        # Runtime state.
        self.status = "initialized"
        self.started_at = None
        self.finished_at = None

        # Question state.
        self.selected_questions = []
        self.remaining_questions = []
        self.current_batch = []

        # User and evaluation results.
        self.user_answers = []
        self.evaluation_results = []

        # Track answered question ids.
        self.answered_question_ids = set()

    def start(self, answered_question_ids: set[str] | None = None) -> list[dict]:
        """
        Start the evaluator instance.

        This method selects questions according to EvaluatorPlanner.

        It returns the first batch of questions that should be given to the user.
        """

        if answered_question_ids is None:
            answered_question_ids = set()

        self.status = "running"
        self.started_at = datetime.utcnow().isoformat()

        # Get sample questions from Dataset Profile.
        questions = self.dataset_profile.get(
            "sample_profile",
            {},
        ).get(
            "sample_questions",
            [],
        )

        # Select questions according to planner config.
        self.selected_questions = self.evaluator_planner.select_questions(
            questions=questions,
            answered_question_ids=answered_question_ids,
            planner_config=self.planner_config,
        )

        # At the beginning, all selected questions are remaining.
        self.remaining_questions = self.selected_questions[:]

        # Log instance start.
        self._log_event(
            event_type="evaluator_instance_started",
            status="success",
            metadata={
                "selected_question_count": len(self.selected_questions),
                "planner_config": self.planner_config,
            },
        )

        # Return first batch.
        return self.next_batch()

    def next_batch(self) -> list[dict]:
        """
        Return the next batch of questions.

        The batch size depends on the collection plan.
        """

        if not self.remaining_questions:
            return []

        self.current_batch = self.evaluator_planner.get_next_batch(
            remaining_questions=self.remaining_questions,
            planner_config=self.planner_config,
        )

        return self.current_batch

    def submit_answers(self, answers: list[dict]) -> list[dict]:
        """
        Submit user answers for the current batch.

        Expected answer format:
        [
            {
                "question_id": "q1",
                "user_answer": "Paris"
            }
        ]

        After answers are submitted:
        1. Store answers.
        2. Remove answered questions from remaining questions.
        3. Optionally evaluate them using evaluator_executor.
        4. Log activity.
        """

        batch_results = []

        for answer_item in answers:
            question_id = str(answer_item.get("question_id"))
            user_answer = answer_item.get("user_answer")

            question = self._find_question_by_id(question_id)

            if question is None:
                continue

            # Store user answer.
            answer_record = {
                "question_id": question_id,
                "question": question.get("question"),
                "user_answer": user_answer,
                "submitted_at": datetime.utcnow().isoformat(),
            }

            self.user_answers.append(answer_record)
            self.answered_question_ids.add(question_id)

            # Run evaluator executor if it exists.
            if self.evaluator_executor is not None:
                evaluation_result = self.evaluator_executor.execute(
                    evaluation_plan=self.evaluation_plan,
                    question=question,
                    user_answer=user_answer,
                )
            else:
                # Placeholder result for early testing.
                evaluation_result = {
                    "question_id": question_id,
                    "final_score": None,
                    "status": "not_evaluated_executor_missing",
                }

            self.evaluation_results.append(evaluation_result)
            batch_results.append(evaluation_result)

            # Log question answered.
            self._log_event(
                event_type="question_answered",
                status="success",
                question_id=question_id,
                question_text=question.get("question"),
                user_answer=user_answer,
                final_score=evaluation_result.get("final_score"),
                metadata={
                    "evaluation_result": evaluation_result,
                },
            )

        # Remove answered questions from remaining questions.
        self._remove_answered_questions_from_remaining()

        return batch_results

    def should_continue(self) -> bool:
        """
        Check whether the instance should continue.

        This delegates progress control to EvaluatorPlanner.
        """

        return self.evaluator_planner.should_continue(
            answered_count=len(self.user_answers),
            total_selected_count=len(self.selected_questions),
            planner_config=self.planner_config,
        )

    def finish(self) -> dict:
        """
        Finish the evaluator instance.

        This method closes the session and returns a summary.
        """

        self.status = "finished"
        self.finished_at = datetime.utcnow().isoformat()

        summary = {
            "instance_id": self.instance_id,
            "user_id": self.user_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "selected_question_count": len(self.selected_questions),
            "answered_question_count": len(self.user_answers),
            "evaluation_result_count": len(self.evaluation_results),
        }

        self._log_event(
            event_type="evaluator_instance_finished",
            status="success",
            metadata=summary,
        )

        return summary

    def _find_question_by_id(self, question_id: str) -> dict | None:
        """
        Find a question from selected questions by question_id.
        """

        for question in self.selected_questions:
            if str(question.get("question_id")) == question_id:
                return question

        return None

    def _remove_answered_questions_from_remaining(self) -> None:
        """
        Remove answered questions from the remaining question list.
        """

        self.remaining_questions = [
            question for question in self.remaining_questions
            if str(question.get("question_id")) not in self.answered_question_ids
        ]

    def _log_event(
        self,
        event_type: str,
        status: str,
        question_id: str | None = None,
        question_text: str | None = None,
        user_answer: str | None = None,
        final_score: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Write one user activity event if activity_logger is available.
        """

        if self.activity_logger is None:
            return

        self.activity_logger.log_event(
            user_id=self.user_id,
            event_type=event_type,
            dataset_id=self.dataset_profile.get(
                "dataset_identity",
                {},
            ).get(
                "dataset_name",
            ),
            plan_id=self.evaluation_plan.get(
                "plan_identity",
                {},
            ).get(
                "plan_id",
            ),
            evaluator_instance_id=self.instance_id,
            question_id=question_id,
            question_text=question_text,
            user_answer=user_answer,
            final_score=final_score,
            status=status,
            metadata=metadata or {},
        )