# src/core/quiz_session_manager.py

"""
Quiz Session Manager.

This module runs the quiz session after a quiz plan has been generated.

Runtime workflow:
1. Show current question.
2. If the question is context-dependent, show the provided context before answering.
3. User may request one hint before answering.
4. Receive user answer or generate AI answer.
5. Run assigned evaluator(s).
6. Show immediate feedback and detailed explanation.
7. Generate progressive report.
8. Continue until the session ends.
9. Return session result for final report generation.

Important:
- Hint happens before answer submission.
- Hint is allowed only once per question.
- Hint does not change the raw evaluator score.
- Hint should be passed to EvaluationEngine before explanation is generated.
- Hint should be stored in question_result and final_report.
- For context-dependent datasets, context should be displayed before collecting the answer.
"""


from __future__ import annotations

from datetime import datetime
from typing import Any

from src.core.hint_generator import HintGenerator


class QuizSessionManager:
    """
    Runtime quiz session manager.
    """

    def __init__(
        self,
        evaluation_engine,
        answer_provider,
        logger=None,
        hint_generator=None,
    ):
        self.evaluation_engine = evaluation_engine
        self.answer_provider = answer_provider
        self.logger = logger
        self.hint_generator = hint_generator or HintGenerator()

    def run_session(
        self,
        quiz_plan: dict,
        user_id: str = "demo_user",
    ) -> dict:
        """
        Run the quiz session based on a quiz plan.
        """

        if quiz_plan.get("plan_status") == "no_topic_match":
            return {
                "session_id": f"session_{quiz_plan.get('quiz_plan_id')}",
                "dataset_id": quiz_plan.get("dataset_id"),
                "profile_id": quiz_plan.get("profile_id"),
                "quiz_plan_id": quiz_plan.get("quiz_plan_id"),
                "user_id": user_id,
                "answer_source": quiz_plan.get("session_config", {}).get("answer_source"),
                "total_questions": 0,
                "answered_questions": 0,
                "not_answered_questions": 0,
                "answered_question_ids": [],
                "not_answered_question_ids": [],
                "final_score": 0.0,
                "question_results": [],
                "progressive_reports": [],
                "evaluator_usage": {},
                "hint_summary": self._build_hint_summary([]),
                "stop_reason": quiz_plan.get("stop_reason"),
                "created_at": datetime.utcnow().isoformat(),
            }

        session_id = f"session_{quiz_plan.get('quiz_plan_id')}"
        dataset_id = quiz_plan.get("dataset_id")
        profile_id = quiz_plan.get("profile_id")

        session_config = quiz_plan.get("session_config", {})
        answer_source = session_config.get("answer_source", "manual")

        selected_questions = quiz_plan.get("selected_questions", [])
        assigned_evaluators = quiz_plan.get("assigned_evaluators", {})
        rounds = quiz_plan.get("rounds", [])

        question_map = {
            self._get_question_id(question): question
            for question in selected_questions
        }

        answered_question_ids = []
        question_results = []
        progressive_reports = []

        self._safe_log(
            event_type="quiz_session_started",
            user_id=user_id,
            dataset_id=dataset_id,
            profile_id=profile_id,
            plan_id=quiz_plan.get("quiz_plan_id"),
            evaluator_instance_id=session_id,
            selected_evaluators=self._collect_all_evaluators(assigned_evaluators),
            metadata={
                "quiz_plan_id": quiz_plan.get("quiz_plan_id"),
                "rounds": rounds,
                "session_config": session_config,
                "planner_config": quiz_plan.get("planner_config", {}),
                "selection_summary": quiz_plan.get("selection_summary", {}),
            },
        )

        for round_info in rounds:
            round_index = round_info.get("round_index")
            round_question_ids = round_info.get("question_ids", [])

            print(f"\nROUND {round_index}")
            print("-" * 60)
            print(f"Questions in this round: {round_question_ids}")

            round_results = []

            for question_id in round_question_ids:
                question = question_map.get(str(question_id))

                if question is None:
                    continue

                evaluators = assigned_evaluators.get(str(question_id), [])

                answer_payload = self._get_answer_payload(
                    question=question,
                    answer_source=answer_source,
                )

                user_answer = answer_payload.get("user_answer", "")
                hint_used = answer_payload.get("hint_used", False)
                hint_text = answer_payload.get("hint_text")
                hint_payload = answer_payload.get("hint_payload")

                # ------------------------------------------------------------
                # Critical fix:
                # Pass hint_used and hint_text INTO evaluation_engine.
                # The evaluation engine generates answer_explanation,
                # so it must know hint usage before explanation generation.
                # ------------------------------------------------------------
                question_result = self.evaluation_engine.evaluate_question(
                    question=question,
                    user_answer=user_answer,
                    assigned_evaluators=evaluators,
                    answer_source=answer_source,
                    round_index=round_index,
                    hint_used=hint_used,
                    hint_text=hint_text,
                )

                question_result["hint_used"] = hint_used
                question_result["hint_text"] = hint_text
                question_result["hint_payload"] = hint_payload

                question_results.append(question_result)
                round_results.append(question_result)
                answered_question_ids.append(str(question_id))

                self._print_immediate_feedback(question_result)

                self._safe_log(
                    event_type="question_evaluated",
                    user_id=user_id,
                    dataset_id=dataset_id,
                    profile_id=profile_id,
                    plan_id=quiz_plan.get("quiz_plan_id"),
                    evaluator_instance_id=session_id,
                    question_id=str(question_id),
                    question_text=question_result.get("question"),
                    user_answer=question_result.get("user_answer"),
                    reference_answer=question_result.get("reference_answer"),
                    selected_evaluators=evaluators,
                    metric_scores=self._extract_metric_scores(question_result),
                    final_score=question_result.get("final_score"),
                    metadata={
                        "question_result": question_result,
                        "hint_used": hint_used,
                        "hint_text": hint_text,
                        "hint_payload": hint_payload,
                        "answered_question_ids": answered_question_ids,
                        "not_answered_question_ids": self._get_not_answered_question_ids(
                            selected_questions=selected_questions,
                            answered_question_ids=answered_question_ids,
                        ),
                    },
                )

            progressive_report = self._build_progressive_report(
                round_index=round_index,
                round_results=round_results,
                selected_questions=selected_questions,
                all_results_so_far=question_results,
                answered_question_ids=answered_question_ids,
            )

            progressive_reports.append(progressive_report)

            self._print_progressive_report(progressive_report)

            self._safe_log(
                event_type="progressive_report_generated",
                user_id=user_id,
                dataset_id=dataset_id,
                profile_id=profile_id,
                plan_id=quiz_plan.get("quiz_plan_id"),
                evaluator_instance_id=session_id,
                selected_evaluators=self._collect_all_evaluators(assigned_evaluators),
                final_score=progressive_report.get("current_average_score"),
                metadata={
                    "progressive_report": progressive_report,
                },
            )

        not_answered_question_ids = self._get_not_answered_question_ids(
            selected_questions=selected_questions,
            answered_question_ids=answered_question_ids,
        )

        final_score = self._calculate_average_score(question_results)
        hint_summary = self._build_hint_summary(question_results)

        session_result = {
            "session_id": session_id,
            "dataset_id": dataset_id,
            "profile_id": profile_id,
            "quiz_plan_id": quiz_plan.get("quiz_plan_id"),
            "user_id": user_id,
            "answer_source": answer_source,
            "total_questions": len(selected_questions),
            "answered_questions": len(answered_question_ids),
            "not_answered_questions": len(not_answered_question_ids),
            "answered_question_ids": answered_question_ids,
            "not_answered_question_ids": not_answered_question_ids,
            "final_score": final_score,
            "question_results": question_results,
            "progressive_reports": progressive_reports,
            "evaluator_usage": self._count_evaluator_usage(question_results),
            "hint_summary": hint_summary,
            "created_at": datetime.utcnow().isoformat(),
        }

        self._safe_log(
            event_type="quiz_session_finished",
            user_id=user_id,
            dataset_id=dataset_id,
            profile_id=profile_id,
            plan_id=quiz_plan.get("quiz_plan_id"),
            evaluator_instance_id=session_id,
            selected_evaluators=self._collect_all_evaluators(assigned_evaluators),
            final_score=final_score,
            metadata={
                "answered_question_ids": answered_question_ids,
                "not_answered_question_ids": not_answered_question_ids,
                "total_questions": len(selected_questions),
                "answer_source": answer_source,
                "hint_summary": hint_summary,
                "session_result_summary": {
                    "final_score": final_score,
                    "answered_questions": len(answered_question_ids),
                    "not_answered_questions": len(not_answered_question_ids),
                },
            },
        )

        return session_result

    # ============================================================
    # Answer handling with hint
    # ============================================================

    def _get_answer_payload(
        self,
        question: dict,
        answer_source: str,
    ) -> dict:
        """
        Return answer payload:
            {
                "user_answer": "...",
                "hint_used": bool,
                "hint_text": "...",
                "hint_payload": {...}
            }
        """

        if answer_source == "manual":
            return self._get_manual_answer_with_optional_hint(question)

        if answer_source == "mock_reference":
            return {
                "user_answer": self._get_reference_answer(question),
                "hint_used": False,
                "hint_text": None,
                "hint_payload": None,
            }

        if answer_source == "empty":
            return {
                "user_answer": "",
                "hint_used": False,
                "hint_text": None,
                "hint_payload": None,
            }

        if answer_source == "llm_generated":
            return {
                "user_answer": self._get_llm_generated_answer(question),
                "hint_used": False,
                "hint_text": None,
                "hint_payload": None,
            }

        if answer_source == "dataset_generated":
            return {
                "user_answer": str(
                    question.get("generated_answer")
                    or question.get("model_answer")
                    or ""
                ),
                "hint_used": False,
                "hint_text": None,
                "hint_payload": None,
            }

        return {
            "user_answer": "",
            "hint_used": False,
            "hint_text": None,
            "hint_payload": None,
        }

    def _get_manual_answer_with_optional_hint(
        self,
        question: dict,
    ) -> dict:
        """
        Manual answer mode.

        The user can request one hint before answering.
        """

        self._print_question_for_user(question)

        hint_used = False
        hint_text = None
        hint_payload = None

        request_hint = input("\nDo you want a hint? (y/n) [n]: ").strip().lower()

        if request_hint in ["y", "yes"]:
            hint_payload = self.hint_generator.generate_hint(question)
            hint_text = hint_payload.get("hint_text")
            hint_used = True

            print("\nHINT")
            print("-" * 60)
            print(hint_text)

            print(
                "\nYou can now answer the question. "
                "The hint cannot be requested again for this question."
            )

        user_answer = input("\nYour answer: ").strip()

        return {
            "user_answer": user_answer,
            "hint_used": hint_used,
            "hint_text": hint_text,
            "hint_payload": hint_payload,
        }

    def _get_llm_generated_answer(
        self,
        question: dict,
    ) -> str:
        if hasattr(self.answer_provider, "get_answer"):
            return self.answer_provider.get_answer(
                question=question,
                answer_source="llm_generated",
            )

        if hasattr(self.answer_provider, "generate_answer"):
            return self.answer_provider.generate_answer(question)

        if hasattr(self.answer_provider, "provide_answer"):
            return self.answer_provider.provide_answer(
                question=question,
                answer_source="llm_generated",
            )

        raise RuntimeError(
            "AnswerProvider does not support LLM-generated answers. "
            "Expected method: get_answer, generate_answer, or provide_answer."
        )

    # ============================================================
    # Question display
    # ============================================================

    def _print_question_for_user(
        self,
        question: dict,
    ) -> None:
        """
        Print question, context, and options before collecting the answer.

        Important:
        For datasets with context_dependency = provided_context_required,
        such as HotpotQA, the context should be shown before the user answers.
        """

        print("\nQuestion:")
        print(self._get_question_text(question))

        context_text = self._format_context_for_display(question)

        if context_text:
            print("\nContext:")
            print("-" * 60)
            print(context_text)
            print("-" * 60)

        options_text = self._format_options_for_display(question)

        if options_text:
            print("\nOptions:")
            print(options_text)

    def _get_question_text(
        self,
        question: dict,
    ) -> str:
        """
        Extract question text from common field names.
        """

        return str(
            question.get("question")
            or question.get("question_text")
            or question.get("query")
            or question.get("input")
            or ""
        )

    def _format_options_for_display(
        self,
        question: dict,
    ) -> str:
        """
        Format multiple-choice options for display.

        Supports:
        1. {"A": "...", "B": "..."}
        2. {"label": ["A", "B"], "text": ["...", "..."]}
        3. [{"label": "A", "text": "..."}, ...]
        4. ["A. ...", "B. ..."]
        """

        options = question.get("options") or question.get("choices")

        if not options:
            return ""

        lines = []

        if isinstance(options, dict):
            # Format used by some HF datasets:
            # {"label": ["A", "B"], "text": ["...", "..."]}
            if "label" in options and "text" in options:
                labels = options.get("label", [])
                texts = options.get("text", [])

                for label, text in zip(labels, texts):
                    lines.append(f"- {label}: {text}")

            else:
                # Simple dict format:
                # {"A": "option text", "B": "option text"}
                for label, text in options.items():
                    lines.append(f"- {label}: {text}")

        elif isinstance(options, list):
            for option in options:
                if isinstance(option, dict):
                    label = (
                        option.get("label")
                        or option.get("key")
                        or option.get("id")
                        or ""
                    )
                    text = (
                        option.get("text")
                        or option.get("content")
                        or option.get("value")
                        or ""
                    )

                    if label and text:
                        lines.append(f"- {label}: {text}")
                    elif text:
                        lines.append(f"- {text}")
                    else:
                        lines.append(f"- {option}")

                else:
                    lines.append(f"- {option}")

        return "\n".join(lines)

    def _format_context_for_display(
        self,
        question: dict,
    ) -> str:
        """
        Extract and format context passages for user display.

        Supports common context formats:

        1. Plain string:
           "context": "..."

        2. HotpotQA-style:
           "context": [
               ["Title 1", ["sentence 1", "sentence 2"]],
               ["Title 2", ["sentence 1", "sentence 2"]]
           ]

        3. List of dicts:
           "context": [
               {"title": "...", "text": "..."},
               {"title": "...", "sentences": [...]}
           ]

        4. Dict:
           "context": {"title": "...", "text": "..."}
        """

        context = self._extract_context(question)

        if not context:
            return ""

        lines = []

        if isinstance(context, str):
            return context.strip()

        if isinstance(context, list):
            for index, item in enumerate(context, start=1):
                formatted_item = self._format_context_item(item, index)

                if formatted_item:
                    lines.append(formatted_item)

            return "\n\n".join(lines).strip()

        if isinstance(context, dict):
            return self._format_context_dict(context).strip()

        return str(context).strip()

    def _extract_context(
        self,
        question: dict,
    ) -> Any:
        """
        Extract context from common field names.

        Different datasets use different names.
        This function tries several common keys.
        """

        possible_keys = [
            "context",
            "contexts",
            "passages",
            "paragraphs",
            "provided_context",
            "source_context",
            "supporting_context",
            "evidence",
        ]

        for key in possible_keys:
            value = question.get(key)

            if value:
                return value

        return None

    def _format_context_item(
        self,
        item: Any,
        index: int,
    ) -> str:
        """
        Format one context item.
        """

        # Plain context sentence/passage.
        if isinstance(item, str):
            return f"[{index}] {item}"

        # Dict format.
        if isinstance(item, dict):
            title = (
                item.get("title")
                or item.get("name")
                or item.get("source")
                or f"Passage {index}"
            )

            text = (
                item.get("text")
                or item.get("content")
                or item.get("passage")
                or item.get("paragraph")
                or ""
            )

            sentences = item.get("sentences")

            if not text and isinstance(sentences, list):
                text = " ".join(str(sentence) for sentence in sentences)

            if text:
                return f"[{index}] {title}\n{text}"

            return f"[{index}] {title}\n{item}"

        # HotpotQA-style item:
        # ["Title", ["sentence1", "sentence2"]]
        if isinstance(item, list) and len(item) == 2:
            title, sentences = item

            if isinstance(sentences, list):
                text = " ".join(str(sentence) for sentence in sentences)
            else:
                text = str(sentences)

            return f"[{index}] {title}\n{text}"

        # Fallback for unknown list/object format.
        return f"[{index}] {str(item)}"

    def _format_context_dict(
        self,
        context: dict,
    ) -> str:
        """
        Format context if the whole context object is a dictionary.
        """

        title = (
            context.get("title")
            or context.get("name")
            or context.get("source")
            or "Context"
        )

        text = (
            context.get("text")
            or context.get("content")
            or context.get("passage")
            or context.get("paragraph")
            or ""
        )

        sentences = context.get("sentences")

        if not text and isinstance(sentences, list):
            text = " ".join(str(sentence) for sentence in sentences)

        if text:
            return f"{title}\n{text}"

        lines = []
        for key, value in context.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines)

    # ============================================================
    # Immediate feedback
    # ============================================================

    def _print_immediate_feedback(
        self,
        question_result: dict,
    ) -> None:
        print("\nIMMEDIATE FEEDBACK")
        print("-" * 60)
        print(f"Question ID:     {question_result.get('question_id')}")
        print(f"Score:           {question_result.get('final_score')}")
        print(f"Correct:         {question_result.get('is_correct')}")
        print(f"Your answer:     {question_result.get('user_answer')}")
        print(f"Correct answer:  {question_result.get('correct_answer')}")
        print(f"Hint used:       {question_result.get('hint_used')}")

        if question_result.get("hint_used"):
            print(f"Hint:            {question_result.get('hint_text')}")

        answer_explanation = question_result.get("answer_explanation", {}) or {}

        short_feedback = answer_explanation.get("short_feedback")
        question_understanding = answer_explanation.get("question_understanding")
        key_concept = answer_explanation.get("key_concept")
        teaching_explanation = answer_explanation.get("teaching_explanation")
        detailed_explanation = answer_explanation.get("detailed_explanation")
        evidence_from_context = answer_explanation.get("evidence_from_context", [])
        reasoning_steps = answer_explanation.get("reasoning_steps", [])
        option_analysis = answer_explanation.get("option_analysis", [])
        hint_used_comment = answer_explanation.get("hint_used_comment")
        hint_if_retried = answer_explanation.get("hint_if_retried")
        error_type = answer_explanation.get("error_type")

        feedback = question_result.get("feedback")

        if feedback:
            print(f"\nEvaluator feedback:\n{feedback}")

        if short_feedback:
            print(f"\nShort feedback:\n{short_feedback}")

        if question_understanding:
            print(f"\nWhat the question is asking:\n{question_understanding}")

        if key_concept:
            print(f"\nKey concept:\n{key_concept}")

        if teaching_explanation:
            print(f"\nTeaching explanation:\n{teaching_explanation}")

        if detailed_explanation:
            print(f"\nDetailed answer explanation:\n{detailed_explanation}")

        if evidence_from_context:
            print("\nEvidence / answer support:")
            for index, item in enumerate(evidence_from_context, start=1):
                snippet = item.get("snippet", "")
                support = item.get("how_it_supports_answer", "")
                print(f"{index}. Evidence: {snippet}")
                if support:
                    print(f"   How it supports the answer: {support}")

        if reasoning_steps:
            print("\nReasoning steps:")
            for index, step in enumerate(reasoning_steps, start=1):
                print(f"{index}. {step}")

        if not question_result.get("is_correct"):
            wrong_answer_explanation = question_result.get("wrong_answer_explanation")
            if wrong_answer_explanation:
                print(f"\nWhy your answer is wrong:\n{wrong_answer_explanation}")

        correct_answer_explanation = question_result.get("correct_answer_explanation")

        if correct_answer_explanation:
            print(f"\nWhy the correct answer is correct:\n{correct_answer_explanation}")

        if option_analysis:
            print("\nOption analysis:")
            for item in option_analysis:
                label = item.get("option_label", "")
                text = item.get("option_text", "")
                is_correct = item.get("is_correct", False)
                explanation = item.get("explanation", "")
                print(f"- {label}. {text}")
                print(f"  Correct option: {is_correct}")
                print(f"  Explanation: {explanation}")

        if hint_used_comment:
            print(f"\nHint usage comment:\n{hint_used_comment}")

        if hint_if_retried:
            print(f"\nHint if retried:\n{hint_if_retried}")

        learning_feedback = question_result.get("learning_feedback")

        if learning_feedback:
            print(f"\nLearning feedback:\n{learning_feedback}")

        if error_type:
            print(f"\nError type:\n{error_type}")

    # ============================================================
    # Progressive report
    # ============================================================

    def _build_progressive_report(
        self,
        round_index: int,
        round_results: list[dict],
        selected_questions: list[dict],
        all_results_so_far: list[dict],
        answered_question_ids: list[str],
    ) -> dict:
        round_score = self._calculate_average_score(round_results)
        current_average_score = self._calculate_average_score(all_results_so_far)

        current_answered = len(answered_question_ids)
        total_questions = len(selected_questions)
        remaining = max(0, total_questions - current_answered)

        wrong_questions = [
            {
                "question_id": result.get("question_id"),
                "question": result.get("question"),
                "user_answer": result.get("user_answer"),
                "correct_answer": result.get("correct_answer"),
                "score": result.get("final_score"),
                "hint_used": result.get("hint_used", False),
                "hint_text": result.get("hint_text"),
                "wrong_answer_explanation": result.get("wrong_answer_explanation"),
                "correct_answer_explanation": result.get("correct_answer_explanation"),
                "learning_feedback": result.get("learning_feedback"),
                "answer_explanation": result.get("answer_explanation", {}),
            }
            for result in round_results
            if not result.get("is_correct")
        ]

        return {
            "report_type": "progressive",
            "round_index": round_index,
            "round_score": round_score,
            "current_average_score": current_average_score,
            "answered_questions": current_answered,
            "total_questions": total_questions,
            "remaining_questions": remaining,
            "round_results": round_results,
            "wrong_questions": wrong_questions,
            "hint_summary_so_far": self._build_hint_summary(all_results_so_far),
            "created_at": datetime.utcnow().isoformat(),
        }

    def _print_progressive_report(
        self,
        progressive_report: dict,
    ) -> None:
        print("\nPROGRESSIVE REPORT")
        print("-" * 60)
        print(f"Round:                 {progressive_report.get('round_index')}")
        print(f"Round score:           {progressive_report.get('round_score')}")
        print(f"Current average score: {progressive_report.get('current_average_score')}")
        print(f"Answered questions:    {progressive_report.get('answered_questions')}")
        print(f"Total questions:       {progressive_report.get('total_questions')}")
        print(f"Remaining questions:   {progressive_report.get('remaining_questions')}")

        hint_summary = progressive_report.get("hint_summary_so_far", {})

        if hint_summary:
            print("\nHint summary so far:")
            print(f"- Hints used:             {hint_summary.get('hint_used_count')}")
            print(f"- Correct with hint:      {hint_summary.get('correct_with_hint_count')}")
            print(f"- Wrong with hint:        {hint_summary.get('wrong_with_hint_count')}")
            print(f"- Correct without hint:   {hint_summary.get('correct_without_hint_count')}")
            print(f"- Wrong without hint:     {hint_summary.get('wrong_without_hint_count')}")

        wrong_questions = progressive_report.get("wrong_questions", [])

        if wrong_questions:
            print("\nWrong questions in this round:")
            for item in wrong_questions:
                print(f"- {item.get('question_id')}: score={item.get('score')}")
                print(f"  Correct answer: {item.get('correct_answer')}")
                print(f"  Hint used: {item.get('hint_used')}")
                if item.get("hint_used"):
                    print(f"  Hint: {item.get('hint_text')}")
                print(f"  Explanation: {item.get('wrong_answer_explanation')}")

    # ============================================================
    # Utility methods
    # ============================================================

    def _build_hint_summary(
        self,
        question_results: list[dict],
    ) -> dict:
        total = len(question_results)

        hint_used = [
            result for result in question_results
            if result.get("hint_used")
        ]

        no_hint = [
            result for result in question_results
            if not result.get("hint_used")
        ]

        correct_with_hint = [
            result for result in hint_used
            if result.get("is_correct")
        ]

        wrong_with_hint = [
            result for result in hint_used
            if not result.get("is_correct")
        ]

        correct_without_hint = [
            result for result in no_hint
            if result.get("is_correct")
        ]

        wrong_without_hint = [
            result for result in no_hint
            if not result.get("is_correct")
        ]

        return {
            "total_answered_questions": total,
            "hint_used_count": len(hint_used),
            "hint_not_used_count": len(no_hint),
            "hint_usage_rate": round(len(hint_used) / total, 4) if total else 0.0,
            "correct_with_hint_count": len(correct_with_hint),
            "wrong_with_hint_count": len(wrong_with_hint),
            "correct_without_hint_count": len(correct_without_hint),
            "wrong_without_hint_count": len(wrong_without_hint),
        }

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

    def _get_reference_answer(
        self,
        question: dict,
    ) -> str:
        return str(
            question.get("reference_answer")
            or question.get("answer")
            or question.get("label")
            or question.get("target")
            or ""
        )

    def _get_not_answered_question_ids(
        self,
        selected_questions: list[dict],
        answered_question_ids: list[str],
    ) -> list[str]:
        answered_set = set(str(qid) for qid in answered_question_ids)

        return [
            self._get_question_id(question)
            for question in selected_questions
            if self._get_question_id(question) not in answered_set
        ]

    def _calculate_average_score(
        self,
        question_results: list[dict],
    ) -> float:
        if not question_results:
            return 0.0

        scores = [
            float(result.get("final_score", 0.0))
            for result in question_results
        ]

        return round(sum(scores) / len(scores), 4)

    def _extract_metric_scores(
        self,
        question_result: dict,
    ) -> dict:
        metric_scores = {}

        for output in question_result.get("evaluator_outputs", []):
            evaluator_name = output.get("evaluator_name")
            metric_scores[evaluator_name] = output.get("score")

        return metric_scores

    def _collect_all_evaluators(
        self,
        assigned_evaluators: dict,
    ) -> list[str]:
        evaluators = []

        for value in assigned_evaluators.values():
            for evaluator in value:
                if evaluator not in evaluators:
                    evaluators.append(evaluator)

        return evaluators

    def _count_evaluator_usage(
        self,
        question_results: list[dict],
    ) -> dict:
        usage = {}

        for result in question_results:
            for evaluator_name in result.get("assigned_evaluators", []):
                usage[evaluator_name] = usage.get(evaluator_name, 0) + 1

        return usage

    def _safe_log(
        self,
        event_type: str,
        user_id: str,
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
        status: str = "success",
        metadata: dict | None = None,
    ) -> None:
        if self.logger is None:
            return

        payload = {
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
            "selected_evaluators": selected_evaluators or [],
            "metric_scores": metric_scores or {},
            "final_score": final_score,
            "status": status,
            "metadata": metadata or {},
        }

        try:
            if hasattr(self.logger, "log_event"):
                self.logger.log_event(**payload)
                return

            if hasattr(self.logger, "log_activity"):
                self.logger.log_activity(payload)
                return

            if hasattr(self.logger, "write"):
                self.logger.write(payload)
                return

        except TypeError:
            try:
                self.logger.log_event(payload)
            except Exception:
                pass
        except Exception:
            pass