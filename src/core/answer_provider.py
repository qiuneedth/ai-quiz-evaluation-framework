# src/core/answer_provider.py

"""
Answer Provider.

This module provides candidate answers for quiz questions.

In the supervisor's diagrams, the Quiz Session Manager receives a user answer.
In this implementation, "user_answer" means the candidate answer to evaluate.

The candidate answer can come from:
1. manual user input
2. dataset-provided generated answer
3. LLM-generated answer
4. mock reference answer, only for testing the pipeline

Important:
mock_reference is NOT a real evaluation mode.
It is only used to test whether the full runtime pipeline works.
"""


from src.core.env_loader import (
    get_openai_api_key,
    get_openai_chat_model,
    load_project_env,
)


class AnswerProvider:
    """
    Provides candidate answers for questions.
    """

    def __init__(self, llm_client=None):
        """
        Initialize AnswerProvider.

        Parameters:
            llm_client:
                Optional project LLM client.
                It should support:
                    llm_client.chat(system_prompt=..., user_prompt=...)
        """

        load_project_env()
        self.llm_client = llm_client

    def get_answer(
        self,
        question: dict,
        answer_source: str,
    ) -> str:
        """
        Return a candidate answer according to answer_source.

        Supported answer_source:
            - manual
            - dataset_generated
            - llm_generated
            - mock_reference
            - empty
        """

        if answer_source == "manual":
            return self._get_manual_answer(question)

        if answer_source == "dataset_generated":
            return self._get_dataset_generated_answer(question)

        if answer_source == "llm_generated":
            return self._get_llm_generated_answer(question)

        if answer_source == "mock_reference":
            return self._get_mock_reference_answer(question)

        if answer_source == "empty":
            return ""

        raise ValueError(f"Unknown answer_source: {answer_source}")

    def _get_manual_answer(self, question: dict) -> str:
        """
        Ask the user to type an answer in the terminal.
        """

        print("\nQuestion:")
        print(question.get("question", ""))

        options = question.get("choices") or question.get("options")

        if options:
            print("\nOptions:")
            for option in options:
                print(f"- {option}")

        return input("\nYour answer: ").strip()

    def _get_dataset_generated_answer(self, question: dict) -> str:
        """
        Use generated answer already stored in the dataset.

        Some datasets, especially RAG evaluation datasets, may already contain
        generated responses.
        """

        answer = (
            question.get("model_answer")
            or question.get("generated_answer")
            or question.get("response")
            or question.get("candidate_answer")
        )

        if answer is None:
            raise ValueError(
                "No dataset-generated answer found in this question. "
                "Use --answer-source manual or --answer-source llm_generated instead."
            )

        return str(answer)

    def _get_llm_generated_answer(self, question: dict) -> str:
        """
        Generate an answer using an LLM.

        This is used when we want to evaluate AI-generated answers.

        The generated answer is NOT the evaluator.
        It is the candidate answer that will later be evaluated.
        """

        system_prompt = (
            "You are a quiz answering assistant. "
            "Answer the user's question concisely. "
            "If context is provided, use only the provided context."
        )

        context = question.get("context")
        question_text = question.get("question", "")

        if context:
            user_prompt = f"""
Context:
{context}

Question:
{question_text}

Answer:
"""
        else:
            user_prompt = f"""
Question:
{question_text}

Answer:
"""

        return self._call_llm_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _get_mock_reference_answer(self, question: dict) -> str:
        """
        Use reference answer as candidate answer.

        This is only for testing whether the full pipeline works.
        It should not be used as real evaluation evidence.
        """

        return str(
            question.get("reference_answer")
            or question.get("answer")
            or ""
        )

    def _call_llm_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Call an LLM and return plain text.

        Priority:
            1. Use project llm_client if provided.
            2. Use OpenAI SDK with OPENAI_API_KEY from .env.

        The .env file can be stored either in:
            - project root .env
            - src/.env
        """

        load_project_env()

        if self.llm_client is not None and hasattr(self.llm_client, "chat"):
            return self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ).strip()

        api_key = get_openai_api_key()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY was not found. "
                "Please check your .env file. It can be placed in project root or src/.env. "
                "Example: OPENAI_API_KEY=sk-..."
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai"
            ) from error

        model = get_openai_chat_model()

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content.strip()