# src/llm/client.py

"""
This file wraps OpenAI API calls.

Why this file exists:
The rest of the framework should not directly call OpenAI everywhere.
Instead, all OpenAI calls go through this client.

This makes it easier to later replace OpenAI with:
- GWDG API
- local model
- another LLM provider
"""

import os
from openai import OpenAI
from dotenv import load_dotenv


class LLMClient:
    """
    Simple OpenAI client for chat and embeddings.
    """

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

        self.client = OpenAI(api_key=api_key)

        # Model names are read from .env.
        # If not found, default models are used.
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the OpenAI chat model and return text output.
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert text into embedding vectors.
        Used later for retrieval / similarity.
        """

        response = self.client.embeddings.create(
            model=self.embed_model,
            input=texts
        )

        return [item.embedding for item in response.data]