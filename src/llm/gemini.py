import os

import openai

from src.llm.openai import OpenAIProvider

# STUB, not exercised live in this submission. Gemini exposes an
# OpenAI-compatible endpoint, so the same chat-completions adapter works by
# swapping base_url + api_key -- no separate SDK needed.
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiProvider(OpenAIProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY", "")
        self._client = openai.OpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL)
