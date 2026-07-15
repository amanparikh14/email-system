from __future__ import annotations

import time

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import CONFIG
from src.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or CONFIG.gen_model
        self._client = openai.OpenAI()
        self._async_client = openai.AsyncOpenAI()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(CONFIG.max_retry_attempts),
        wait=wait_random_exponential(multiplier=1, max=20),
        reraise=True,
    )
    def complete(self, system: str, user: str, **cfg) -> str:
        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **cfg,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(f"provider=openai model={self.model} latency_ms={latency_ms}")
        return response.choices[0].message.content

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(CONFIG.max_retry_attempts),
        wait=wait_random_exponential(multiplier=1, max=20),
        reraise=True,
    )
    async def acomplete(self, system: str, user: str, **cfg) -> str:
        start = time.monotonic()
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **cfg,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(f"provider=openai model={self.model} latency_ms={latency_ms}")
        return response.choices[0].message.content
