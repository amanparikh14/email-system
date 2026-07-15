from __future__ import annotations

import time

import anthropic
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
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)

# Claude Opus 4.8 (and later Claude models) reject non-default `temperature` —
# CONFIG.judge_temperature is recorded in run metadata as the intended
# determinism setting, but is deliberately not sent on the wire. Determinism
# instead comes from a fixed model + fixed prompt + thinking left off.
_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or CONFIG.judge_model
        self._client = anthropic.Anthropic()
        self._async_client = anthropic.AsyncAnthropic()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(CONFIG.max_retry_attempts),
        wait=wait_random_exponential(multiplier=1, max=20),
        reraise=True,
    )
    def complete(self, system: str, user: str, **cfg) -> str:
        start = time.monotonic()
        cfg.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            **cfg,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(f"provider=anthropic model={self.model} latency_ms={latency_ms}")
        return next(block.text for block in response.content if block.type == "text")

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(CONFIG.max_retry_attempts),
        wait=wait_random_exponential(multiplier=1, max=20),
        reraise=True,
    )
    async def acomplete(self, system: str, user: str, **cfg) -> str:
        start = time.monotonic()
        cfg.setdefault("max_tokens", _DEFAULT_MAX_TOKENS)
        response = await self._async_client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            **cfg,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(f"provider=anthropic model={self.model} latency_ms={latency_ms}")
        return next(block.text for block in response.content if block.type == "text")
