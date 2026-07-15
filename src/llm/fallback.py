from __future__ import annotations

from src.llm.base import Provider
from src.logging import get_logger

logger = get_logger(__name__)


class FallbackProvider:
    """Tries providers in order; each provider already retries transient
    failures internally (see openai.py/anthropic.py), so this class only
    advances to the next provider once a provider's own retries are exhausted.
    """

    name = "fallback"

    def __init__(self, chain: list[Provider]) -> None:
        if not chain:
            raise ValueError("FallbackProvider requires at least one provider")
        self.chain = chain

    def complete(self, system: str, user: str, **cfg) -> tuple[str, str]:
        last_error: Exception | None = None
        for provider in self.chain:
            try:
                return provider.complete(system, user, **cfg), provider.name
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
                # provider failure here means its own retries are exhausted
                logger.warning(f"provider={provider.name} exhausted retries, falling over: {exc}")
                last_error = exc
        raise RuntimeError(f"all providers in fallback chain failed: {last_error}") from last_error

    async def acomplete(self, system: str, user: str, **cfg) -> tuple[str, str]:
        last_error: Exception | None = None
        for provider in self.chain:
            try:
                return await provider.acomplete(system, user, **cfg), provider.name
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"provider={provider.name} exhausted retries, falling over: {exc}")
                last_error = exc
        raise RuntimeError(f"all providers in fallback chain failed: {last_error}") from last_error
