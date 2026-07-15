import time

from src.data.schema import GeneratorOutput
from src.generator.prompts import (
    FALLBACK_EXAMPLES,
    GEN_SYSTEM,
    build_fewshot_prompt,
)
from src.llm.fallback import FallbackProvider
from src.retrieval.store import FALLBACK, RetrievalStore


def _prepare(email: str, store: RetrievalStore, k: int, threshold: float) -> tuple[str, list[str], bool]:
    neighbors = store.search(email, k=k, threshold=threshold)

    fallback_used = neighbors is FALLBACK
    if fallback_used:
        examples = FALLBACK_EXAMPLES
        retrieved_ids: list[str] = []
    else:
        examples = [
            {"category": row.category, "email": row.email, "reply": row.sent_reply}
            for row in neighbors
        ]
        retrieved_ids = [row.id for row in neighbors]

    return build_fewshot_prompt(email, examples), retrieved_ids, fallback_used


def generate(
    email: str, store: RetrievalStore, provider: FallbackProvider, k: int, threshold: float
) -> GeneratorOutput:
    start = time.monotonic()
    user_prompt, retrieved_ids, fallback_used = _prepare(email, store, k, threshold)
    reply_text, provider_used = provider.complete(GEN_SYSTEM, user_prompt)
    latency_ms = int((time.monotonic() - start) * 1000)

    return GeneratorOutput(
        reply_text=reply_text.strip(),
        retrieved_example_ids=retrieved_ids,
        fallback_used=fallback_used,
        provider_used=provider_used,
        latency_ms=latency_ms,
    )


async def agenerate(
    email: str, store: RetrievalStore, provider: FallbackProvider, k: int, threshold: float
) -> GeneratorOutput:
    """Async counterpart of generate(), used by the FastAPI /suggest endpoint (FR-13)."""
    start = time.monotonic()
    user_prompt, retrieved_ids, fallback_used = _prepare(email, store, k, threshold)
    reply_text, provider_used = await provider.acomplete(GEN_SYSTEM, user_prompt)
    latency_ms = int((time.monotonic() - start) * 1000)

    return GeneratorOutput(
        reply_text=reply_text.strip(),
        retrieved_example_ids=retrieved_ids,
        fallback_used=fallback_used,
        provider_used=provider_used,
        latency_ms=latency_ms,
    )
