import time

from src.data.schema import GeneratorOutput
from src.generator.prompts import (
    FALLBACK_EXAMPLES,
    GEN_SYSTEM,
    build_fewshot_prompt,
)
from src.llm.fallback import FallbackProvider
from src.retrieval.store import FALLBACK, RetrievalStore


def generate(
    email: str, store: RetrievalStore, provider: FallbackProvider, k: int, threshold: float
) -> GeneratorOutput:
    start = time.monotonic()
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

    user_prompt = build_fewshot_prompt(email, examples)
    reply_text, provider_used = provider.complete(GEN_SYSTEM, user_prompt)
    latency_ms = int((time.monotonic() - start) * 1000)

    return GeneratorOutput(
        reply_text=reply_text.strip(),
        retrieved_example_ids=retrieved_ids,
        fallback_used=fallback_used,
        provider_used=provider_used,
        latency_ms=latency_ms,
    )
