from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CONFIG
from src.data.schema import PerResponseRecord
from src.evaluator.prompts import JUDGE_SYSTEM, build_judge_prompt
from src.llm.anthropic import AnthropicProvider
from src.logging import get_logger

logger = get_logger(__name__)

_DIMENSIONS = ("relevance", "completeness", "actionability", "correctness", "tone")

_JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "integer"},
        "completeness": {"type": "integer"},
        "actionability": {"type": "integer"},
        "correctness": {"type": "integer"},
        "tone": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": list(_DIMENSIONS) + ["reason"],
    "additionalProperties": False,
}

_similarity_model: SentenceTransformer | None = None


def _get_similarity_model() -> SentenceTransformer:
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer(CONFIG.embed_model)
    return _similarity_model


def _reply_similarity(generated_reply: str, reference_reply: str) -> float:
    model = _get_similarity_model()
    embeddings = model.encode([generated_reply, reference_reply], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def _weighted_score(dimension_scores: dict) -> float:
    # each dimension is 0-5; normalize to 0-1 then apply config weights -> 0-100
    total = sum((dimension_scores[dim] / 5.0) * CONFIG.weights[dim] for dim in _DIMENSIONS)
    return round(total * 100, 2)


def _parse_judge_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    data = json.loads(text)
    for dim in _DIMENSIONS:
        value = int(data[dim])
        data[dim] = min(5, max(0, value))  # defensive clamp; judge is instructed to stay in range
    return data


def judge(
    email: str,
    generated_reply: str,
    reference_reply: str,
    provider: AnthropicProvider,
    *,
    id_: str = "",
    category: str = "",
    gen_provider_used: str = "",
    gen_fallback_used: bool = False,
) -> PerResponseRecord:
    """Scores a single generated reply. `provider` must be the pinned judge
    provider -- this function never falls over to another model (NFR-2/NFR-3):
    if the call fails after the provider's own internal retries, it raises.

    `gen_provider_used`/`gen_fallback_used` come from the GeneratorOutput
    produced for this reply -- they describe generation, not judging, and are
    carried through onto the record per the FR-8 data contract.
    """
    prompt = build_judge_prompt(email, generated_reply, reference_reply)

    last_error: Exception | None = None
    parsed: dict | None = None
    for attempt in range(2):  # JSON-parse failures retry as their own class (NFR-5)
        raw = provider.complete(
            JUDGE_SYSTEM,
            prompt,
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_OUTPUT_SCHEMA}},
        )
        try:
            parsed = _parse_judge_json(raw)
            break
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(f"judge JSON parse failed on attempt {attempt + 1}: {exc}")
            last_error = exc
    if parsed is None:
        raise RuntimeError(f"judge returned unparseable output after retries: {last_error}")

    dimension_scores = {dim: parsed[dim] for dim in _DIMENSIONS}
    score_overall = _weighted_score(dimension_scores)
    similarity = _reply_similarity(generated_reply, reference_reply)

    return PerResponseRecord(
        id=id_,
        category=category,
        score_overall=score_overall,
        dimension_scores=dimension_scores,
        similarity=similarity,
        judge_reason=parsed.get("reason", ""),
        judge_model=CONFIG.judge_model,
        provider_used=gen_provider_used,
        fallback_used=gen_fallback_used,
    )
