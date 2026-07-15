from dataclasses import dataclass, field


@dataclass(frozen=True)
class Row:
    id: str
    category: str
    email: str
    sent_reply: str
    source: str  # "corpus" | "synthesized" -- provenance, see README


@dataclass(frozen=True)
class GeneratorOutput:
    reply_text: str
    retrieved_example_ids: list[str]
    fallback_used: bool
    provider_used: str
    latency_ms: int


@dataclass(frozen=True)
class PerResponseRecord:
    id: str
    category: str
    score_overall: float
    dimension_scores: dict
    similarity: float
    judge_reason: str
    judge_model: str
    provider_used: str
    fallback_used: bool


@dataclass(frozen=True)
class AggregateRecord:
    overall_score: float
    per_category_scores: dict
    n: int
    run_metadata: dict = field(default_factory=dict)
