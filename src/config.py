import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Retrieval
    k: int = int(os.getenv("RETRIEVAL_K", "3"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    embed_model: str = "all-MiniLM-L6-v2"

    # Generation
    gen_model: str = os.getenv("GEN_MODEL", "gpt-5.6-terra")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gen_chain: tuple = ("openai", "gemini")

    # Judge -- pinned, single-sourced, never falls over (NFR-2, NFR-3)
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-opus-4-8")
    judge_temperature: float = 0.0

    # Scoring weights -- must sum to 1.0 (validated below)
    weights: dict = field(
        default_factory=lambda: {
            "relevance": 0.25,
            "completeness": 0.25,
            "actionability": 0.20,
            "correctness": 0.175,
            "tone": 0.125,
        }
    )

    # Reproducibility
    seed: int = 42

    # Retry / backoff (NFR-5)
    max_retry_attempts: int = 4

    # Paths
    data_cache_dir: str = "data_cache"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"config.weights must sum to 1.0, got {total}")


CONFIG = Config()
