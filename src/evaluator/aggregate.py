from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.config import CONFIG
from src.data.schema import AggregateRecord, PerResponseRecord


def aggregate(records: list[PerResponseRecord]) -> AggregateRecord:
    if not records:
        raise ValueError("cannot aggregate an empty list of records")

    overall_score = round(sum(r.score_overall for r in records) / len(records), 2)

    by_category: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_category[r.category].append(r.score_overall)
    per_category_scores = {
        cat: round(sum(scores) / len(scores), 2) for cat, scores in by_category.items()
    }

    return AggregateRecord(
        overall_score=overall_score,
        per_category_scores=per_category_scores,
        n=len(records),
        run_metadata={
            "judge_model": CONFIG.judge_model,
            "generation_model": CONFIG.gen_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def write_results(
    records: list[PerResponseRecord], aggregate_record: AggregateRecord, results_dir: str | None = None
) -> Path:
    out_dir = Path(results_dir or CONFIG.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "per_response.jsonl"
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

    aggregate_path = out_dir / "aggregate.json"
    with open(aggregate_path, "w") as f:
        json.dump(asdict(aggregate_record), f, indent=2)

    return out_dir


def render_summary_table(records: list[PerResponseRecord], aggregate_record: AggregateRecord) -> str:
    lines = []
    header = f"{'id':<14}{'category':<20}{'score':>7}{'sim':>7}{'fallback':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in records:
        lines.append(
            f"{r.id:<14}{r.category[:19]:<20}{r.score_overall:>7.1f}{r.similarity:>7.2f}"
            f"{str(r.fallback_used):>10}"
        )
    lines.append("-" * len(header))
    lines.append(f"Overall score: {aggregate_record.overall_score:.2f} / 100  (n={aggregate_record.n})")
    lines.append("Per-category:")
    for cat, score in sorted(aggregate_record.per_category_scores.items()):
        lines.append(f"  {cat:<30} {score:.2f}")
    return "\n".join(lines)
