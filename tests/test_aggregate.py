import pytest

from src.data.schema import PerResponseRecord
from src.evaluator.aggregate import aggregate
from src.evaluator.validate import _spearman


def _record(id_, category, score):
    return PerResponseRecord(
        id=id_,
        category=category,
        score_overall=score,
        dimension_scores={"relevance": 5, "completeness": 5, "actionability": 5, "correctness": 5, "tone": 5},
        similarity=0.5,
        judge_reason="",
        judge_model="claude-opus-4-8",
        provider_used="openai",
        fallback_used=False,
    )


def test_aggregate_overall_mean():
    records = [_record("1", "billing", 80), _record("2", "billing", 100)]
    result = aggregate(records)
    assert result.overall_score == pytest.approx(90.0)
    assert result.n == 2


def test_aggregate_per_category():
    records = [_record("1", "billing", 80), _record("2", "shipping", 60)]
    result = aggregate(records)
    assert result.per_category_scores["billing"] == pytest.approx(80.0)
    assert result.per_category_scores["shipping"] == pytest.approx(60.0)


def test_spearman_perfect_correlation():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_inverse_correlation():
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
