import pytest

from src.evaluator.judge import _parse_judge_json, _weighted_score


def test_parse_judge_json_plain():
    raw = '{"relevance": 5, "completeness": 4, "actionability": 3, "correctness": 5, "tone": 4, "reason": "ok"}'
    parsed = _parse_judge_json(raw)
    assert parsed["relevance"] == 5
    assert parsed["reason"] == "ok"


def test_parse_judge_json_fenced_code_block():
    raw = '```json\n{"relevance": 5, "completeness": 5, "actionability": 5, "correctness": 5, "tone": 5, "reason": "ok"}\n```'
    parsed = _parse_judge_json(raw)
    assert parsed["completeness"] == 5


def test_parse_judge_json_clamps_out_of_range():
    raw = '{"relevance": 7, "completeness": -1, "actionability": 3, "correctness": 5, "tone": 4, "reason": "ok"}'
    parsed = _parse_judge_json(raw)
    assert parsed["relevance"] == 5
    assert parsed["completeness"] == 0


def test_weighted_score_all_max_is_100():
    scores = {"relevance": 5, "completeness": 5, "actionability": 5, "correctness": 5, "tone": 5}
    assert _weighted_score(scores) == pytest.approx(100.0)


def test_weighted_score_all_zero_is_0():
    scores = {"relevance": 0, "completeness": 0, "actionability": 0, "correctness": 0, "tone": 0}
    assert _weighted_score(scores) == pytest.approx(0.0)


def test_weighted_score_respects_weights():
    # only relevance (weight .25) maxed out -> 25 points
    scores = {"relevance": 5, "completeness": 0, "actionability": 0, "correctness": 0, "tone": 0}
    assert _weighted_score(scores) == pytest.approx(25.0)
