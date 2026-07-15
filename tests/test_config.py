from src.config import CONFIG


def test_weights_sum_to_one():
    assert abs(sum(CONFIG.weights.values()) - 1.0) < 1e-6


def test_weights_cover_all_dimensions():
    expected = {"relevance", "completeness", "actionability", "correctness", "tone"}
    assert set(CONFIG.weights.keys()) == expected
