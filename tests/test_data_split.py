from src.data.build import split_indices


def test_split_is_disjoint():
    store_idx, test_idx = split_indices(n_rows=200, store_size=60, test_size=18, seed=42)
    assert set(store_idx).isdisjoint(set(test_idx))
    assert len(store_idx) == 60
    assert len(test_idx) == 18


def test_split_is_deterministic_given_seed():
    a = split_indices(n_rows=200, store_size=60, test_size=18, seed=42)
    b = split_indices(n_rows=200, store_size=60, test_size=18, seed=42)
    assert a == b


def test_split_differs_across_seeds():
    a = split_indices(n_rows=200, store_size=60, test_size=18, seed=1)
    b = split_indices(n_rows=200, store_size=60, test_size=18, seed=2)
    assert a != b
