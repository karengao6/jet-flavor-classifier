import numpy as np

from jet_flavor_classifier.data.sampling import select_indices


def test_sampling_is_reproducible():
    a = select_indices(
        n_samples=100,
        max_samples=10,
        seed=42,
    )

    b = select_indices(
        n_samples=100,
        max_samples=10,
        seed=42,
    )

    np.testing.assert_array_equal(a, b)


def test_sampling_returns_requested_size():
    indices = select_indices(
        n_samples=100,
        max_samples=10,
        seed=42,
    )

    assert len(indices) == 10
    assert len(np.unique(indices)) == 10