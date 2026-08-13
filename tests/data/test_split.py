import numpy as np

from jet_flavor_classifier.data.split import create_split_indices


def test_split_is_disjoint():
    indices = np.arange(100)

    train, val, test = create_split_indices(
        indices,
        seed=42,
    )

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_split_is_reproducible():
    indices = np.arange(100)

    split1 = create_split_indices(indices, seed=42)
    split2 = create_split_indices(indices, seed=42)

    for a, b in zip(split1, split2):
        np.testing.assert_array_equal(a, b)