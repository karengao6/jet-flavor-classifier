import numpy as np

from jet_flavor_classifier.data.normalize import StandardNormalizer


def test_normalizer():
    data = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    normalizer = StandardNormalizer()
    result = normalizer.fit_transform(data)

    np.testing.assert_allclose(
        result.mean(axis=0),
        [0.0, 0.0],
        atol=1e-6,
    )

    np.testing.assert_allclose(
        result.std(axis=0),
        [1.0, 1.0],
        atol=1e-6,
    )