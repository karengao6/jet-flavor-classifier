from __future__ import annotations

import numpy as np


def select_indices(
    n_samples: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Select dataset indices reproducibly.

    Args:
        n_samples:
            Total number of available samples.

        max_samples:
            Maximum number of samples to select.
            If None, select all samples.

        seed:
            Random seed used for reproducible sampling.

    Returns:
        1D NumPy array containing selected indices.
    """

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")

    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            "max_samples must be positive."
        )

    # Use every sample if no limit was specified.
    if max_samples is None or max_samples >= n_samples:
        return np.arange(n_samples)

    # Local RNG keeps this operation reproducible
    # without modifying NumPy's global random state.
    rng = np.random.default_rng(seed)

    return rng.choice(
        n_samples,
        size=max_samples,
        replace=False,
    )