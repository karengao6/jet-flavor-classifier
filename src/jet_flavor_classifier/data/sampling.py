from __future__ import annotations

import numpy as np


def select_indices(
    n_samples: int,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    """
    Select which samples will participate in an experiment.

    If max_samples is None:
        use the entire dataset.

    Otherwise:
        randomly select max_samples samples.

    The seed makes the selection reproducible.
    """

    indices = np.arange(n_samples)

    if max_samples is None:
        return indices

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be positive."
        )

    if max_samples > n_samples:
        raise ValueError(
            f"max_samples={max_samples} exceeds "
            f"dataset size={n_samples}"
        )

    # Use a local RNG rather than modifying global NumPy state.
    rng = np.random.default_rng(seed)

    return rng.choice(
        indices,
        size=max_samples,
        replace=False,
    )