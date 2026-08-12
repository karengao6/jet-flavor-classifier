"""
Reproducible train/validation/test splitting.

Jets from the same event remain in the same split to prevent
event-level information leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_by_event(
    event_numbers: np.ndarray,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split jet indices by event number.

    All jets belonging to the same event are assigned to the
    same split.

    Args:
        event_numbers:
            Event number for each jet.

        test_size:
            Fraction of events assigned to the test set.

        validation_size:
            Fraction of events assigned to the validation set.

        random_state:
            Seed used to make the split reproducible.

    Returns:
        train_indices:
            Indices of jets assigned to training.

        validation_indices:
            Indices of jets assigned to validation.

        test_indices:
            Indices of jets assigned to testing.
    """

    if test_size <= 0 or validation_size <= 0:
        raise ValueError(
            "test_size and validation_size must be positive."
        )

    if test_size + validation_size >= 1.0:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    unique_events = np.unique(event_numbers)

    train_events, temp_events = train_test_split(
        unique_events,
        test_size=test_size + validation_size,
        random_state=random_state,
    )

    relative_validation_size = (
        validation_size / (test_size + validation_size)
    )

    validation_events, test_events = train_test_split(
        temp_events,
        test_size=relative_validation_size,
        random_state=random_state,
    )

    train_mask = np.isin(
        event_numbers,
        train_events,
    )

    validation_mask = np.isin(
        event_numbers,
        validation_events,
    )

    test_mask = np.isin(
        event_numbers,
        test_events,
    )

    return (
        np.flatnonzero(train_mask),
        np.flatnonzero(validation_mask),
        np.flatnonzero(test_mask),
    )