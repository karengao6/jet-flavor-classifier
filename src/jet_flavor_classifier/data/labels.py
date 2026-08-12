"""
Convert CERN truth labels into contiguous class IDs.

The HDF5 dataset uses physics labels:

    0  -> light jet
    4  -> charm jet
    5  -> bottom jet
    15 -> tau jet

PyTorch classification models expect contiguous class IDs:

    0 -> light
    1 -> charm
    2 -> bottom
    3 -> tau

The truth label is ONLY the supervised-learning target.
It is never provided as a model input.
"""

from __future__ import annotations

import numpy as np


RAW_TO_CLASS = {
    0: 0,    # light
    4: 1,    # charm
    5: 2,    # bottom
    15: 3,   # tau
}


CLASS_NAMES = (
    "light",
    "charm",
    "bottom",
    "tau",
)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert raw CERN labels into contiguous class IDs.

    Args:
        labels: Raw HadronGhostTruthLabelID values.

    Returns:
        Integer class IDs in [0, 3].
    """
    labels = np.asarray(labels)

    unique_labels = set(np.unique(labels).tolist())
    unknown_labels = unique_labels - set(RAW_TO_CLASS)

    if unknown_labels:
        raise ValueError(
            f"Unexpected truth labels: {sorted(unknown_labels)}. "
            f"Expected: {sorted(RAW_TO_CLASS)}"
        )

    return np.array(
        [RAW_TO_CLASS[int(label)] for label in labels],
        dtype=np.int64,
    )


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    """
    Return the number of examples in each class.
    """
    encoded = encode_labels(labels)

    return {
        CLASS_NAMES[class_id]: int(
            np.sum(encoded == class_id)
        )
        for class_id in range(len(CLASS_NAMES))
    }