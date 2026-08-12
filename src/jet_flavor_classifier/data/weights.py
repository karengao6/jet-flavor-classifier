"""
Class-weight calculation for imbalanced jet-flavor classification.

Class weights are calculated ONLY from the training split.

This prevents validation/test data from influencing the training
procedure.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute balanced class weights.

    Args:
        labels:
            Encoded class labels, e.g. 0, 1, 2, 3.

        num_classes:
            Number of classes.

    Returns:
        Float32 array where weights[i] is the weight for class i.
    """

    classes = np.arange(num_classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return weights.astype(np.float32)