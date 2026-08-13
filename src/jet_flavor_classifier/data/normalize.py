# src/jet_flavor_classifier/data/normalize.py

from __future__ import annotations

import numpy as np


class StandardNormalizer:
    """
    Standardize each feature:

        x_normalized = (x - mean) / std

    Statistics must be fitted on TRAINING data only.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """
        Calculate mean/std for each feature.

        data:
            [number_of_tracks, number_of_features]
        """

        self.mean = np.mean(
            data,
            axis=0,
        )

        self.std = np.std(
            data,
            axis=0,
        )

        # Avoid division by zero for constant features.
        self.std = np.where(
            self.std == 0,
            1.0,
            self.std,
        )

    def transform(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Apply previously learned normalization."""

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Normalizer must be fitted before transform()."
            )

        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the normalizer and transform the same data.

        This is convenient for training data.
        """

        self.fit(data)
        return self.transform(data)