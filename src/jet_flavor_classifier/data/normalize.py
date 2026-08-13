from __future__ import annotations

import numpy as np


class StandardNormalizer:
    """
    Standardizes each feature:

        x_normalized = (x - mean) / std

    The statistics MUST be fitted using training data only.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """
        Calculate normalization statistics.

        data shape:
            [num_tracks, num_features]
        """

        self.mean = np.mean(
            data,
            axis=0,
        )

        self.std = np.std(
            data,
            axis=0,
        )

        # Constant features have std=0.
        #
        # Dividing by zero would cause problems, so leave
        # those features unchanged.
        self.std = np.where(
            self.std == 0,
            1.0,
            self.std,
        )

    def transform(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Apply previously fitted statistics.
        """

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Normalizer must be fitted before transform()."
            )

        return (data - self.mean) / self.std

    def fit_transform(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Fit statistics and transform the same data.
        """

        self.fit(data)

        return self.transform(data)