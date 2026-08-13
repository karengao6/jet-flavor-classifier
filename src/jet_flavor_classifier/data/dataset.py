from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .hdf5 import HDF5Reader
from .normalize import StandardNormalizer
from .types import Jet


class JetDataset(Dataset):
    """
    PyTorch Dataset for jet flavor classification.

    Responsibilities:
        - represent the dataset as an indexable collection
        - lazily retrieve one jet
        - optionally normalize its features
        - return a Jet object

    It does NOT:
        - create batches
        - shuffle data
        - split train/validation/test data
        - run the model
    """

    def __init__(
        self,
        path: str,
        indices: np.ndarray | None = None,
        normalizer: StandardNormalizer | None = None,
    ) -> None:

        self.path = path

        # If indices are supplied, this Dataset represents
        # only those specific jets.
        #
        # This is how train/validation/test subsets are created.
        self.indices = indices

        # Normalizer is fitted using training data and then
        # reused by validation/test datasets.
        self.normalizer = normalizer

        # HDF5Reader performs lazy disk access.
        self.reader = HDF5Reader(path)

    def __len__(self) -> int:
        """
        Number of samples visible through this Dataset.
        """

        if self.indices is not None:
            return len(self.indices)

        return len(self.reader)

    def __getitem__(self, index: int) -> Jet:
        """
        Retrieve one jet.

        Flow:
            Dataset index
                ↓
            actual HDF5 index
                ↓
            read tracks + label
                ↓
            normalize
                ↓
            convert to PyTorch tensors
                ↓
            return Jet
        """

        # Convert Dataset-local index into the actual HDF5 index.
        actual_index = (
            self.indices[index]
            if self.indices is not None
            else index
        )

        # Read only this jet from disk.
        tracks = self.reader.get_tracks(actual_index)
        label = self.reader.get_label(actual_index)

        # Convert HDF5/NumPy data to float32.
        tracks = np.asarray(
            tracks,
            dtype=np.float32,
        )

        # Normalize track features if a normalizer was provided.
        if self.normalizer is not None:
            original_shape = tracks.shape

            # Treat every track as one row:
            #
            # [num_tracks, num_features]
            #
            # This makes normalization operate feature-by-feature.
            tracks = tracks.reshape(
                -1,
                tracks.shape[-1],
            )

            tracks = self.normalizer.transform(tracks)

            # Restore [num_tracks, num_features].
            tracks = tracks.reshape(original_shape)

        return Jet(
            tracks=torch.from_numpy(tracks),
            label=torch.tensor(
                label,
                dtype=torch.long,
            ),
        )