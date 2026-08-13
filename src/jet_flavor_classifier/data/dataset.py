# src/jet_flavor_classifier/data/dataset.py

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import TRACK_FEATURES
from .hdf5_reader import HDF5Reader
from .normalize import StandardNormalizer
from .types import Jet


class JetDataset(Dataset):
    """
    PyTorch Dataset for jet flavor classification.

    The Dataset:
        - lazily reads jets from HDF5
        - filters invalid tracks
        - extracts selected features
        - optionally normalizes features
        - returns Jet objects

    `indices` identifies which jets belong to this Dataset.
    """

    def __init__(
        self,
        path: str,
        indices: np.ndarray | None = None,
        normalizer: StandardNormalizer | None = None,
    ) -> None:

        self.reader = HDF5Reader(path)

        # If no indices are supplied, expose the whole HDF5 dataset.
        #
        # Training code SHOULD supply explicit split indices.
        if indices is None:
            self.indices = np.arange(len(self.reader))
        else:
            self.indices = np.asarray(
                indices,
                dtype=np.int64,
            )

        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Jet:
        jet_index = self.indices[index]

        tracks = self.reader.get_tracks(jet_index)

        # HDF5 `valid` identifies real track slots.
        valid = tracks["valid"]

        # Remove invalid/padded track slots.
        tracks = tracks[valid]

        # Extract only model input features.
        features = np.column_stack(
            [
                tracks[name]
                for name in TRACK_FEATURES
            ]
        ).astype(np.float32)

        # Apply training-set statistics if supplied.
        if self.normalizer is not None:
            features = self.normalizer.transform(features)

        label = self.reader.get_label(jet_index)

        return Jet(
            tracks=torch.from_numpy(features),
            label=torch.tensor(
                label,
                dtype=torch.long,
            ),
        )