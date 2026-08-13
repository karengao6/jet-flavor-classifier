# src/jet_flavor_classifier/data/hdf5.py

from __future__ import annotations

import h5py
import numpy as np


class HDF5Reader:
    """
    Lazy reader for the CERN jet-flavor HDF5 dataset.

    Important properties of the actual file:

        jets:
            shape = [5,619,475]

        tracks:
            shape = [5,619,475, 40]

    Therefore:
        jets[i] and tracks[i] refer to the same jet.

    We never load the entire dataset into memory.
    """

    def __init__(self, path: str) -> None:
        self.path = path

        # The file is opened only when data is first requested.
        self._file: h5py.File | None = None

    def _get_file(self) -> h5py.File:
        """Open the HDF5 file lazily."""

        if self._file is None:
            self._file = h5py.File(self.path, "r")

        return self._file

    def __len__(self) -> int:
        """Return the number of jets."""

        file = self._get_file()

        return len(file["jets"])

    def get_event_numbers(self) -> np.ndarray:
        """
        Return the event number for every jet.

        Used for event-level train/validation/test splitting.
        """

        file = self._get_file()

        return np.asarray(
            file["jets"]["eventNumber"]
        )

    def get_tracks(self, index: int) -> np.ndarray:
        """
        Return all 40 track slots for one jet.

        Shape:
            [40]

        Each element is a structured track record containing
        fields such as pt, eta-related quantities, d0, etc.,
        plus the `valid` field.

        The Dataset will use `valid` to remove invalid slots.
        """

        file = self._get_file()

        return np.asarray(
            file["tracks"][index]
        )

    def get_label(self, index: int) -> int:
        """
        Return the raw jet flavor label.

        Currently using HadronGhostTruthLabelID.

        IMPORTANT:
        Do not use truth-label information as model input.
        This field is the TARGET, not an input feature.
        """

        file = self._get_file()

        return int(
            file["jets"]["HadronGhostTruthLabelID"][index]
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()