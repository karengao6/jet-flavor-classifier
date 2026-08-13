from __future__ import annotations

import h5py


class HDF5Reader:
    """
    Lazily reads jet data from an HDF5 file.

    IMPORTANT:
    Do not load the entire dataset into memory.

    Instead, read only the jet requested by __getitem__().
    """

    def __init__(
        self,
        path: str,
        tracks_key: str = "tracks",
        labels_key: str = "labels",
    ) -> None:
        self.path = path
        self.tracks_key = tracks_key
        self.labels_key = labels_key

        # The file is opened lazily.
        #
        # This is especially useful when PyTorch creates
        # multiple DataLoader worker processes.
        self._file: h5py.File | None = None

    def _get_file(self) -> h5py.File:
        """
        Open the HDF5 file only when it is actually needed.
        """

        if self._file is None:
            self._file = h5py.File(self.path, "r")

        return self._file

    def __len__(self) -> int:
        """
        Return the number of jets without loading them into memory.
        """

        file = self._get_file()

        return len(file[self.labels_key])

    def get_tracks(self, index: int):
        """
        Read ONLY one jet's tracks from disk.
        """

        file = self._get_file()

        return file[self.tracks_key][index]

    def get_label(self, index: int):
        """
        Read ONLY one jet's label from disk.
        """

        file = self._get_file()

        return file[self.labels_key][index]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()