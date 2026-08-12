from __future__ import annotations

import h5py


class HDF5Reader:
    """Lazy reader for jet data stored in HDF5."""

    def __init__(
        self,
        path: str,
        tracks_key: str = "tracks",
        labels_key: str = "labels",
    ) -> None:
        self.path = path
        self.tracks_key = tracks_key
        self.labels_key = labels_key

        # Don't open the file until data is actually requested.
        self._file = None

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")

        return self._file

    def __len__(self) -> int:
        file = self._get_file()
        return len(file[self.labels_key])

    def get_tracks(self, index: int):
        file = self._get_file()
        return file[self.tracks_key][index]

    def get_label(self, index: int):
        file = self._get_file()
        return file[self.labels_key][index]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()