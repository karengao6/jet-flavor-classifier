from pathlib import Path

import h5py
import numpy as np


def create_test_file(path: Path) -> None:
    rng = np.random.default_rng(42)

    n_jets = 5
    max_tracks = 6
    n_features = 4

    tracks = rng.normal(
        size=(n_jets, max_tracks, n_features)
    ).astype(np.float32)

    labels = np.array(
        [0, 1, 2, 0, 1],
        dtype=np.int64,
    )

    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("labels", data=labels)