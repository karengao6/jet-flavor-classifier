from pathlib import Path

import h5py
import numpy as np

from jet_flavor_classifier.data.features import TRACK_FEATURES


def create_test_file(path: Path) -> None:
    rng = np.random.default_rng(42)

    n_jets = 5
    max_tracks = 6

    labels = np.array(
        [0, 1, 2, 0, 1],
        dtype=np.int64,
    )

    # Create structured array for tracks with required fields
    track_dtype = np.dtype(
        [('valid', bool)] + [(name, np.float32) for name in TRACK_FEATURES]
    )
    tracks = np.zeros((n_jets, max_tracks), dtype=track_dtype)

    # Mark first 4 tracks as valid for each jet, rest as padding
    tracks['valid'] = np.tile(
        np.array([True, True, True, True, False, False]),
        (n_jets, 1)
    )

    # Fill track features with random data
    for name in TRACK_FEATURES:
        tracks[name] = rng.normal(size=(n_jets, max_tracks)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        
        # Create a structured dataset for jets with required fields
        event_numbers = np.zeros(n_jets, dtype=np.int32)
        
        jets_dtype = np.dtype([
            ('eventNumber', np.int32),
            ('HadronGhostTruthLabelID', np.int64)
        ])
        jets_data = np.zeros(n_jets, dtype=jets_dtype)
        jets_data['eventNumber'] = event_numbers
        jets_data['HadronGhostTruthLabelID'] = labels
        f.create_dataset("jets", data=jets_data)