import h5py
import numpy as np

from jet_flavor_classifier.data.dataset import JetDataset
from jet_flavor_classifier.data.features import TRACK_FEATURES
from jet_flavor_classifier.data.loader import create_dataloader


def test_full_pipeline(tmp_path):
    path = tmp_path / "jets.h5"

    rng = np.random.default_rng(42)

    n_jets = 10
    max_tracks = 6

    # Create structured array for tracks with required fields
    track_dtype = np.dtype(
        [('valid', bool)] + [(name, np.float32) for name in TRACK_FEATURES]
    )
    tracks_data = np.zeros((n_jets, max_tracks), dtype=track_dtype)

    # Mark first 4 tracks as valid for each jet, rest as padding
    tracks_data['valid'] = np.tile(
        np.array([True, True, True, True, False, False]),
        (n_jets, 1)
    )

    # Fill track features with random data
    for name in TRACK_FEATURES:
        tracks_data[name] = rng.normal(size=(n_jets, max_tracks)).astype(np.float32)

    labels_array = np.arange(n_jets) % 3

    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks_data)
        # Create a structured dataset for jets with required fields
        event_numbers = np.arange(n_jets)
        
        jets_dtype = np.dtype([
            ('eventNumber', np.int32),
            ('HadronGhostTruthLabelID', np.int64)
        ])
        jets_data = np.zeros(n_jets, dtype=jets_dtype)
        jets_data['eventNumber'] = event_numbers
        jets_data['HadronGhostTruthLabelID'] = labels_array
        f.create_dataset("jets", data=jets_data)

    dataset = JetDataset(str(path))

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))

    # Batch size 4, max valid tracks 4 per jet, 12 track features
    assert batch.tracks.shape == (4, 4, len(TRACK_FEATURES))
    assert batch.mask.shape == (4, 4)
    assert batch.labels.shape == (4,)