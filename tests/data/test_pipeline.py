import h5py
import numpy as np

from jet_flavor_classifier.data.dataset import JetDataset
from jet_flavor_classifier.data.loader import create_dataloader


def test_full_pipeline(tmp_path):
    path = tmp_path / "jets.h5"

    rng = np.random.default_rng(42)

    tracks = rng.normal(
        size=(10, 6, 4)
    ).astype(np.float32)

    labels = np.arange(10) % 3

    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("labels", data=labels)

    dataset = JetDataset(str(path))

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch.tracks.shape == (4, 6, 4)
    assert batch.mask.shape == (4, 6)
    assert batch.labels.shape == (4,)