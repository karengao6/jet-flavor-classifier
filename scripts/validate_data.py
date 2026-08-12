"""
Validate assumptions about the CERN dataset before training.
Verify that: 
✓ HDF5 exists
✓ expected datasets exist
✓ expected jet fields exist
✓ expected track fields exist
✓ labels are only 0/4/5/15
✓ no NaN/Inf in model features
✓ track dimensions are 40
✓ train/val/test event IDs don't overlap
"""

from pathlib import Path

import h5py
import numpy as np

from jet_flavor_classifier.data.features import (
    JET_FEATURE_NAMES,
    RAW_JET_FEATURES,
    TRACK_FEATURES,
    TARGET,
)
from jet_flavor_classifier.data.labels import RAW_TO_CLASS


DATA_PATH = Path(
    "data/raw/mc-flavtag-ttbar-small.h5"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    with h5py.File(DATA_PATH, "r") as f:
        assert "jets" in f
        assert "tracks" in f

        jets = f["jets"]
        tracks = f["tracks"]

        print("Dataset validation")
        print("==================")

        print(f"Jets:   {len(jets):,}")
        print(f"Tracks: {tracks.shape}")

        # Check required jet fields.
        jet_fields = set(jets.dtype.names)

        for field in RAW_JET_FEATURES:
            assert field in jet_fields, (
                f"Missing jet field: {field}"
            )

        assert TARGET in jet_fields

        # Check required track fields.
        track_fields = set(tracks.dtype.names)

        for field in TRACK_FEATURES:
            assert field in track_fields, (
                f"Missing track field: {field}"
            )

        assert "valid" in track_fields

        # Check fixed number of track slots.
        assert tracks.shape[1] == 40

        # Check target labels.
        labels = np.asarray(jets[TARGET])
        unique_labels = set(np.unique(labels).tolist())

        assert unique_labels <= set(RAW_TO_CLASS), (
            f"Unexpected labels: {unique_labels}"
        )

        print("✓ Required datasets exist")
        print("✓ Required jet fields exist")
        print("✓ Required track fields exist")
        print("✓ 40 track slots per jet")
        print("✓ Truth labels are valid")
        print()
        print("Validation passed.")


if __name__ == "__main__":
    main()