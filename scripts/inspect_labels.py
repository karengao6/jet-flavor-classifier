"""
Inspect the truth-label distributions in the CERN jet-flavor dataset.

Truth labels are NOT model inputs.

They are used only as the target that the model learns to predict.
"""

from collections import Counter
from pathlib import Path

import h5py
import numpy as np


DATA_PATH = Path("data/raw/mc-flavtag-ttbar-small.h5")


LABEL_FIELDS = (
    "HadronGhostTruthLabelID",
    "HadronGhostExtendedTruthLabelID",
    "HadronConeExclTruthLabelID",
    "HadronConeExclExtendedTruthLabelID",
    "PartonTruthLabelID",
)


def main() -> None:
    print(f"Looking for dataset at:")
    print(f"  {DATA_PATH}\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {DATA_PATH}"
        )

    print(f"File exists: {DATA_PATH.stat().st_size:,} bytes\n")

    with h5py.File(DATA_PATH, "r") as f:
        print("Successfully opened HDF5 file.\n")

        print("Available jet fields:")
        for name in f["jets"].dtype.names:
            print(f"  {name}")

        print()

        for field in LABEL_FIELDS:
            print(f"Inspecting: {field}")

            if field not in f["jets"].dtype.names:
                print("  FIELD NOT FOUND\n")
                continue

            values = np.asarray(f["jets"][field])

            print(f"  Number of values: {len(values):,}")
            print(f"  dtype: {values.dtype}")
            print(f"  min: {values.min()}")
            print(f"  max: {values.max()}")

            counts = Counter(values.tolist())

            print(f"  unique values: {len(counts)}\n")

            for label, count in sorted(counts.items()):
                percentage = count / len(values) * 100

                print(
                    f"    {label:>4}: "
                    f"{count:>10,} "
                    f"({percentage:>6.2f}%)"
                )

            print()


if __name__ == "__main__":
    print("Starting label inspection...")
    main()
    print("Finished.")