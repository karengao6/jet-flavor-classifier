from pathlib import Path

import h5py
import numpy as np

from jet_flavor_classifier.data.split import split_by_event


DATA_PATH = Path("data/raw/mc-flavtag-ttbar-small.h5")

LABEL_FIELD = "HadronGhostTruthLabelID"

CLASS_NAMES = {
    0: "light",
    4: "charm",
    5: "bottom",
    15: "tau",
}


def print_split_summary(
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Print class counts and percentages for each split."""

    splits = [
        ("Train", train_indices),
        ("Validation", validation_indices),
        ("Test", test_indices),
    ]

    for name, indices in splits:
        split_labels = labels[indices]

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Jets: {len(indices):,}")

        values, counts = np.unique(
            split_labels,
            return_counts=True,
        )

        for value, count in zip(values, counts):
            class_name = CLASS_NAMES.get(
                int(value),
                f"unknown ({value})",
            )

            percentage = count / len(indices) * 100

            print(
                f"  {class_name:10s}: "
                f"{count:>10,} "
                f"({percentage:5.2f}%)"
            )


def main() -> None:
    print(f"Loading: {DATA_PATH}")

    with h5py.File(DATA_PATH, "r") as f:
        jets = f["jets"]

        event_numbers = np.asarray(
            jets["eventNumber"]
        )

        labels = np.asarray(
            jets[LABEL_FIELD]
        )

    print(f"Total jets: {len(labels):,}")
    print(
        f"Unique events: "
        f"{len(np.unique(event_numbers)):,}"
    )

    train_indices, validation_indices, test_indices = (
        split_by_event(
            event_numbers,
            test_size=0.15,
            validation_size=0.15,
            random_state=42,
        )
    )

    print("\nJet split:")
    print(
        f"  Train:      {len(train_indices):,} "
        f"({len(train_indices) / len(labels):.1%})"
    )
    print(
        f"  Validation: {len(validation_indices):,} "
        f"({len(validation_indices) / len(labels):.1%})"
    )
    print(
        f"  Test:       {len(test_indices):,} "
        f"({len(test_indices) / len(labels):.1%})"
    )

    # Verify every jet belongs to exactly one split.
    all_indices = np.concatenate(
        [
            train_indices,
            validation_indices,
            test_indices,
        ]
    )

    assert len(all_indices) == len(labels)
    assert len(np.unique(all_indices)) == len(labels)

    print("\n✓ Every jet belongs to exactly one split.")

    print_split_summary(
        labels,
        train_indices,
        validation_indices,
        test_indices,
    )


if __name__ == "__main__":
    main()