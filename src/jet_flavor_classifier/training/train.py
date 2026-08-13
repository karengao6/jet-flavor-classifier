# src/jet_flavor_classifier/training/train.py

import argparse

from jet_flavor_classifier.data.dataset import JetDataset
from jet_flavor_classifier.data.hdf5_reader import HDF5Reader
from jet_flavor_classifier.data.loader import create_dataloader
from jet_flavor_classifier.data.sampling import limit_indices
from jet_flavor_classifier.data.split import split_by_event
from jet_flavor_classifier.utils.seed import set_seed


def parse_args():
    """
    Define command-line arguments.

    Example:

        uv run python -m jet_flavor_classifier.training.train \
            --data data/jets.h5 \
            --max-samples 1000 \
            --batch-size 32 \
            --seed 42
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the HDF5 dataset.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of jets to use per split. "
            "If omitted, use all jets."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Make random operations reproducible.
    set_seed(args.seed)

    # Lazy HDF5 reader.
    reader = HDF5Reader(args.data)

    print(f"Total jets: {len(reader):,}")

    # eventNumber is loaded as a small 1D array.
    #
    # We need this to split jets by event.
    event_numbers = reader.get_event_numbers()

    # ---------------------------------------------------------
    # EVENT-LEVEL SPLIT
    # ---------------------------------------------------------
    #
    # This happens BEFORE max_samples.
    #
    # Therefore jets from one event can never be split
    # between train/validation/test.
    train_indices, val_indices, test_indices = split_by_event(
        event_numbers,
        seed=args.seed,
    )

    print(
        f"Train before limit: {len(train_indices):,}"
    )
    print(
        f"Validation before limit: {len(val_indices):,}"
    )
    print(
        f"Test before limit: {len(test_indices):,}"
    )

    # ---------------------------------------------------------
    # OPTIONAL SUBSAMPLING
    # ---------------------------------------------------------
    #
    # --max-samples 1000 means:
    #
    #   up to 1000 training jets
    #   up to 1000 validation jets
    #   up to 1000 test jets
    #
    # The seed offsets ensure each split gets a different
    # deterministic random sample.
    train_indices = limit_indices(
        train_indices,
        args.max_samples,
        args.seed,
    )

    val_indices = limit_indices(
        val_indices,
        args.max_samples,
        args.seed + 1,
    )

    test_indices = limit_indices(
        test_indices,
        args.max_samples,
        args.seed + 2,
    )

    print(
        f"Train after limit: {len(train_indices):,}"
    )
    print(
        f"Validation after limit: {len(val_indices):,}"
    )
    print(
        f"Test after limit: {len(test_indices):,}"
    )

    # ---------------------------------------------------------
    # DATASETS
    # ---------------------------------------------------------

    train_dataset = JetDataset(
        args.data,
        train_indices,
    )

    val_dataset = JetDataset(
        args.data,
        val_indices,
    )

    test_dataset = JetDataset(
        args.data,
        test_indices,
    )

    # ---------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------------------------------------------------------
    # PIPELINE SANITY CHECK
    # ---------------------------------------------------------

    batch = next(iter(train_loader))

    print("\nFirst batch:")
    print(f"  tracks: {batch.tracks.shape}")
    print(f"  mask:   {batch.mask.shape}")
    print(f"  labels: {batch.labels.shape}")

    # Example:
    #
    # tracks: [32, 37, 17]
    # mask:   [32, 37]
    # labels: [32]
    #
    # 37 is the largest number of VALID tracks in this batch.
    # 17 is the number of selected track features.


if __name__ == "__main__":
    main()