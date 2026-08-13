import argparse

from jet_flavor_classifier.data.dataset import JetDataset
from jet_flavor_classifier.data.hdf5 import HDF5Reader
from jet_flavor_classifier.data.loader import create_dataloader
from jet_flavor_classifier.data.sampling import select_indices
from jet_flavor_classifier.data.split import create_split_indices
from jet_flavor_classifier.utils.seed import set_seed


def parse_args():
    """Parse command-line configuration."""

    parser = argparse.ArgumentParser()

    # Path to the HDF5 dataset.
    parser.add_argument(
        "--data",
        required=True,
    )

    # Optional subset for quick experiments/debugging.
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    # Number of jets per training batch.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    # Number of DataLoader worker processes.
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    # Controls all random decisions.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Make random behavior reproducible.
    set_seed(args.seed)

    # Create a lightweight HDF5 reader.
    reader = HDF5Reader(args.data)

    # Select the samples for this experiment.
    #
    # Example:
    #   --max-samples 100000
    #
    # means only 100,000 jets will be used.
    indices = select_indices(
        n_samples=len(reader),
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Split the selected samples.
    train_indices, val_indices, test_indices = (
        create_split_indices(
            indices,
            seed=args.seed,
        )
    )

    # Dataset does NOT load everything.
    # Each jet is read when __getitem__() is called.
    train_dataset = JetDataset(
        args.data,
        indices=train_indices,
    )

    val_dataset = JetDataset(
        args.data,
        indices=val_indices,
    )

    test_dataset = JetDataset(
        args.data,
        indices=test_indices,
    )

    # DataLoader handles batching.
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

    # Temporary sanity check.
    #
    # Later this becomes:
    #
    #     batch → model → loss → backward()
    #
    batch = next(iter(train_loader))

    print("tracks:", batch.tracks.shape)
    print("mask:", batch.mask.shape)
    print("labels:", batch.labels.shape)


if __name__ == "__main__":
    main()