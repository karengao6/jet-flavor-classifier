# src/jet_flavor_classifier/data/loader.py

from torch.utils.data import DataLoader

from .collate import collate_jets
from .dataset import JetDataset


def create_dataloader(
    dataset: JetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader with the project's standard configuration.
    """

    return DataLoader(
        dataset,

        # Number of jets per batch.
        batch_size=batch_size,

        # Shuffle training data.
        shuffle=shuffle,

        # Parallel data-loading workers.
        num_workers=num_workers,

        # Our custom function handles variable-length tracks.
        collate_fn=collate_jets,

        # Makes CPU → CUDA transfer more efficient later.
        pin_memory=True,
    )