from torch.utils.data import DataLoader

from .collate import collate_jets
from .dataset import JetDataset


def create_dataloader(
    dataset: JetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_jets,
        pin_memory=True,
    )