from __future__ import annotations

import torch

from .types import Jet, JetBatch


def collate_jets(
    batch: list[Jet],
) -> JetBatch:
    """
    Combine variable-length Jets into one rectangular batch.

    Example:

        Jet 1 → 23 tracks
        Jet 2 → 37 tracks
        Jet 3 → 8 tracks

    Batch becomes:

        tracks → [3, 37, num_features]
        mask   → [3, 37]

    The mask tells the model which positions are real tracks.
    """

    batch_size = len(batch)

    # Find the largest number of VALID tracks in this batch.
    max_tracks = max(
        jet.tracks.shape[0]
        for jet in batch
    )

    n_features = batch[0].tracks.shape[1]

    # Zero-padded track tensor.
    tracks = torch.zeros(
        batch_size,
        max_tracks,
        n_features,
        dtype=torch.float32,
    )

    # False initially means padding.
    mask = torch.zeros(
        batch_size,
        max_tracks,
        dtype=torch.bool,
    )

    labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    for i, jet in enumerate(batch):

        n_tracks = jet.tracks.shape[0]

        # Copy real tracks.
        tracks[i, :n_tracks] = jet.tracks

        # Mark those positions as real.
        mask[i, :n_tracks] = True

        labels[i] = jet.label

    return JetBatch(
        tracks=tracks,
        mask=mask,
        labels=labels,
    )