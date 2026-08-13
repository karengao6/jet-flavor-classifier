'''
This is where the masks get created for variable-length jets. 
The mask is a boolean tensor that indicates which tracks are real and which are padding.
'''
from __future__ import annotations

import torch

from .types import Jet, JetBatch


def collate_jets(batch: list[Jet]) -> JetBatch:
    """
    Convert a list of variable-length Jets into one padded batch.

    Example input:

        Jet 1 -> 3 tracks
        Jet 2 -> 5 tracks
        Jet 3 -> 2 tracks

    Output:

        tracks -> [3, 5, num_features]
        mask   -> [3, 5]
        labels -> [3]
    """

    batch_size = len(batch)

    # Find the largest number of tracks among the jets in this batch.
    #
    # We only pad to the largest jet in THIS batch rather than some
    # global maximum. This saves memory.
    max_tracks = max(
        jet.tracks.shape[0]
        for jet in batch
    )

    # Every track should have the same number of features.
    n_features = batch[0].tracks.shape[1]

    # Allocate the padded track tensor.
    #
    # Everything starts as zero, so unused positions automatically
    # contain zero-padding.
    tracks = torch.zeros(
        batch_size,
        max_tracks,
        n_features,
        dtype=torch.float32,
    )

    # Allocate the mask.
    #
    # Everything starts as False, meaning "padding".
    #
    # We'll change positions corresponding to real tracks to True.
    mask = torch.zeros(
        batch_size,
        max_tracks,
        dtype=torch.bool,
    )

    # One label per jet.
    labels = torch.empty(
        batch_size,
        dtype=torch.long,
    )

    for i, jet in enumerate(batch):
        # Number of REAL tracks in this particular jet.
        n_tracks = jet.tracks.shape[0]

        # Copy the real tracks into the beginning of the padded tensor.
        #
        # Example:
        #   jet has 3 tracks
        #   max_tracks = 5
        #
        #   [track1, track2, track3, 0, 0]
        tracks[i, :n_tracks] = jet.tracks

        # Mark those positions as real.
        #
        # Example:
        #   [True, True, True, False, False]
        mask[i, :n_tracks] = True

        # Store this jet's class label.
        labels[i] = jet.label

    return JetBatch(
        tracks=tracks,
        mask=mask,
        labels=labels,
    )