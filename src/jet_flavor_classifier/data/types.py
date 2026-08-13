# src/jet_flavor_classifier/data/types.py

from dataclasses import dataclass

import torch


@dataclass
class Jet:
    """
    One jet returned by the Dataset.

    The original HDF5 file stores up to 40 track slots per jet.
    The Dataset removes invalid slots using the HDF5 `valid` field.

    Therefore this object can contain a variable number of tracks.

    tracks:
        Shape [num_valid_tracks, num_features].

    label:
        Integer classification label for the jet.
    """

    tracks: torch.Tensor
    label: torch.Tensor


@dataclass
class JetBatch:
    """
    A batch of jets produced by the DataLoader.

    Because jets have different numbers of valid tracks,
    collate_jets() pads them to the largest number of tracks
    in the batch.

    tracks:
        Shape [batch_size, max_tracks, num_features].

    mask:
        Shape [batch_size, max_tracks].

        True  = real/valid track
        False = padding

    labels:
        Shape [batch_size].
    """

    tracks: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor