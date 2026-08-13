from dataclasses import dataclass

import torch


@dataclass
class Jet:
    """One jet returned by the Dataset."""

    # Shape: [num_tracks, num_features]
    #
    # num_tracks can be different for every jet.
    # Example: [17, 8] or [42, 8].
    tracks: torch.Tensor

    # Integer class label for this jet.
    # Example: 0 = light, 1 = charm, 2 = bottom, 3 = tau.
    label: torch.Tensor


@dataclass
class JetBatch:
    """A batch of jets produced by the DataLoader.
    For example, suppose a batch has:

    Jet 1: 3 tracks
    Jet 2: 5 tracks
    Jet 3: 2 tracks

    The batch must have one common max_tracks = 5 dimension:

    tracks:

    Jet 1: [track1, track2, track3, PAD,   PAD]
    Jet 2: [track1, track2, track3, track4, track5]
    Jet 3: [track1, track2, PAD,   PAD,   PAD]

    The mask is:

    Jet 1: [True, True, True, False, False]
    Jet 2: [True, True, True, True,  True]
    Jet 3: [True, True, False, False, False]
    """

    # Padded track tensor.
    #
    # Shape: [batch_size, max_tracks, num_features]
    #
    # Because jets have different numbers of tracks, shorter
    # jets are padded with zeros up to max_tracks.
    tracks: torch.Tensor

    # Boolean tensor identifying which entries in `tracks` are real.
    #
    # Shape: [batch_size, max_tracks]
    #
    # True  = real track, False = padding
    #
    # The model uses this to ignore padded tracks.
    mask: torch.Tensor

    # Class labels for each jet.
    #
    # Shape: [batch_size]
    labels: torch.Tensor