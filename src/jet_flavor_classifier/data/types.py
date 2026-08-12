from dataclasses import dataclass

import torch


@dataclass
class Jet:
    tracks: torch.Tensor
    label: torch.Tensor


@dataclass
class JetBatch:
    tracks: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor