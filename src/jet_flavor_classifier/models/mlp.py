"""Multi-layer perceptron for jet flavor classification."""

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Configurable feed-forward neural network."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        """
        Initialize the MLP.

        Parameters
        ----------
        input_dim:
            Number of input features per jet.

        num_classes:
            Number of output classes.

        hidden_dims:
            Number of neurons in each hidden layer.

        dropout:
            Dropout probability applied after hidden layers.

        batch_norm:
            Whether to use batch normalization after linear layers.
        """
        super().__init__()

        layers: list[nn.Module] = []

        # Keep track of the number of features entering each layer.
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            # Learn a linear transformation:
            #
            # current_dim features -> hidden_dim features
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Batch normalization can make training more stable.
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Introduce nonlinearity.
            layers.append(nn.ReLU())

            # Dropout randomly disables some activations during training.
            # This can reduce overfitting.
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        # Final layer produces one logit for each class.
        #
        # IMPORTANT:
        # We do NOT apply softmax here.
        # CrossEntropyLoss expects raw logits.
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of jets through the network."""
        return self.network(x)