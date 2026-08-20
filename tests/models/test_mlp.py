import torch

from jet_flavor_classifier.models.mlp import MLP


def test_mlp_output_shape() -> None:
    """The model should produce one logit per class."""

    model = MLP(
        input_dim=20,
        num_classes=4,
        hidden_dims=(128, 64),
    )

    # Simulate a batch of 32 jets.
    x = torch.randn(32, 20)

    output = model(x)

    assert output.shape == (32, 4)