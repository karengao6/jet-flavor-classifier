import torch

from jet_flavor_classifier.data.collate import collate_jets
from jet_flavor_classifier.data.types import Jet


def test_collate_variable_length_jets():
    """
    Verify that variable-length jets are padded correctly
    and that the mask identifies real tracks.
    """
    batch = [
        Jet(
            tracks=torch.ones(3, 4),
            label=torch.tensor(0),
        ),
        Jet(
            tracks=torch.ones(5, 4),
            label=torch.tensor(1),
        ),
        Jet(
            tracks=torch.ones(2, 4),
            label=torch.tensor(2),
        ),
    ]

    result = collate_jets(batch)

    assert result.tracks.shape == (3, 5, 4)
    assert result.mask.shape == (3, 5)
    assert result.labels.shape == (3,)

    assert result.mask[0].tolist() == [
        True, True, True, False, False
    ]

    assert result.mask[1].tolist() == [
        True, True, True, True, True
    ]

    assert result.mask[2].tolist() == [
        True, True, False, False, False
    ]