import torch
from torch import nn
from functional_geometry.interventions import residual_output_intervention, single_token_addition


def test_zero_intervention_and_cleanup():
    layer = nn.Linear(4, 4, bias=False)
    inputs = torch.randn(1, 3, 4)
    baseline = layer(inputs)
    with residual_output_intervention(layer, single_token_addition(1, torch.zeros(4))):
        assert torch.equal(layer(inputs), baseline)
    assert torch.equal(layer(inputs), baseline)

