import torch
from functional_geometry.metrics import functional_energy_capture
from functional_geometry.subspaces import cumulative_energy, effective_ranks, mean_projection_prototype, projection_overlap, rank_for_energy


def test_cumulative_energy_and_ranks():
    values = torch.tensor([4.0, 2.0, 1.0, 1.0])
    assert torch.allclose(cumulative_energy(values), torch.tensor([0.5, 0.75, 0.875, 1.0]))
    assert rank_for_energy(values, 0.75) == 2
    assert effective_ranks(values)["participation"] > 2


def test_projection_overlap_extremes():
    eye = torch.eye(6)
    assert abs(projection_overlap(eye[:, :2], eye[:, :2]) - 1) < 1e-6
    assert abs(projection_overlap(eye[:, :2], eye[:, 2:4])) < 1e-6


def test_prototype_and_functional_capture():
    eye = torch.eye(5)
    prototype = mean_projection_prototype([eye[:, :2], eye[:, :2]], 2)
    capture = functional_energy_capture(torch.tensor([4.0, 2.0]), eye[:, :2], prototype, total_trace=6.0)
    assert capture > 0.999

