#!/usr/bin/env python3
"""Fast installation smoke test; downloads no models."""

import torch
from functional_geometry.fisher import categorical_fisher_product
from functional_geometry.subspaces import projection_overlap

p = torch.tensor([0.2, 0.3, 0.5])
v = torch.tensor([1.0, -1.0, 0.25])
assert v @ categorical_fisher_product(p, v) >= 0
eye = torch.eye(4)
assert abs(projection_overlap(eye[:, :2], eye[:, :2]) - 1.0) < 1e-6
print("functional_geometry installation validated")

