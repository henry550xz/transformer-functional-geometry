"""Tools for studying downstream-visible geometry in neural representations."""

from .fisher import categorical_fisher_product, fisher_from_score_gradients
from .metrics import categorical_kl, functional_energy_capture
from .subspaces import cumulative_energy, effective_ranks, projection_overlap

__all__ = [
    "categorical_fisher_product",
    "fisher_from_score_gradients",
    "categorical_kl",
    "functional_energy_capture",
    "cumulative_energy",
    "effective_ranks",
    "projection_overlap",
]

