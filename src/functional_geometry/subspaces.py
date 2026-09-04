"""Spectrum summaries and Grassmannian subspace operations."""

from __future__ import annotations
import torch
from torch import Tensor


def cumulative_energy(eigenvalues: Tensor) -> Tensor:
    """Return normalized cumulative energy for descending eigenvalues."""

    values = eigenvalues.clamp_min(0)
    total = values.sum()
    if total <= 0:
        raise ValueError("eigenvalues must have positive total energy")
    return values.cumsum(0) / total


def rank_for_energy(eigenvalues: Tensor, fraction: float) -> int | None:
    """Smallest one-based rank capturing ``fraction`` of known energy."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    hits = torch.nonzero(cumulative_energy(eigenvalues) >= fraction)
    return int(hits[0]) + 1 if hits.numel() else None


def effective_ranks(eigenvalues: Tensor) -> dict[str, float]:
    """Participation and entropy effective ranks."""

    values = eigenvalues.clamp_min(0).to(torch.float64)
    probabilities = values / values.sum()
    positive = probabilities > 0
    return {
        "participation": float(values.sum().square() / values.square().sum()),
        "entropy": float(torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum())),
    }


def projection_overlap(left: Tensor, right: Tensor) -> float:
    """Subspace-invariant overlap ``||U^T V||_F^2 / q``."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("subspace bases must have the same [dimension, rank] shape")
    return float(torch.linalg.matrix_norm(left.T @ right).square() / left.shape[1])


def grassmann_distance(left: Tensor, right: Tensor) -> float:
    """Projection-overlap distance used by the public experiments."""

    return 1.0 - projection_overlap(left, right)


def mean_projection_prototype(subspaces: list[Tensor], rank: int) -> Tensor:
    """Top eigenspace of the mean projection matrix."""

    if not subspaces:
        raise ValueError("at least one subspace is required")
    projection = sum(u @ u.T for u in subspaces) / len(subspaces)
    _, vectors = torch.linalg.eigh((projection + projection.T) / 2)
    return vectors[:, -rank:].flip(1)

