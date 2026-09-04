"""Categorical Fisher operations and stochastic score-VJP estimation.

The central estimator never forms a vocabulary-by-hidden-state Jacobian.
Instead, callers produce score gradients with respect to a hidden vector. Their
sample covariance estimates the pullback categorical Fisher.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class FisherSpectrum:
    """Compact spectrum plus trace-estimation diagnostics."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    trace: float
    trace_standard_error: float
    probe_count: int
    rank_cap: int


def categorical_fisher_product(probabilities: Tensor, vector: Tensor) -> Tensor:
    """Apply ``diag(p) - p p^T`` without constructing the matrix."""

    if probabilities.shape != vector.shape:
        raise ValueError("probabilities and vector must have identical shapes")
    centered = (probabilities * vector).sum(dim=-1, keepdim=True)
    return probabilities * (vector - centered)


def categorical_score_vectors(
    probabilities: Tensor, samples: Tensor
) -> Tensor:
    """Return categorical score vectors ``one_hot(sample) - p``.

    ``probabilities`` has shape ``[..., classes]`` and ``samples`` has shape
    ``[probes, ...]``. The result has shape ``[probes, ..., classes]``.
    """

    probes = samples.shape[0]
    scores = -probabilities.unsqueeze(0).expand(probes, *probabilities.shape).clone()
    return scores.scatter_add_(-1, samples.unsqueeze(-1), torch.ones_like(samples.unsqueeze(-1), dtype=scores.dtype))


def fisher_from_score_gradients(
    gradients: Tensor, *, retain: int | None = None, center: bool = True
) -> FisherSpectrum:
    """Estimate a hidden-space Fisher from score-gradient probes.

    Args:
        gradients: ``[probes, hidden_dim]`` score gradients.
        retain: Number of leading eigenvectors to retain. Defaults to all.
        center: Remove the finite-sample mean score gradient.

    The empirical covariance rank is at most ``probes - 1`` when centered.
    Code using this function must not infer low rank beyond that cap.
    """

    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must have shape [at least 2 probes, hidden_dim]")
    g = gradients.to(dtype=torch.float64)
    if center:
        g = g - g.mean(dim=0, keepdim=True)
    fisher = g.T @ g / (g.shape[0] - int(center))
    fisher = (fisher + fisher.T) / 2
    values, vectors = torch.linalg.eigh(fisher)
    values = values.clamp_min(0).flip(0)
    vectors = vectors.flip(1)
    keep = values.numel() if retain is None else min(retain, values.numel())
    norms = g.square().sum(dim=1)
    return FisherSpectrum(
        eigenvalues=values[:keep].to(torch.float32),
        eigenvectors=vectors[:, :keep].to(torch.float32),
        trace=float(values.sum()),
        trace_standard_error=float(norms.std(unbiased=True) / norms.numel() ** 0.5),
        probe_count=gradients.shape[0],
        rank_cap=gradients.shape[0] - int(center),
    )


def fisher_matrix_vector(spectrum: FisherSpectrum, vector: Tensor) -> Tensor:
    """Apply a retained eigendecomposition to a hidden-space vector."""

    u = spectrum.eigenvectors.to(vector)
    values = spectrum.eigenvalues.to(vector)
    return u @ (values * (u.T @ vector))

