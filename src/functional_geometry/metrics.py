"""Functional fidelity metrics used by the experiments."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def categorical_kl(reference_logits: Tensor, candidate_logits: Tensor) -> Tensor:
    """Tokenwise ``KL(reference || candidate)``."""

    log_p = F.log_softmax(reference_logits.float(), dim=-1)
    log_q = F.log_softmax(candidate_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def delta_nll(reference_logits: Tensor, candidate_logits: Tensor, labels: Tensor) -> Tensor:
    """Candidate minus reference mean negative log likelihood."""

    return F.cross_entropy(candidate_logits.float(), labels) - F.cross_entropy(reference_logits.float(), labels)


def functional_energy_capture(
    eigenvalues: Tensor, eigenvectors: Tensor, candidate_subspace: Tensor, *, total_trace: float | None = None
) -> float:
    """Fraction of Fisher energy captured by a candidate subspace."""

    trace = float(eigenvalues.sum()) if total_trace is None else total_trace
    coordinates = eigenvectors.T @ candidate_subspace
    return float((coordinates.square() * eigenvalues[:, None]).sum() / trace)

