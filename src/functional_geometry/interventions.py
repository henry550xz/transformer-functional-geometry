"""Small, explicit residual-stream intervention utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator
import torch
from torch import Tensor, nn


def replace_primary_output(original: object, hidden: Tensor) -> object:
    """Preserve auxiliary module outputs while replacing its hidden state."""

    if isinstance(original, tuple):
        return (hidden, *original[1:])
    return hidden


@contextmanager
def residual_output_intervention(
    module: nn.Module, transform: Callable[[Tensor], Tensor]
) -> Iterator[None]:
    """Temporarily transform the primary output of a residual-stream module."""

    def hook(_module: nn.Module, _args: tuple[object, ...], output: object) -> object:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, Tensor):
            raise TypeError("module primary output is not a Tensor")
        return replace_primary_output(output, transform(hidden))

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def single_token_addition(position: int, error: Tensor) -> Callable[[Tensor], Tensor]:
    """Create a transform that adds ``error`` at one sequence position."""

    def transform(hidden: Tensor) -> Tensor:
        changed = hidden.clone()
        changed[..., position, :] += error.to(changed)
        return changed

    return transform

