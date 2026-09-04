#!/usr/bin/env python3
"""Reproduce a small pointwise downstream-Fisher spectrum.

Quick mode is an implementation demo, not an exact reproduction of the
validated 384-probe, multi-context experiment.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from functional_geometry.fisher import fisher_from_score_gradients
from functional_geometry.subspaces import cumulative_energy, rank_for_energy
from functional_geometry.utils import seed_everything, write_json


def replace_primary(original: object, hidden: torch.Tensor) -> object:
    return (hidden, *original[1:]) if isinstance(original, tuple) else hidden


def estimate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_index: int,
    position: int,
    horizon: int,
    probes: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    """Estimate one local Fisher using batched categorical-score VJPs."""

    layer = model.model.layers[layer_index]
    holder: dict[str, torch.Tensor] = {}

    def capture(_module: object, _args: object, output: object) -> object:
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden.detach().requires_grad_(True)
        holder["hidden"] = leaf
        return replace_primary(output, leaf)

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(input_ids, use_cache=False).logits.float()[0, position : position + horizon]
    finally:
        handle.remove()

    probabilities = F.softmax(logits, dim=-1).detach()
    hidden = holder["hidden"]
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    gradients = []
    for offset in range(0, probes, batch_size):
        count = min(batch_size, probes - offset)
        samples = torch.multinomial(probabilities, count, replacement=True, generator=generator).T
        scores = -probabilities.unsqueeze(0).expand(count, -1, -1).clone()
        scores.scatter_add_(2, samples.unsqueeze(-1), torch.ones_like(samples.unsqueeze(-1), dtype=scores.dtype))
        scores /= math.sqrt(horizon)
        gradient = torch.autograd.grad(
            logits,
            hidden,
            grad_outputs=scores,
            is_grads_batched=True,
            retain_graph=offset + count < probes,
        )[0][:, 0, position]
        gradients.append(gradient.float().cpu())

    spectrum = fisher_from_score_gradients(torch.cat(gradients), retain=min(256, probes - 1))
    energy = spectrum.eigenvalues.cumsum(0) / spectrum.trace
    def trace_rank(fraction: float) -> int | None:
        hits = torch.nonzero(energy >= fraction)
        return int(hits[0]) + 1 if hits.numel() else None
    return {
        "position": position,
        "horizon": horizon,
        "probe_count": probes,
        "rank_cap": spectrum.rank_cap,
        "trace": spectrum.trace,
        "trace_standard_error": spectrum.trace_standard_error,
        "r50_retained": trace_rank(0.50),
        "r90_retained": trace_rank(0.90),
        "r95_retained": trace_rank(0.95),
        "eigenvalues": spectrum.eigenvalues.tolist(),
        "cumulative_retained_energy": energy.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer", type=int, default=10, help="Zero-based block whose output is analyzed")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--probes", type=int, default=384)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--position", type=int, default=32)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.probes = min(args.probes, 32)
        args.probe_batch_size = min(args.probe_batch_size, 8)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for the model-level demonstration.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.text_file:
        text = next(line.strip() for line in args.text_file.read_text().splitlines() if line.strip())
    else:
        seed_text = (
            "A controlled experiment compares a hypothesis against its strongest simple baseline. "
            "Negative evidence is recorded before the next question is tested. "
        )
        text = seed_text * 24

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    needed = args.position + args.horizon + 1
    if input_ids.shape[1] < needed:
        raise SystemExit(f"Input has {input_ids.shape[1]} tokens; at least {needed} are required.")
    result = estimate(
        model,
        input_ids,
        args.layer,
        args.position,
        args.horizon,
        args.probes,
        args.probe_batch_size,
        args.seed,
    )
    result.update({
        "model": args.model,
        "zero_based_block": args.layer,
        "mode": "QUICK_DEMO_APPROXIMATE" if args.quick else "FULL_SINGLE_TOKEN_ESTIMATE",
        "warning": "A single context is not equivalent to the validated multi-sequence experiment.",
    })
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "spectrum.json", result)

    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, len(result["cumulative_retained_energy"]) + 1), result["cumulative_retained_energy"])
    axis.axhline(0.95, color="0.4", linestyle="--")
    axis.set(xlabel="Retained directions", ylabel="Cumulative retained energy", title=result["mode"])
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output / "spectrum.png", dpi=180)
    print(json.dumps({key: result[key] for key in ["mode", "r50_retained", "r90_retained", "r95_retained", "rank_cap"]}, indent=2))


if __name__ == "__main__":
    main()
