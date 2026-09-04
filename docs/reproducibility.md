# Reproducibility

## Validated environment

- Model: Qwen2.5-0.5B-Instruct, 494M parameters.
- Architecture: 24 decoder blocks, hidden size 896, gated-MLP size 4864, 14 attention heads, 2 KV heads.
- Tested software: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.41.2.
- GPU: NVIDIA GeForce RTX 5090 D, 32 GB VRAM.
- Experiment seed: 20260904 for the final local-regime gate.

Model revisions and weights are not redistributed. For archival reproduction, record the exact model revision resolved by your model host.

## Activation convention

Transformers `output_hidden_states[i]` is the residual stream entering zero-based block `i`. The main intervention approximated `hidden_states[10] -> hidden_states[11]`; downstream Fisher geometry was evaluated at `hidden_states[11]`, before block 11.

## Data protocol

The validated corpus contained 128 deterministic multi-domain sequences with sequence-disjoint 80/24/24 train/validation/test membership. Its internal SHA-256 was `2e4f1a17d3bb815f1f841d619232996cd4935f3be58a5c00fe29f3eb27124d37`.

The corpus is not redistributed because its source mixture was assembled from local environment text whose redistribution status was not uniformly established. Public reproduction requires a user-supplied, redistributable text file. Do not compare exact values unless sequence membership and model revision match.

## Local Fisher protocol

- Activation dimension: 896.
- Primary fixed causal horizon: H=32.
- Score-VJP probes per token: 384, batch size 16.
- Eigensolver: symmetric FP32 eigendecomposition.
- Retained eigenvectors: 256.
- Empirical covariance rank cap: 383.
- Phase 1: 12 balanced validation positions at token indices 32/96/160.
- Expanded analysis: 96 train, 48 validation, 48 test token positions.
- q grid: 32, 64, 128.
- K grid: 1, 2, 4, 8, 16.
- Clustering: deterministic train-only k-medoids with projection-overlap distance.
- Prototypes: top eigenspace of mean train projection matrix.
- q/K: selected on validation; test opened once for reporting.

The 384-probe covariance can manufacture apparent rank no greater than 383. The study retained only q≤128 for recurrence analysis and reports pointwise r95 as an estimator-dependent sketch statistic, not an exact algebraic rank. The matched aggregate rank is likewise approximate because it reconstructs 256 retained components and omits a small tail.

## Commands

```bash
pip install -e ".[experiments,test]"
pytest -q
python scripts/validate_install.py
python scripts/reproduce_core_geometry.py --quick --output outputs/demo
python scripts/make_public_figures.py
```

## Intentionally excluded artifacts

- Pretrained model weights and tokenizer caches.
- Raw natural-text corpus and token IDs.
- Full hidden-state, KV-cache, and Fisher tensors.
- Temporary LoRA and surrogate checkpoints.
- Actual entropy-coded research streams.
- Machine-specific GPU-worker helpers, hostnames, and absolute paths.

These artifacts are unnecessary for inspecting the algorithms or verifying the synthetic unit tests. Regeneration is documented instead of bundling questionable or large files.

