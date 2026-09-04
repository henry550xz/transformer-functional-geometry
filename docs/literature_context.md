# Literature context

This repository is an empirical exploration, not a claim that Fisher geometry, activation-aware compression, teacher–student training, or KV quantization is new. Several components have close conceptual prior work.

- **Information geometry and pullbacks.** The hidden-space metric studied here is a pullback of the categorical output Fisher. Arvanitidis et al., [*Pulling back information geometry*](https://proceedings.mlr.press/v151/arvanitidis22b.html) (AISTATS 2022), develops Fisher–Rao pullback geometry for decoder distributions.
- **Per-example Fisher structure.** Matena and Raffel, [*NPEFF: Non-Negative Per-Example Fisher Factorization*](https://arxiv.org/abs/2310.04649) (2023), factor per-example parameter-space Fisher representations for interpretability. The present activation-space study asks a narrower empirical question about local downstream-visible subspaces.
- **Fisher spectra in neural networks.** Hayase and Karakida, [*The Spectrum of Fisher Information of Deep Networks Achieving Dynamical Isometry*](https://proceedings.mlr.press/v130/hayase21a.html) (AISTATS 2021), studies Fisher spectra in deep networks from a trainability perspective.
- **KV-cache compression.** Hooper et al., [*KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*](https://arxiv.org/abs/2401.18079) (2024), studies low-bit KV quantization, including pre-RoPE key handling and sensitivity-aware datatypes. This repository's KV experiments instead tested whether causal prediction added practical rate–distortion benefit beyond delta coding.
- **Transformer compression and distillation.** The block-surrogate experiments belong to the broad literature on knowledge distillation, structured pruning, and activation-aware low-rank approximation. Their purpose here is controlled falsification, not a priority claim for block replacement.

The potentially interesting observation is narrower: under one controlled Qwen experiment, pointwise downstream Fisher geometry was far more concentrated than matched aggregate geometry, yet train-only prototype dictionaries did not recur well on held-out contexts. Broader replication and a dedicated novelty review would be required before treating that as a general transformer result.

