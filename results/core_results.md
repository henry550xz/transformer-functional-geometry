# Core validated results

| Experiment | Primary result | Decision | Caveat |
|---|---:|---|---|
| KV temporal residual rank | delta/state effective-rank ratio 1.811 | Discouraging | Structural, not a compression claim |
| Temporal AR coding | 1.157× AR(1) vs delta before overhead | Weak signal | Below 1.20 threshold; overhead removed gain |
| One-block compilation | MLP 0.08398; latent-linear 0.08571 KL | Killed | Compute-matched at ~7.68% analytic block MACs |
| Aggregate downstream Fisher | r95=432/896 | Killed | Weak finite-KL prediction |
| Functional prehabilitation | 0.085315 vs 0.084611 fixed KL | Killed | Teacher drift was only 0.000965 |
| Local H=32 Fisher | median sketch r95=27 | Supported structurally | 384-probe rank-capped estimator |
| Local regime dictionary | 0.262 vs 0.254 median test capture | Killed | Validation-selected q=128, K=8 |

Values are sanitized aggregates transcribed from validated machine-readable scorecards. Raw model activations, corpora, and Fisher tensors are intentionally excluded.

