# Methodology

## Experimental discipline

All learned predictors, transforms, surrogates, and cluster prototypes were fit on sequence-disjoint training data. Validation selected regularization, architecture budget, or q/K settings. Test data was reporting-only. Causal predictors consumed reconstructed history, and functional claims came from actual model interventions rather than hidden-state MSE alone.

The primary model was Qwen2.5-0.5B-Instruct. Layer numbers are zero-based. For block replacement, `hidden_states[10]` is the residual stream entering block 10 and `hidden_states[11]` is its output/entry to block 11.

## Predictive KV evaluation

K and V were extracted separately at representative layers. Comparisons used the same quantizer family and calibration protocol for independent, previous-state delta, reference, AR(1), and AR(4) residuals. Sequential reconstruction fed quantized historical states back into causal predictors. Reported rate included stream estimates, real deterministic Huffman byte streams, anchors, and predictor metadata.

## Compute-matched block surrogates

The one-block tournament compared dense and low-rank linear maps, a residual MLP, and latent linear/nonlinear models. Analytic MAC accounting used one convention for all candidates and included attention projections, sequence-length attention products, and the gated MLP in the original block. Hooks replaced the selected block output inside the frozen model before KL/NLL evaluation.

## Downstream Fisher geometry

For activation h and downstream logits G(h), the local pullback metric is

```text
M_h = J_G(h)^T [diag(p) - p p^T] J_G(h).
```

The local-regime experiment fixed a causal horizon H=32 and averaged categorical Fisher contributions from the next 32 prediction positions. It never materialized a vocabulary-by-896 Jacobian. Independent categorical score samples produced batched VJPs with respect to one token activation; their covariance estimates the Fisher.

Each reported local estimate used 384 probes, creating an empirical covariance rank cap of 383. FP32 eigendecomposition retained 256 leading vectors, and full sample-gradient norms estimated the trace. The estimator therefore supports the q∈{32,64,128} subspace comparisons, but exact unobserved-tail ranks are not claimed.

Numerical checks covered symmetry, positive semidefiniteness, and exact zero intervention. Finite-radius quadratic calibration in BF16 was poor and is reported as a limitation.

## Recurring-subspace test

Subspace similarity was the Grassmann projection overlap

```text
||U_i^T U_j||_F^2 / q.
```

Deterministic k-medoids used distance `1 - overlap`. For each train cluster, the prototype was the top-q eigenspace of the mean projection matrix. Validation selected q∈{32,64,128} and K∈{1,2,4,8,16}. The primary held-out metric was the fraction of each token Fisher's trace captured by its oracle-best prototype, compared at the same q with the top subspace of the aggregate train Fisher.

