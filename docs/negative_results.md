# Negative results

| Hypothesis | Strongest baseline | Result | Decision |
|---|---|---|---|
| Temporal AR beyond delta | Previous-token delta | Learned incremental gain disappeared under realistic overhead | Killed |
| Token×depth innovation | Temporal delta and direct-depth residual | Dense oracle showed no coding headroom | Killed |
| Conditional PCA/KLT | Same transform on raw and conditioned signals | Unique conditional benefit was negligible | Killed |
| Heterogeneous fixed codec | Strongest global temporal codec | About 0.5% gain | Killed |
| Learned latent-linear dynamics | Compute-matched residual MLP/nonlinear latent control | No meaningful Pareto advantage | Killed |
| One global functional subspace | Aggregate downstream Fisher | r95=432/896; poor finite-KL prediction | Killed |
| Functional prehabilitation | Fixed-teacher width-658 MLP | 0.085315 vs 0.084611 test KL | Killed |
| Small dictionary of local subspaces | Matched global q-space | +0.011 median test capture | Killed |

## Learned temporal KV prediction

**Why plausible:** adjacent K/V states can have high cosine similarity.

**What killed it:** previous-token delta captured most rate–distortion benefit; learned-predictor and codec metadata erased the remaining small AR advantage.

**What we learned:** geometric similarity is not the same as predictive coding advantage.

## Token×depth innovation

**Why plausible:** the same token innovation propagates through adjacent transformer layers.

**What killed it:** temporal delta and direct depth were stronger controls, and even a dense cross-layer oracle showed no conditional coding headroom.

**What we learned:** adjacent-depth redundancy did not reveal a distinct token×depth innovation code.

## Conditional transform coding

**Why plausible:** innovations can be high-rank in native coordinates yet compressible after a learned transform.

**What killed it:** train-fitted PCA/KLT helped raw and conditioned signals similarly; conditioned innovations gained no unique frontier advantage after transform metadata.

**What we learned:** transform benefit must be compared against an identically transformed unconditioned baseline.

## Heterogeneous fixed codec

**Why plausible:** layers, heads, K, and V have visibly different statistics.

**What killed it:** validation-selected modes improved only about 0.5% over the strongest single global method.

**What we learned:** heterogeneous statistics do not necessarily imply useful heterogeneous decisions.

## Learned-coordinate block dynamics

**Why plausible:** nonlinear transformer computation might linearize in learned coordinates.

**What killed it:** compute-matched small MLP and nonlinear-latent controls matched or beat the latent-linear model; multi-block damage stayed high.

**What we learned:** nonlinear encoder/decoder capacity must not be mistaken for evidence of latent linearity.

## Global functional geometry

**Why plausible:** Euclidean hidden error may spend capacity on downstream-invisible directions.

**What killed it:** aggregate Fisher r95 remained 432/896, and its quadratic error predicted finite intervention KL only marginally better than MSE.

**What we learned:** local information geometry may be poorly calibrated for finite surrogate errors.

## Functional prehabilitation

**Why plausible:** a teacher could move within a behavior-preserving neighborhood toward an easier internal realization.

**What killed it:** teacher drift stayed tiny, but the identical deployed surrogate became slightly worse on untouched test data.

**What we learned:** preserving output behavior does not guarantee that block-level surrogateability can be reorganized cheaply.

## Reusable local regimes

**Why plausible:** low pointwise Fisher ranks could reflect switching among a small number of recurring functional subspaces.

**What killed it:** the best validation-selected dictionary captured only 0.262 median energy on test versus 0.254 for one matched global subspace; prototype use and stability were weak.

**What we learned:** pointwise concentration can coexist with idiosyncratic context-dependent orientation.

