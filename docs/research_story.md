# Research story

This project asked a sequence of progressively narrower questions about whether transformer inference exposes cheap, reusable structure. Each stage used a predeclared gate; weak directions were stopped instead of enlarged.

## 1. Can KV memory be predicted rather than stored?

The first hypothesis was that new key/value states might be cheaper to predict from causal history than to code independently. Tests included independent coding, previous-token delta, reference residuals, train-only AR(1)/AR(4), reconstructed-history feedback, actual entropy-coded streams, and next-token KL.

Delta coding materially beat independent coding, but learned AR prediction did not materially beat delta once realistic predictor and codec overhead were included. Adjacent-layer token×depth innovation, conditional PCA/KLT, and a validation-selected heterogeneous codec also failed their predefined thresholds.

**Lesson:** high adjacent similarity does not imply a low-entropy innovation or a practical predictive coding advantage.

## 2. Can transformer blocks be compiled into cheap dynamics?

The second hypothesis was that transformer computation might become approximately linear in learned coordinates. At matched analytic MAC budgets, the experiment compared identity, dense linear, residual low-rank, residual MLP, latent-linear, and latent-nonlinear surrogates. Functional quality was measured by actually replacing the frozen block and computing next-token KL and delta NLL.

Latent linearity gave no meaningful advantage over the compute-matched residual MLP or nonlinear latent control. Multi-block replacement remained substantially damaging.

**Lesson:** a sophisticated coordinate system is not evidence of simple dynamics unless it beats equally cheap nonlinear distillation.

## 3. Does downstream-visible functional dimension save us?

Hidden-state MSE may waste capacity on directions the downstream model barely distinguishes. The next experiment pulled the categorical output Fisher back to the residual stream using score VJPs.

The aggregate 896-dimensional Fisher required 432 dimensions for 95% of estimated energy. Its quadratic form predicted actual finite replacement KL only marginally better than MSE, with negative held-out calibration R² for both.

**Lesson:** a theoretically motivated local metric is not automatically a useful finite-error compression objective.

## 4. Can the teacher reorganize itself?

A small LoRA adaptation was applied only to the target block while preserving the original model distribution. The adapted expensive block was then discarded and replaced by the exact same cheap MLP class used in the fixed-teacher baseline.

Teacher drift stayed extremely small (0.000965 test KL), but deployed replacement KL changed from 0.084611 to 0.085315—slightly worse.

**Lesson:** behavior-preserving parameter movement was possible, but surrogateability did not emerge under this bounded retrofit.

## 5. Local functional geometry

The final hypothesis separated pointwise from aggregate geometry. At each context/token, the Fisher considered a fixed H=32 causal horizon. Pointwise spectra were strongly concentrated: validation median estimated r95 was 27, while a matched aggregate estimate required approximately 107 dimensions.

The stronger question was whether these subspaces recur. Train-only Grassmann clustering selected q=128 and K=8 on validation. On untouched test tokens, the dictionary captured 0.262 median functional energy versus 0.254 for one global q-dimensional subspace. Prototype use and independent-half stability were weak.

**Ending:** transformer functional geometry appeared locally concentrated but strongly context-dependent. The pointwise observation survived; the reusable-regime and compression stories did not.

