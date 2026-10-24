# Frozen diagonal/CV confirmation protocol

Frozen after the 431000000 and 463000000 development seed namespaces, before
any result in the 587000000 or 617000000 confirmation namespaces.

The primary revised CV method is `sqrt_refit_df_hc3_bc_log` in every scenario.
Use three folds, common validation windows, and refitted nuisance prediction
loss. Select the minimum-loss pilot (`pilot_se=0`). Cap final candidates at
`2.5*n^(-1/5)` and require pilot candidates at least 1.5 times the largest
capped final candidate. Select the largest final candidate in a one-SE loss
window. No extra undersmoothing multiplier is used.

The original final candidate grid is `[0.6,0.9,1.2]`; the original pilot grid
is `[0.6,0.9,1.2,1.5]`; penalty multipliers are `[0.4,0.7,1.0]`. The index is
on the standard Gaussian or specified uniform scale. Tuning evaluates
`[-0.5,0,0.5]` and the three prespecified entries. Final inference uses the
21-point grid and entries `(1,1)`, `(1,2)`, and `(1,p-2)`.

Refit the union nuisance support. Normalize residuals by residual-maker row
variance. Use HC3 scores and subtract the second-order inverse bias. Construct
log-scale diagonal endpoints and additive off-diagonal endpoints. Both use
the same 1999-draw observation-multiplier critical value. This is one fixed
rule; do not choose a different variant after seeing each scenario's coverage.

`diagonal_confirmation.toml` uses 1000 replications per scenario. It compares
original prediction CV with the revised rule on matched datasets, including
known and generated indices. Fixed wider-pilot scenarios retain the previous
scaled-lasso/ACLIME comparators and test the finite-sample corrections.

`diagonal_stress.toml` uses 1000 replications for varying nuisances, chain and
grid graphs, t8 errors, scale mixtures, stronger curvature, and support
boundaries. The `p=320` and weak-index stress cases use 500 replications.
The additive corrected interval is a prespecified secondary comparator.
Oracle nuisance fits diagnose smoothing and distributional effects; they do
not supply any quantity used by the feasible selector.

Report pointwise diagonal and edge coverage, curvewise and joint grid coverage,
Wilson Monte Carlo intervals, matched-seed differences, interval widths,
bias, empirical standard deviation versus estimated scale, selected tuning,
and all numerical failures. Retain adverse findings. Finite-grid results do
not establish continuum coverage. No general coverage-optimal tuning theorem
is asserted. Further changes require a new development and confirmation study.
