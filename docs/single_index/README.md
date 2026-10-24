# Single-index precision inference

`CovarianceSelection.SingleIndexGraph` implements selected-entry inference for a
conditionally mean-zero response whose covariance depends on a scalar index.
Rows are observations throughout. The module can use an observed index, a
caller-supplied estimate, or its sparse known-score Stein estimator.

This is research software. The manuscript's primary estimator is implemented
literally. Diagonal intervals, localized inverse pilots, matrix one-step
corrections, tuning rules, and feasible multiplier bands are separately identified
extensions. Simulation success is evidence about tested designs, not a general
coverage theorem. See [METHODS.md](METHODS.md), [SIMULATIONS.md](SIMULATIONS.md), and
[RESULTS.md](RESULTS.md). Finite-sample corrections and the new CV rules are
described in [DIAGONAL_TUNING.md](DIAGONAL_TUNING.md). First-stage selection is
described in [INDEX_TUNING.md](INDEX_TUNING.md).

```julia
using CovarianceSelection
using CovarianceSelection.SingleIndexGraph
using Random

# X: n by p centered responses; index: length n, on a fixed oriented scale.
fit = fit_precision(X, index;
    points=collect(range(-0.75, 0.75; length=41)),
    entries=[(1, 1), (1, 2), (1, 8)],
    bandwidth=0.65, pilot_bandwidth=0.9,
    pilot=:sqrtlasso, multiplier=0.7)
pointwise = confint(fit; level=0.95)
family = confband(fit; level=0.95, draws=1999,
                  joint=true, rng=MersenneTwister(20261024))
# Curvewise bands, each at 95%; these are not jointly 95% across entries:
curvewise = confband(fit; joint=false, rng=MersenneTwister(20261024))
```

The two bandwidths are on the scale of `index`. The example values illustrate
syntax; they are not universally recommended constants. Prespecify the entries
before looking at estimated edges. A post-selection interpretation requires
additional analysis. For a known-score first stage:

```julia
idx = estimate_index(X, score; probe=(p-1,p), blocks=5, multiplier=0.5)
index = Z * idx.direction
```

`score` is the negative log-density gradient of the covariates. It equals `Z`
only for independent standard Gaussian coordinates. Supply the correct known
score for other distributions. A selected probe can have zero derivative signal;
normalization cannot repair that identification failure. Inspect the moment,
threshold, sparsity, discarded observations, and index sensitivity.

## API

| Function | Result |
|---|---|
| `estimate_index` | Median-of-means Stein moment, soft threshold, oriented unit direction |
| `select_index_tuning` | Held-out Stein alignment, threshold/support and block-count selection |
| `local_covariance` | Nonnegative local second moments, weights, realized effective sample size |
| `local_design` | Polynomial coefficient map, intercept weights, conditioning and local size |
| `precision_pilot` | Local scaled-lasso/ACLIME inverse and inverse-equation residual |
| `fit_precision` | Selected entries, standard errors, influence contributions and failures |
| `confint` | Pointwise normal intervals |
| `confband` | Gaussian observation-multiplier bands on the requested grid |
| `select_tuning` | Held-out nuisance and residual-product losses, chosen bandwidths and penalty |
| `precision_errors` | Entrywise, Frobenius, spectral and induced-norm errors |

`fit_precision` accepts `pilot=:sqrtlasso`, `:scaledlasso`, or `:aclime`.
The default `correction=:block` residualizes on the common complement.
`refit=true` optionally removes shrinkage by least squares on selected supports.
`correction=:onestep` is available for full inverse pilots.
For simulation benchmarks, `oracle_coefficients=(point,J,A)->B` replaces the
nuisance fit. It does not replace the index supplied to the function.

The core has no new package dependencies. A callable kernel from
[KernelSmoothing.jl](https://github.com/mlakolar/KernelSmoothing.jl) can be passed
as `kernel`, for example `KernelSmoothing.BiweightKernel()`. Like that package,
we keep the complete local-polynomial weights and observation-level variance
vectors. CovarianceSelection's exact scaled-lasso homotopy and ACLIME solvers
supply the sparse fits. CoordinateDescent, ProximalBase, and ProximalSolvers
provided API and optimization conventions. KernelSmoothing's univariate
coverage-optimal selector is not asserted to solve this generated-index,
high-dimensional inference problem.

## Diagnostics and failures

Inspect `fit.failure` and `fit.diagnostics`. Each estimate retains its failure
message. A failed point has NaN center, standard error, and influence column.
`confint` preserves NaNs. `confband` requires every requested fit to succeed.
No failed entries are removed from its maximum. No indefinite matrix is silently
projected onto the positive-definite cone. The requested grid is not silently
trimmed. These rules prevent artificially favorable coverage.

The response must have conditional mean zero. Local covariance uses second
moments without estimated-mean subtraction. Nonzero conditional means require
an additional mean-regression stage and its error analysis. Observations must
be independent; the bootstrap supplied here does not handle serial dependence.
