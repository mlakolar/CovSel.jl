# Finite-sample diagonal corrections and inference tuning

These options address residual fitting, Studentization, and inversion separately.
They are experimental finite-sample modifications. Their formulas do not establish
coverage after support selection, bandwidth selection, or index estimation.
Defaults preserve the original estimator so comparisons remain reproducible.

## Residual degrees of freedom

Set `residual_adjustment=:df` for block inversion. At each target point, take the
union of the nonzero nuisance supports for the one or two target responses.
Refit both responses on that common support by weighted least squares, with the
pilot kernel weights. This refit is performed even if `refit=false`: the linear
residual-maker calculation requires a least-squares fit on a fixed support.
No intercept is added; conditional means are assumed to be zero.

Write the selected predictor matrix as `U`, its normalized pilot weight matrix
as `W`, and its hat matrix as

$$ H=U(U^{\mathsf T}WU)^{-1}U^{\mathsf T}W. $$

For fixed predictors and a fixed selected support, constant conditional residual
covariance gives the exact identity

$$ E(R_iR_i^{\mathsf T}\mid U)=g_i V,\qquad
   g_i=1-2H_{ii}+\sum_j H_{ij}^2. $$

This identity also requires that the conditional regression mean lies in the
selected span. It does not condition away data-dependent selection. Kernel WLS
is generally not an orthogonal projection in the unweighted inner product, so
`1-H_ii` alone is not the correct row variance factor. A replacement based only
on the number of selected variables and a Kish sample size would also ignore
the mismatch between the pilot and final weights.

The adjusted residual is `R_i/sqrt(g_i)`. Smooth its outer products with the
original signed final polynomial weights, then invert the resulting block.
The implementation computes `g_i` using small Gram matrices. It never stores
an observation-by-observation hat matrix. Empty support gives `g_i=1` exactly.
A singular selected Gram matrix or a degenerate factor is a recorded failure.

For varying residual covariance, the expectation is a weighted mixture of the
observation-specific covariance matrices. Dividing by `g_i` is then a local
constant approximation. For a misspecified or selected support, the residual
mean and selection effects remain. The simulation suite tests these limitations.

## Final smoothing leverage

`studentization=:hc3` divides each original influence contribution by
`1-l_i`, where `l_i` is the diagonal of the final local polynomial hat matrix.
The same adjusted scores define the sandwich variance and the multiplier
process. The pilot residual factor `g_i` and the final smoothing leverage
`l_i` describe different fits. Neither substitutes for the other.

HC3 scores need not sum to zero. Independent mean-zero multipliers still have
conditional mean zero. Numerical unit leverage is rejected. HC0 remains the
default and retains the original polynomial normal-equation identity.

## Second-order inverse bias

`inverse_bias=:secondorder` subtracts the plug-in second-order term in matrix
inversion. Let `Q` be the inverse smoothed residual covariance, `a_i` the final
intercept weight, and `E_i` the adjusted residual outer product minus its fitted
polynomial value at observation `i`. The correction is

$$ \widehat B=\sum_i a_i^2 Q E_i Q E_i Q. $$

For HC3, replace `a_i` in this expression by `a_i/(1-l_i)`. Subtract the requested
entry of this matrix from the point estimate. Retain the original first-order
influence process. In the scalar diagonal case, the subtraction is exactly
`se^2/Q`. This addresses nonlinear inverse curvature. It does not estimate the
nuisance remainder or smoothing bias. The corrected block need not be positive
definite. No positivity clipping is applied.

## Tuning rules

`select_tuning` accepts `rule=:prediction`, `:onese`, or `:inference`.
All rules use identical validation weights across candidates. They use neither
the true covariance nor simulation coverage. The supplied index is held fixed;
it is not re-estimated within folds.

* `:prediction` preserves minimum prediction CV and the caller's undersmoothing
  factor. This is retained as the undercoverage baseline.
* `:onese` keeps pilot candidates within one standard error of the minimum CV
  loss. It selects the smallest average nuisance support. Ties prefer a larger
  pilot bandwidth, then a larger penalty multiplier. Final smoothing uses the
  largest bandwidth within one standard error of its minimum residual-product
  loss. Standard errors are the standard deviations of fold losses divided by
  the square root of the fold count. They are selection heuristics, not
  confidence intervals for prediction risk.
* `:inference` applies the one-SE rule after capping final candidates at
  `bandwidth_cap` (default `2.5*n^(-1/5)`). It excludes pilot candidates below
  `pilot_ratio*maximum(final_candidates)` (default ratio 1.5). The cap limits oversmoothing; the floor prevents selecting an equally
  narrow pilot solely for prediction. It requires at least one eligible pilot candidate.
  The validation windows remain the minima of the originally supplied grids.

Use `refit=true` to score support-refitted regression candidates when the final
fit will use a support refit. The candidate support size is returned alongside
its loss and standard error. Every candidate failure receives infinite loss;
all-candidate failure throws. Return values include the rule, fold assignment,
cap, pilot ratio, pilot SE allowance, and full candidate scores.

The cap, ratio, candidate grids, fold count, and undersmoothing factor are
prespecified inputs. The default cap is on the supplied index scale; standardize
that scale or supply an appropriate cap. They require sensitivity checks. These rules make no
coverage-optimal bandwidth claim. Confirmation uses seeds independent of rule
development, reports failures as noncoverage, and records interval lengths.

## Log-scale diagonal endpoints

`confint` and `confband` accept `diagonal_transform=:log`. For a positive fitted
diagonal `q` and its original-scale standard error `s`, endpoints are

$$ q\exp(-c s/q),\qquad q\exp(c s/q). $$

Off-diagonal endpoints remain additive. The multiplier score for the logarithm
is the original diagonal score divided by `q`; its standard error is also
divided by `q`. Thus every standardized multiplier coordinate, and its joint
critical value, is unchanged. This is a first-order delta-method log interval.
It is not an exact chi-square interval or a bootstrap bias correction.

The log transformation addresses the asymmetry induced by inversion. It can
worsen coverage when positive nuisance bias remains. It must therefore be
assessed jointly with the nuisance fit and inverse-bias correction. A
nonpositive successful diagonal cannot be log transformed and throws. Failed
fits retain their original NaN endpoints. The rescoring script evaluates these
endpoints from saved fits without repeating fitting or multiplier draws.

`pilot_se=0` restricts pilot selection to the minimum refitted prediction loss;
positive values allow that many fold standard errors before support complexity
breaks ties. Final bandwidth selection still uses a one-SE window. The
minimum-loss variant avoids discarding a useful nuisance variable solely to
simplify a statistically indistinguishable prediction model. It remains a
heuristic selector and requires independent confirmation.

## Frozen CV candidate

The confirmation protocol uses this explicit combination. Candidate grids are
in units of the supplied index and must be chosen before looking at coverage.

```julia
selected = select_tuning(X, index; points=[-0.5,0,0.5], entries,
    bandwidths=[0.6,0.9,1.2], pilot_bandwidths=[0.6,0.9,1.2,1.5],
    multipliers=[0.4,0.7,1.0], rule=:inference, refit=true,
    pilot_se=0, pilot_ratio=1.5, rng=MersenneTwister(17))
fit = fit_precision(X, index; points=grid, entries,
    bandwidth=selected.bandwidth, pilot_bandwidth=selected.pilot_bandwidth,
    multiplier=selected.multiplier, refit=true, residual_adjustment=:df,
    studentization=:hc3, inverse_bias=:secondorder)
intervals = confint(fit; diagonal_transform=:log)
bands = confband(fit; diagonal_transform=:log, draws=1999,
    rng=MersenneTwister(18))
```

See [CONFIRMATION_PROTOCOL.md](CONFIRMATION_PROTOCOL.md) for the fixed rule and
seed separation, and [RESULTS.md](RESULTS.md) for its completed assessment.
