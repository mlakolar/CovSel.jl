# Estimation, Studentization, and bands

## Statistical target and first stage

Write `t = z' gamma`, with a unit direction and a positive first coordinate.
The target is the inverse of the actual conditional covariance, not a latent
copula correlation. Nonzero precision entries describe conditional partial
correlations. Conditional independence requires stronger distributional
conditions. We do not substitute a Kendall correlation for covariance, because
that would change this target under general conditional distributions.

For a fixed probe pair, the score moment is the expectation of the product of
the two responses times the known covariate score. Stein's identity makes it a
scalar multiple of the index direction. The implementation takes equal
consecutive blocks, uses their coordinatewise median, soft-thresholds, and
normalizes. It rejects a zero direction or zero orientation anchor. Threshold
calibration and probe identification remain explicit user choices.

The current practical threshold uses an empirical coordinate variance. The
median-of-means moment is robust under finite second moments, but this scale
estimate is not itself a robust high-probability calibration. A supplied scalar
or vector threshold is supported. Block ordering is deterministic; randomize
observations beforehand when ordering is not exchangeable.

## Default common-complement estimator

For an off-diagonal pair `J={a,b}`, regress both responses on every coordinate
outside `J`, with the same complement for both regressions. For a diagonal,
use `J={a}` and regress on every other coordinate. The population residual
covariance is the inverse of the target principal precision block. Thus the
diagonal construction needs a scalar inverse; the off-diagonal construction
needs a 2 by 2 inverse. Diagonals obtained from different paired fits are not
pooled. Each requested diagonal uses its singleton regression.

The nuisance fit uses normalized nonnegative kernel weights at the pilot
bandwidth. Its Gram matrix is positive semidefinite. Predictor loadings are
square roots of its diagonal. The weighted square-root lasso minimizes

$$
\left\{\sum_i w_i(X_{ic}-X_{iA}^{\mathsf T}\beta)^2\right\}^{1/2}
+\lambda\sum_{j\in A}\sqrt{S_{jj}}|\beta_j|.
$$

The scaled-lasso formulation with a free residual standard deviation has the
same beta minimizer. We reuse the parent's exact single-column homotopy solver.
The penalty is a supplied multiplier times the square root of twice the log
predictor count divided by the realized Kish sample size. Zero-loading
predictors are removed. A singular active set is a failure, not a ridge refit.

At each evaluation point, hold the nuisance coefficient fixed for every
observation. Form its residual products, fit a degree-two local polynomial,
and invert its intercept matrix. This is the manuscript's same-bandwidth
local-linear robust-bias-correction construction. It is not the estimator
obtained by recomputing residuals at each observation's index. It is also not
obtained by inverting a local covariance matrix of all responses.

The fitted local polynomial has design rows `(1,u,u^2)`. If `C` is its
coefficient map, the intercept weights are the first row of `C`. They sum to
one and annihilate linear and quadratic monomials. Some weights can be negative.
We reject designs with condition number at least 1e12, fewer than degree+5
positive-weight observations, or a nonpositive final residual covariance.
`degree=0`, `1`, and `3` are supported for explicit smoothing sensitivities;
the manuscript's primary procedure uses degree 2.

## Full inverse pilots and one-step correction

`precision_pilot` localizes the package's scaled-lasso inverse or ACLIME. Both
use nonnegative kernel covariance estimates and a realized effective sample
size. ACLIME uses its floor for the diagonal ridge and two-stage constraints.
The effective-size substitution is a practical localized modification. Its
coverage cannot be deduced directly from the iid ACLIME theorem. ACLIME's
truncation is disabled; a zero or nonpositive target block is reported.

A full pilot can supply common-complement coefficients through the block
identity `B = -P[A,J] / P[J,J]`. Alternatively, for symmetric final local
covariance `S`, the one-step matrix is

$$ P+P^{\mathsf T}-P^{\mathsf T}SP. $$

The pilot is symmetrized by the parent's minimum-absolute-entry convention.
We keep the general expression above in the code. The one-step estimate need
not be positive definite, which is acceptable for entrywise inference and must
be remembered when reporting full-matrix errors. The block and one-step
constructions need not agree in finite samples.

[ROCKET](https://arxiv.org/abs/1502.07641) motivates common-complement
residualization and targeted inference with imperfect nuisance recovery.
[Gu, Cao, Ning and Liu](https://arxiv.org/abs/1502.02347) motivates
score decorrelation. Their nonparanormal target and rank-statistic variance
are different from the conditional-covariance target and observation-score
variance here. We compare related orthogonal corrections without claiming to
reproduce either complete rank-based estimator.

## HC0 variance, including diagonals

For a fitted residual covariance `H` and its inverse `Q`, the differential of
its inverse is `-Q (dH) Q`. The diagonal derivative uses the same expression;
there is no extra manually inserted factor of two. For Gaussian data, the
variance of a squared transformed residual supplies the factor of two itself.
The HC0 score centers each residual product at the full fitted polynomial
evaluated at that observation's own index. This removes local mean drift from
the squared score. The variance is the sum of squared realized weighted scores.
The one-step variant applies the same polynomial centering to the product of
the responses transformed by the two pilot columns.

The influence contributions sum to zero by polynomial normal equations. This
is an algebraic fact; it is not proof of conditional centering for nuisance
fits made on the same sample. All variance calculations keep the final
polynomial weights, including variation from bias estimation. The simulations
report empirical standard deviation versus root-mean estimated variance.

## Multiplier maximum and the continuum

Each bootstrap draw generates one independent normal multiplier per
observation. The same vector multiplies every entry's influence contributions
at every evaluation point. Studentize each coordinate with its own HC0 scale,
then maximize its absolute value over the grid and, for `joint=true`, over all
requested entries. The critical value uses a conservative empirical inverse
CDF with a pointwise-normal floor. `joint=false` gives separate curvewise
bands; their simultaneous family coverage is not 95% in general.

This algorithm controls a finite-grid maximum when its fitted score and
Gaussian approximation conditions hold. Smoothing bias, generated-index error,
and nuisance error remain. A dense grid is not by itself a proof of continuum
coverage. A continuum claim additionally needs uniform regularity and a
quantified discretization bound. The manuscript's oracle and feasible results
must be distinguished. No general feasible diagonal or continuum theorem is
claimed by this implementation. Grid refinement is an empirical diagnostic.

## Selecting tuning parameters

`select_tuning` never uses truth or observed simulation coverage. Within each
fold it estimates common-complement nuisances on training observations, scores
weighted prediction on held-out observations, and selects a pilot bandwidth
and penalty. Candidate bandwidths are scored on a common, fixed validation window, chosen
as the smallest bandwidth in the respective candidate grid. This prevents
changing the validation population when changing a candidate. It then selects
the final bandwidth by held-out prediction of
residual products using those training nuisances. It optionally multiplies
this final bandwidth by a prespecified undersmoothing factor.

The supplied index is held fixed in this cross-validation. A caller-supplied
index estimated on all observations is not re-estimated within folds.

These losses target prediction, not coverage error. The optional factor
`n^(-1/20)` is a prespecified sensitivity rule. Its exponent is not a theorem
or universal recommendation. A simulation-based rule must be frozen after
calibration and assessed on the separate confirmation seed namespace.
Good coverage must be assessed with interval length and failure rates; inflating
intervals without limit is not useful tuning. Same-sample selection remains
part of the procedure whose repeated-sample coverage is tested.

## Why a sparse inverse can be a poor direct debiasing pilot

At a fixed point, let the true covariance be `S` and its inverse be `O`. For
any pilot `P`, the population one-step error has the exact identity

$$ P+P^{\mathsf T}-P^{\mathsf T}SP-O
=-(P-O)^{\mathsf T}S(P-O). $$

Thus the population diagonal error is nonpositive and quadratic in the pilot
column error. A small inverse-equation residual alone does not establish that
this term is small compared with the final standard error. An aggressively
shrunk inverse can also shrink the fitted influence scores and standard errors.
Both effects matter for coverage.

For example, if `P=c*O` for a scalar `c`, the direct correction is
`c*(2-c)*O`. In contrast, the common-complement coefficient
`-P[A,J]/P[J,J]` equals its population target exactly. This example explains
why block inversion can be more tolerant of diagonal-scale shrinkage; it does
not establish a universal comparison for arbitrary pilots.

`refit=true` removes coefficient shrinkage on the selected support. For
square-root lasso it uses the parent's exact least-squares support refit. For
a scaled-lasso inverse it uses the parent's refitted inverse. For ACLIME it
refits each selected neighborhood, estimates its residual variance, and then
uses the same minimum-absolute-entry symmetrization convention. Refits require
nonsingular selected Gram matrices and positive residual variances. They can
increase variance or overfit; coverage is assessed separately. The refinement
experiment compares each refit with the matching unrefitted method.
