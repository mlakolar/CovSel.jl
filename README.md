# CovarianceSelection.jl

[![CI](https://github.com/mlakolar/CovarianceSelection.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/CovarianceSelection.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/CovarianceSelection.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/CovarianceSelection.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Various methods for learning high-dimensional Gaussian graphical models.

```julia
using Pkg
# not registered yet, and neither are the versions of ProximalBase and
# ProximalSolvers that it needs: add the three from GitHub in one call
Pkg.add([PackageSpec(url = "https://github.com/mlakolar/ProximalBase.jl", rev = "master"),
         PackageSpec(url = "https://github.com/mlakolar/ProximalSolvers.jl", rev = "master"),
         PackageSpec(url = "https://github.com/mlakolar/CovarianceSelection.jl")])
```

CovarianceSelection was called CovSel before its first release.

* CLIME (Cai, Liu and Luo, 2011) and adaptive CLIME (Cai, Liu and Zhou, 2016),
  sparse precision matrices by constrained ℓ1 minimization: `clime`, `aclime`,
  with cross validation `climecv`, `aclimecv`. The default solver follows the
  exact solution path of each column by the parametric simplex method, so a
  whole path of penalties costs about as much as its smallest penalty; the
  columns are solved in parallel on Julia's threads.
* Sparse matrix inversion with the scaled lasso of each column (Sun and Zhang,
  2013), which needs no tuning: `slassoinv`
* Graphical lasso by ADMM: `covsel!`, `covselpath`
* Fused graphical lasso for K groups (Danaher, Wang and Witten, 2014), with all pairs fused: `fusedGraphicalLasso`, `fusedGraphicalLasso!`
* K precision matrices with a shared sparsity pattern, the group graphical lasso (Danaher, Wang and Witten, 2014): `groupGraphicalLasso`, `groupGraphicalLasso!`
* Graphs that vary over time (Zhou, Lafferty and Wasserman, 2010), the group graphical lasso of kernel-weighted covariances: `tvCovSel`, `kernelCovariances`
* Fused neighborhood selection: `fusedNeighborhoodSelection`
* Direct estimation of the difference of two precision matrices `Ωx - Ωy` (Zhao, Cai and Li, 2014)
  * ℓ1 penalized, by active shooting coordinate descent: `differencePrecisionActiveShooting`; the loss is `directDifferenceLoss`
  * the exact refit on a support: `differencePrecisionRefit`
  * the residual of its estimating equation, a validation loss for choosing λ: `diffLoss`
  * sparsity constrained, by iterative hard thresholding: `differencePrecisionIHT`
  * sparse plus low rank: `differenceLatentPrecisionIHT`

CovarianceSelection is one of five Julia packages for high-dimensional
statistics, three for estimation and inference and two for the optimization
beneath them:

| Package | |
|---|---|
| [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| **CovarianceSelection.jl** (this package) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Usage

```julia
using CovarianceSelection, ProximalBase, ProximalSolvers, LinearAlgebra, Statistics

p, n = 20, 1000
Ωy = Matrix(1.0I, p, p)
Ωy[1, 2] = Ωy[2, 1] = 0.4
X = randn(n, p)
Y = CovarianceSelection.generateData(Ωy, n)
Sx, Sy = cov(X), cov(Y)

# graphical lasso: minimize tr(SΩ) - log det Ω + λ‖Ω‖₁
Ω = covselpath(Sy, [0.2, 0.1])[end]

# direct difference estimation:
# minimize tr(Sx⋅Δ⋅Sy⋅Δ)/2 - tr(Δ⋅(Sy - Sx)) + λ‖Δ‖₁
Δ = differencePrecisionActiveShooting(Symmetric(Sx), Symmetric(Sy), NormL1(0.1))
Matrix(Δ)

# the same with ProximalSolvers' solver options, here a cyclic order, and along a
# path of 20 penalties from λmax
Δ = differencePrecisionActiveShooting(Symmetric(Sx), Symmetric(Sy), NormL1(0.1), CDSolver(; order=:cyclic))
f = directDifferenceLoss(Symmetric(Sx), Symmetric(Sy))
λs = lambda_max(f, NormL1(1.0), SymmetricSparseIterate(p)) .* 0.8 .^ (0:19)
path = solve_path(f, NormL1(1.0), λs; x=SymmetricSparseIterate(p), screening=false)
```

The losses and penalties are those of
[ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl): a penalty is
`NormL1(λ)`, or `NormL1(λ, W)` with a `p × p` matrix of weights `W`, and the
direct difference loss is its `KronQuadratic`. The graphical lasso runs on the
ADMM of [ProximalSolvers.jl](https://github.com/mlakolar/ProximalSolvers.jl), and the
direct difference estimator on its coordinate descent, whose options are
`CDSolver`'s.

### Fused graphical lasso for multiple groups

Pass a vector of covariance matrices or a `p × p × K` array. The fusion
penalty couples every pair of groups, as in
[Danaher, Wang and Witten (2014)](https://doi.org/10.1111/rssb.12033).

```julia
S = [S1, S2, S3]                               # sample covariances
ns = [200, 300, 500]                           # observations in each group
Z = fusedGraphicalLasso(S, ns, 0.02, 0.05; penalize_diag=false)
Ω1, Ω2, Ω3 = eachslice(Z; dims=3)

# Equal likelihood weights by default, or supply positive weights explicitly.
Z = fusedGraphicalLasso(S, 0.02, 0.05; weights=ns/sum(ns))

# Warm start: retain both the estimates and scaled dual variables.
Z, U = zeros(size(Z)), zeros(size(Z))
info = ADMMInfo()
fusedGraphicalLasso!(Z, U, S, ns, 0.02, 0.05; info)
fusedGraphicalLasso!(Z, U, S, ns, 0.02, 0.04;
                    options=ADMMOptions(; ρ=info.ρ), info)
```

The objective is `Σₖ wₖ (tr(SₖΩₖ) - logdet(Ωₖ)) + λ1 Σₖ ‖Ωₖ‖₁ +
λ2 Σₖ<ₗ ‖Ωₖ - Ωₗ‖₁`. Sample sizes set `wₖ = nₖ / sum(ns)`; to use
penalties from the paper's objective with unnormalized sample-size weights,
divide both by `sum(ns)`. `penalize_diag=false` omits the diagonal from the
ℓ1 term while retaining its fusion. The existing two-matrix call
`fusedGraphicalLasso(Sx, nx, Sy, ny, λ1, λ2)` still returns `(Zx, Zy)`.

The implementation uses ProximalBase's all-pairs `NormL1Fused` proximal
operator through `FiberSum`, together with `SliceSum(LogDetLoss(...))` and
ProximalSolvers' ADMM. See
[`examples/fused_graphical_lasso.jl`](examples/fused_graphical_lasso.jl) for
fits with two and three groups.
[`examples/danaher2014.jl`](examples/danaher2014.jl) recreates the paper's
three-group simulation design, with small demonstration and full 100-replicate
profiles, saved metrics and Figure 2-style plots; see its
[reproduction protocol](examples/danaher2014/README.md) for the exact scope.

### CLIME and ACLIME

Column `i` of the CLIME estimate minimizes `‖β‖₁` subject to `‖Sβ - eᵢ‖∞ ≤ λ`,
and the columns are symmetrized by keeping the smaller of each pair of
entries.

```julia
using CovarianceSelection, Statistics

# X: n × p data; the penalty of CLIME is in the units of SΩ, so standardize
Z = (X .- mean(X; dims=1)) ./ std(X; dims=1)
Θ = clime(cov(Z), 0.2)                       # a SparseMatrixCSC
λs = exp.(range(log(0.6), log(0.05); length=30))
path = clime(cov(Z), λs)                     # path.λ, path[k]
cv = climecv(Z, λs)                          # 5-fold cross validation, likelihood loss
Θ = bestestimate(cv)
Ω = Θ ./ (std(X; dims=1)' * std(X; dims=1))  # back to the scale of X

# ACLIME adapts the constraints to the variances; δ = 2 is the theory's value
Ω = aclime(cov(X), size(X, 1))
cv = aclimecv(X, (1:100) ./ 50)              # δ by cross validation
```

The solver is the optional third argument: `ParametricSimplex()` (the
default), `DualSimplex()`, `LPSolver(optimizer)` for any
[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) solver
(for example `using HiGHS; LPSolver(HiGHS.Optimizer)`), or the first order
method `PDHG()`. Start Julia with threads (`julia -t auto`) to solve the
columns in parallel. See [`examples/clime.jl`](examples/clime.jl), and
[`benchmark/clime`](benchmark/clime) for a comparison of the solvers with each
other and with the R packages fastclime, flare and clime.

### Sparse matrix inversion with the scaled lasso

Column `j` of the estimate of Sun and Zhang (2013) is the scaled lasso of
variable `j` on the others, which estimates the residual standard deviation
`σ̂ⱼ` together with the coefficients and scales the penalty of the column with
it: `Ωⱼⱼ = 1/σ̂ⱼ²` and `Ωₖⱼ = -β̂ₖ/σ̂ⱼ²`. The penalty level `λ₀` is a universal
constant, so there is nothing to tune, and the estimate is equivariant to the
units of the variables.

```julia
Ω = slassoinv(cov(X), size(X, 1))               # λ₀ of Sun and Zhang's simulations
Ω = slassoinv(cov(X), size(X, 1); refit=true)   # least squares on the selected support
```

Each column follows its lasso path exactly until the penalty of the scaled
lasso; see [`benchmark/slassoinv`](benchmark/slassoinv) for its speed
against the R package scalreg and its accuracy against CLIME and ACLIME.

## Examples

The [`examples`](examples) folder has a script for each estimator: the
graphical lasso along a path of penalties, direct difference estimation and
refitting, the fused graphical lasso and fused neighborhood selection, graphs
that vary over time, a sparse plus low rank difference, and a simulation of
inference for one entry of `Ωx - Ωy` when p > n. They have their own
environment; from the repository root:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=examples examples/graphical_lasso.jl
```

See [`examples/README.md`](examples/README.md) for what each script shows and
for Julia 1.10.

## Benchmarks

The [`benchmark`](benchmark) folder has a BenchmarkTools suite comparing the
solvers of the direct difference estimator and timing the ADMM solvers:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=benchmark benchmark/run.jl
```

See [`benchmark/README.md`](benchmark/README.md) for details.

### IHT with automatic step sizes

Omit the step-size arguments to use ProximalSolvers' backtracking methods:

```julia
using LinearAlgebra, ProximalSolvers
Δ, objectives = differencePrecisionIHT(Symmetric(Σx), Symmetric(Σy), s)
Δ, L, U, objectives, iterations = differenceLatentPrecisionIHT(
    Symmetric(Σx), Symmetric(Σy), s, r;
    method = MonotoneAccProxGradDescent())
S, L, U, objectives, iterations = covselLatentIHT!(Σ, s, r;
    method = ProximalGradientDescent(; normalized = true))
```

The latent models use `S + U*U'` (or `Δ + U*U'`) and one step size for both
blocks. The log-det loss checks a Cholesky factorization before taking an
inverse, and the line search rejects points outside the positive definite
cone. `init=(S,U)` supplies a copied feasible starting point. The latent
D-trace in-place overload updates existing `Δ,L,U` buffers.

`options=ProximalOptions(...)` controls stopping and callbacks. The returned
objective vector contains the initial value and each accepted iteration.
These are nonconvex local methods; different initializations can lead to
different supports and objectives. Calls with explicit step-size arguments
retain their original fixed-step behavior. Automatic steps use the exact
Frobenius projection from ProximalBase 0.5.3, which can differ from the legacy
magnitude threshold when diagonal entries count toward the sparsity budget.
