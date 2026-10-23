# Examples

Scripts that show how to use CovSel. Each one simulates data, fits one or more
estimators, and prints what it found; plots are drawn in the terminal with
[UnicodePlots](https://github.com/JuliaPlots/UnicodePlots.jl).

| script | what it shows |
|---|---|
| [`graphical_lasso.jl`](graphical_lasso.jl) | The graphical lasso with `covsel!` and `covselpath`, and edge recovery (TPR, FPR, precision) along a path of penalties. |
| [`difference_estimation.jl`](difference_estimation.jl) | Direct estimation of `Ωx - Ωy` with `differencePrecisionActiveShooting`, refitting with `differencePrecisionRefit` (checked against the closed form), `differencePrecisionIHT`, and the indirect approaches. |
| [`fused_graphical_lasso.jl`](fused_graphical_lasso.jl) | `fusedGraphicalLasso` for two groups and the effect of the fusion penalty λ2; `fusedNeighborhoodSelection` for a single node. |
| [`latent_difference.jl`](latent_difference.jl) | A sparse plus low rank difference with `differenceLatentPrecisionIHT!`: choosing the step sizes, monitoring convergence with a callback, and the comparison with the sparse estimator. |
| [`clime.jl`](clime.jl) | CLIME and ACLIME with `clime` and `aclime`: a path of penalties, cross validation with `climecv` and `aclimecv`, edge recovery, and the solvers. |
| [`inference.jl`](inference.jl) | A simulation of the test for one entry of `Ωx - Ωy` when p > n (the Gaussian case of ROCKET): size, coverage, and the null distribution against N(0, 1). |

These replace the scripts `example1.jl`, `example2.jl` and `exampleInference.jl`
that were removed in [#3](https://github.com/mlakolar/CovSel.jl/pull/3). The
timing comparison from `example1.jl` is in [`../benchmark`](../benchmark).

## Setup

The examples have their own environment, [`Project.toml`](Project.toml), which
uses the CovSel in this repository. From the repository root, on Julia 1.11 or
later:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

Julia 1.10 ignores the `[sources]` entry of `Project.toml` that points to the
repository, so add it explicitly:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Until ProximalOPT 0.2 and CoordinateDescent 0.7 are registered, `[sources]`
also takes them from GitHub, and on Julia 1.10 they must be added in the same
call:
`Pkg.add([PackageSpec(url="https://github.com/mlakolar/ProximalOPT.jl", rev="aeed6002d0c791b086cc6da18c2f092f35af0439"), PackageSpec(url="https://github.com/mlakolar/CoordinateDescent.jl", rev="fd3f1da29b92982c1092d956d7b98f1f3a7b56b4")])`.

## Running

```sh
julia --project=examples examples/graphical_lasso.jl
```

or, in the REPL, `] activate examples` followed by
`include("examples/graphical_lasso.jl")`.

Each script takes a few seconds. With the environment variable
`COVSEL_EXAMPLES_QUICK=1`, `inference.jl` uses a smaller setting than its
default (p = 500, n = 400, 500 replications); CI runs all the examples this
way.
