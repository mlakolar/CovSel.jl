# Benchmarks

[`benchmarks.jl`](benchmarks.jl) defines a
[BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) suite, `SUITE`,
in the format used by [PkgBenchmark](https://github.com/JuliaCI/PkgBenchmark.jl).
[`run.jl`](run.jl) runs it and prints the median time, memory and number of
allocations of each benchmark.

## What is measured

- **direct difference**: the ℓ1 penalized direct difference estimator on the
  problem of the removed `example1.jl` (Σx = I, Σy the AR(1) covariance with
  ρ = 0.7, n = 1000), for p = 20, 50, 100 and λ = 0.1, 0.2, solved by
  - `differencePrecisionActiveShooting`: the package solver, ProximalSolvers'
    coordinate descent, at λ directly from zero, and along a path of 50
    decreasing λ values from λmax (`solve!` with `continuation = 50`),
  - `Reference.differencePrecisionNaive` from
    [`../test/reference.jl`](../test/reference.jl): full coordinate descent
    sweeps over dense matrices, the reference the tests check the package
    solver against;

  and, with the size of the true support, `differencePrecisionIHT` and the
  sparse plus rank one `differenceLatentPrecisionIHT`. Before the timings,
  `run.jl` checks that the package solver agrees with the reference solver.
- **ADMM**: `covsel!` and `fusedGraphicalLasso` on the same covariances.
- **CLIME**: `clime` along a path of 20 penalties and at one penalty, and
  `aclime`, on the correlation and covariance matrices of the AR(1) sample.
  The comparison of the CLIME solvers with each other and with the R packages
  is in [`clime/`](clime).
- **generateData**: 1000 draws from a tridiagonal precision matrix, from `Ω`
  and from its precomputed Cholesky factor.

## Setup and running

The benchmarks have their own environment, [`Project.toml`](Project.toml),
which uses the CovSel in this repository. From the repository root, on Julia
1.11 or later:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl
```

On Julia 1.10, which ignores the `[sources]` entry of `Project.toml`, set up
the environment with
`julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`.

The full suite takes a few minutes. With `COVSEL_BENCHMARK_QUICK=1` it runs
p = 10 once per benchmark, which is what CI does to keep the suite working.

## Comparing two versions

Pass a file name to save the results, then compare them with `judge`:

```sh
julia --project=benchmark benchmark/run.jl before.json
# ... change the code ...
julia --project=benchmark benchmark/run.jl after.json
julia --project=benchmark -e '
  using BenchmarkTools
  before, after = BenchmarkTools.load("before.json")[1], BenchmarkTools.load("after.json")[1]
  display(judge(median(after), median(before)))'
```

## Against the code before ProximalSolvers (`compare/`)

[`compare/`](compare) times the solvers of CovSel master before the move onto
ProximalSolvers (its own coordinate descent loss and ADMM loop, on
CoordinateDescent 0.6.0 and ProximalBase 0.4.1; `compare/old/`) against this
version (`compare/new/`), on the same problems and seeds. `run.jl` runs one side
in a fresh process; `ab.sh` alternates the sides over a number of rounds, since
load on a shared machine changes over minutes, and `compare.jl` keeps the
fastest median of each case and compares the passes and the solutions:

```sh
julia --project=benchmark/compare/old -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/compare/new -e 'using Pkg; Pkg.instantiate()'
benchmark/compare/ab.sh 4
```

Results of 23 October 2026: four rounds on one core of an AMD EPYC 7513
(USC CARC Discovery, job 13326410), Julia 1.13.2, one BLAS thread; this
version at ProximalOPT 1f82ae5 (the solver code of ProximalSolvers 0.1.0) and ProximalBase 0.5.0. Times are in ms; the
passes are the iterations for ADMM and IHT.

| case | old | new | new / old | passes, old / new | largest difference |
|---|---:|---:|---:|---|---:|
| dtrace p=20 λ=0.5λmax random | 0.10 | 0.09 | 0.85 | 18 / 16 | 7e-09 |
| dtrace p=20 λ=0.2λmax random | 0.72 | 0.60 | 0.83 | 89 / 89 | 7e-08 |
| dtrace p=50 λ=0.5λmax random | 1.38 | 1.22 | 0.88 | 34 / 36 | 2e-08 |
| dtrace p=50 λ=0.2λmax random | 8.84 | 6.79 | 0.77 | 128 / 131 | 5e-08 |
| dtrace p=100 λ=0.5λmax random | 10.68 | 8.13 | 0.76 | 47 / 40 | 1e-08 |
| dtrace p=100 λ=0.2λmax random | 56.68 | 44.97 | 0.79 | 135 / 130 | 1e-07 |
| dtrace p=50 λ=0.2λmax cyclic | 4.76 | 3.60 | 0.76 | 64 / 64 | 1e-08 |
| differencePrecisionRefit p=50 | 3.28 | 0.33 | 0.10 | | 7e-08 |
| differencePrecisionRefit p=100 | 18.74 | 1.54 | 0.08 | | 1e-07 |
| covsel! p=50 λ=0.1 | 6.85 | 6.66 | 0.97 | 32 / 32 | 9e-15 |
| covsel! p=100 λ=0.1 | 29.13 | 28.33 | 0.97 | 36 / 36 | 1e-14 |
| covsel! p=50 λ=0.1 penalize_diag=false | 5.81 | 5.63 | 0.97 | 27 / 27 | 2e-14 |
| covselpath p=50 4λ | 14.98 | 14.66 | 0.98 | | 1e-14 |
| fusedGraphicalLasso p=50 | 7.39 | 7.23 | 0.98 | 17 / 17 | 4e-15 |
| fusedNeighborhoodSelection n=200 p=100 | 1.76 | 1.74 | 0.99 | 9 / 9 | 0 |
| differencePrecisionIHT p=50 s=99 | 17.51 | 15.87 | 0.91 | 454 / 454 | 0 |
| differenceLatentPrecisionIHT p=50 s=99 r=1 | 31.27 | 30.05 | 0.96 | 500 / 500 | 0 |
| diffLoss p=100 tridiagonal Δ | 34.11 | 0.26 | 0.01 | | 4e-15 |

No case is slower: over the 18 cases the geometric mean of new / old is 0.53,
and the ratio within each of the four rounds is at most 1.01.

- The direct difference estimator (`dtrace`, ProximalSolvers' coordinate descent
  on `KronQuadratic`) takes 12 to 24% less time. Its passes differ by up to 15%
  in a random order: the old coordinate step stored a coordinate that ends at
  zero, which reorders the active coordinates. In the cyclic order the passes
  are the same. Both stop at the same tolerance, and the solutions agree to
  1e-7.
- `differencePrecisionRefit` is 10 to 12 times faster and exact. The old refit,
  coordinate descent with penalty weights of 1e10 off the support, is within
  1e-7 of it.
- `covsel!` takes the iterations of the old ADMM loop and agrees with it to
  rounding. The fused solvers and the IHT solvers run the same loops on the
  ProximalBase 0.5 names: the same iterations, solutions equal to rounding
  (the IHT solutions are identical).
- `diffLoss` multiplies dense matrices where it made `p²` sparse products.
