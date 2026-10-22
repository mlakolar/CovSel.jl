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
  - `differencePrecisionActiveShooting`: the package solver, at λ directly
    from zero (the default), and with `warmStart=false`, which first solves
    along a path of 50 decreasing λ values,
  - `Reference.differencePrecisionNaive` from
    [`../test/reference.jl`](../test/reference.jl): full coordinate descent
    sweeps over dense matrices, the reference the tests check the package
    solver against;

  and, with the size of the true support, `differencePrecisionIHT` and the
  sparse plus rank one `differenceLatentPrecisionIHT`. Before the timings,
  `run.jl` checks that the package solver agrees with the reference solver.
- **ADMM**: `covsel!` and `fusedGraphicalLasso` on the same covariances.

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
