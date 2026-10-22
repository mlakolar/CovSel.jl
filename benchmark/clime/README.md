# CLIME solvers

A comparison of the solvers of [`clime`](../../src/clime.jl) with each other
and with the R packages that implement CLIME, and a replication of the
simulation of Cai, Liu and Zhou (2016) for ACLIME.

<!-- RESULTS -->

## What is compared

Every solver gets the same sample correlation matrix `S` and the same 30
penalties, from 0.8 down to `√(log p / n)`, equally spaced on the log scale.
Each is timed on the whole path and on the smallest penalty alone, on one CPU
(an AMD EPYC 7513 node of USC CARC Discovery), and its column solutions,
before symmetrization, are compared with those of `ParametricSimplex`.

CovSel:

- `ParametricSimplex()`: the parametric simplex method, which follows the
  piecewise linear solution path of each column through all its breakpoints,
  with one dual simplex pivot per breakpoint, on the k × k block of `S` formed
  by the tight constraints and the nonzero entries of the solution;
- `DualSimplex()`: the same pivots, warm started from the optimal basis of the
  previous penalty, without following the path between penalties;
- `LPSolver(HiGHS.Optimizer)`: each column problem as a linear program for
  HiGHS (dual simplex), warm started from the previous basis;
- `PDHG()`: the primal-dual hybrid gradient method with the restarts, primal
  weights and adaptive step sizes of PDLP, on all columns at once, with its
  solutions polished (solved exactly on the support of the iterates and
  checked against the optimality conditions).

R ([`bench_r.R`](bench_r.R), packages installed by
[`install_r.R`](install_r.R)), on `S` as given (no standardization, no
perturbation):

- [fastclime](https://cran.r-project.org/src/contrib/Archive/fastclime/) 1.4.1.1
  (Pang, Liu and Vanderbei, 2014; archived on CRAN in 2023): the parametric
  simplex method in C, on the full linear program with Vanderbei's sparse LU
  factorization. It returns the breakpoints of each column; the solution at a
  penalty is interpolated between the two breakpoints around it. (The
  package's own `fastclime.selector` returns the solution at the next
  breakpoint instead, which can differ by much more.) Its memory grows with
  the number of breakpoints times p², which limits it to p ≤ 500 here;
- [flare](https://cran.r-project.org/package=flare) 1.8 (Li, Zhao, Yuan and
  Liu, 2015), `sugm(method = "clime")`: linearized ADMM with screening, at its
  default tolerance;
- [clime](https://cran.r-project.org/package=clime) 0.5.0 (Cai, Liu and Luo,
  2011): one linear program per column and penalty with lpSolve
  (`linsolver = "simplex"`), or a primal-dual interior point method
  (`"primaldual"`, the default, which needs a nonsingular `S`).

The problems ([`problems.jl`](problems.jl)) are the precision matrices of Cai,
Liu and Luo (2011) and Cai, Liu and Zhou (2016): the AR(1) covariance
`Σᵢⱼ = 0.6^|i-j|` (a tridiagonal precision matrix), the band precision matrix
with `ωᵢ,ᵢ±₁ = 0.6` and `ωᵢ,ᵢ±₂ = 0.3`, and an Erdős–Rényi graph with about 4
neighbors per node and weights uniform on (0.4, 0.8); with p = 100, 200, 500,
1000, 2000 (and 4000 for AR(1)) and n = p/2 or 2p.

## Running

The campaign ran as one Slurm job on 8 CPUs, through
[`campaign.sbatch`](campaign.sbatch): [`prepare.jl`](prepare.jl) writes the
problems, [`make_tasks.jl`](make_tasks.jl) the list of tasks, which GNU
Parallel runs one CPU each ([`run_julia.jl`](run_julia.jl) for the Julia
solvers, [`bench_r.R`](bench_r.R) for the R packages,
[`aclime_table.jl`](aclime_table.jl) for the replication), and
[`summarize.jl`](summarize.jl) makes the tables. On a laptop, for one problem:

```sh
julia --project=benchmark/clime -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/clime benchmark/clime/prepare.jl /tmp/clime quick
julia --project=benchmark/clime benchmark/clime/run_julia.jl /tmp/clime/ar1_p50_n100_s1001 /tmp/clime-out \
  ParametricSimplex DualSimplex HiGHS PDHG
Rscript benchmark/clime/install_r.R ~/Rlib-clime
Rscript benchmark/clime/bench_r.R /tmp/clime/ar1_p50_n100_s1001/S.bin 50 \
  /tmp/clime/ar1_p50_n100_s1001/lambda.txt fastclime /tmp/clime-out/ar1_p50_n100_s1001__fastclime lib=~/Rlib-clime
```
