# CLIME solvers

A comparison of the solvers of [`clime`](../../src/clime.jl) with each other
and with the R packages that implement CLIME, and a replication of the
simulation of Cai, Liu and Zhou (2016) for ACLIME.

## Recommendations

- **Use `ParametricSimplex()`, the default.** It gives the exact solutions of
  the linear programs, and it was the fastest solver on every problem below, for
  one penalty and for a path: a whole path costs about as much as its smallest
  penalty alone. Start Julia with threads (`julia -t auto`) for the columns to
  be solved in parallel; 8 threads were 5.6–7.8 times faster than one, except
  when a few columns are much harder than the others (below).
- `DualSimplex()` was never faster: about as fast for one penalty but up to 3.5
  times slower in the worst cases, and up to 3 times slower for a path. It is
  there for comparison.
- `LPSolver(optimizer)` makes any MathOptInterface solver available, as an
  independent check or for solvers not tried here, but a generic solver does not
  exploit the structure of the problem: HiGHS was 12–210 times slower for one
  penalty and 380–3400 times slower for a path, and near the feasibility
  boundary of a singular `S` it can stop with an unknown status.
- `PDHG()` is the most effective first order method for this problem in the
  literature we surveyed (restarted, adaptive PDHG as in PDLP, plus polishing),
  and it solves most columns exactly, but on a CPU it was 80–310 times slower
  than `ParametricSimplex` for a path, and on the hardest columns it stops at its
  tolerance. It could pay off on a GPU, which CovSel does not support.
- Of the R packages, fastclime is exact (to its 1e-5 storage threshold), but it
  was 40–530 times slower, runs on one thread, and its memory grows with p² times
  the length of the longest column path (12.4 GB for p = 500 when one column has
  3015 breakpoints). flare and clime were 3000 to 30000 times slower; flare's
  solutions miss the linear program by up to 20% in ℓ1 norm and violate its
  constraints by up to 8% at its default tolerance, and the interior point
  method of clime is accurate to about 1e-4.

On the statistics side (the replication below): the penalty of CLIME is in the
units of `SΩ`, so standardize the data, and choose λ by `climecv` with the
likelihood loss; the trace loss of Cai, Liu and Luo (2011) chose penalties much
too large. ACLIME's δ = 2 is conservative: choose δ by `aclimecv`. Its rule for
large variances in step 1 matters: in the design of the paper it applies to
most variables, and without it (`truncate=false`) ACLIME was the most accurate
estimator for two of the three models.

## Results

USC CARC Discovery, AMD EPYC 7513, one CPU per run unless stated otherwise,
Julia 1.13.2 and R 4.5.3 with single-threaded OpenBLAS, CovSel at commit
`72a6d55` (later commits do not change the solvers). Run
`20261022T2303Z-covsel-clime`: one job of 24 minutes on 8 CPUs, plus a rerun of
the two fastclime tasks that needed more memory. Times of a few milliseconds
include fixed overheads, which is why one penalty can take longer than a whole
path for the smallest problems. A dash means not run: the slow solvers were
limited to the smaller problems (HiGHS paths to p ≤ 500, PDHG to p ≤ 1000,
fastclime to p ≤ 500, flare to p ≤ 200, clime to p = 100), and the R packages
always compute a whole path.

### Seconds for the whole path of 30 penalties, one thread

| model | p | n | ParametricSimplex | DualSimplex | HiGHS | PDHG | fastclime | flare | clime_simplex | clime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar1 | 100 | 50 | 0.0012 | 0.0016 | 1.307 | 0.160 | 0.067 | 6.630 | 32.0 | — |
| ar1 | 100 | 200 | 0.0014 | 0.0019 | 1.322 | 0.207 | 0.085 | 4.701 | 32.3 | 14.8 |
| ar1 | 200 | 100 | 0.0042 | 0.0063 | 8.713 | 0.769 | 0.854 | 57.1 | — | — |
| ar1 | 200 | 400 | 0.0058 | 0.0081 | 8.871 | 0.840 | 1.240 | 53.7 | — | — |
| ar1 | 500 | 250 | 0.037 | 0.052 | 126 | 5.718 | 19.5 | — | — | — |
| ar1 | 500 | 1000 | 0.058 | 0.076 | 129 | 7.869 | 27.6 | — | — | — |
| ar1 | 1000 | 500 | 0.345 | 0.402 | — | — | — | — | — | — |
| ar1 | 1000 | 2000 | 0.455 | 0.520 | — | 49.0 | — | — | — | — |
| ar1 | 2000 | 1000 | 3.678 | 4.092 | — | — | — | — | — | — |
| ar1 | 2000 | 4000 | 4.362 | 4.713 | — | — | — | — | — | — |
| ar1 | 4000 | 2000 | 39.7 | — | — | — | — | — | — | — |
| ar1 | 4000 | 8000 | 42.3 | — | — | — | — | — | — | — |
| band | 100 | 50 | 0.0011 | 0.0014 | 1.302 | 0.113 | 0.064 | 4.799 | 31.2 | — |
| band | 100 | 200 | 0.0039 | 0.0039 | 1.462 | 0.429 | 0.175 | 19.6 | 33.1 | 17.6 |
| band | 200 | 100 | 0.011 | 0.011 | 9.073 | 1.009 | 1.365 | 117 | — | — |
| band | 200 | 400 | 0.015 | 0.017 | 9.509 | 1.806 | 1.689 | 178 | — | — |
| band | 500 | 250 | 0.161 | 0.167 | 143 | 13.8 | 33.2 | — | — | — |
| band | 500 | 1000 | 0.180 | 0.190 | 143 | 20.1 | 36.7 | — | — | — |
| band | 1000 | 500 | 1.638 | 1.950 | — | — | — | — | — | — |
| band | 1000 | 2000 | 1.702 | 1.858 | — | 136 | — | — | — | — |
| band | 2000 | 1000 | 22.4 | 33.5 | — | — | — | — | — | — |
| band | 2000 | 4000 | 16.7 | 19.7 | — | — | — | — | — | — |
| er | 100 | 50 | 0.0018 | 0.0020 | 1.324 | 0.284 | 0.076 | 9.590 | 31.8 | — |
| er | 100 | 200 | 0.0035 | 0.0035 | 1.424 | 1.025 | 0.141 | 20.1 | 32.5 | 16.0 |
| er | 200 | 100 | 0.010 | 0.011 | 8.786 | 3.089 | 1.065 | 99.8 | — | — |
| er | 200 | 400 | 0.011 | 0.013 | 9.055 | 3.293 | 1.246 | 127 | — | — |
| er | 500 | 250 | 0.221 | 0.417 | 138 | 33.3 | 31.9 | — | — | — |
| er | 500 | 1000 | 0.090 | 0.135 | 138 | 22.0 | 23.6 | — | — | — |
| er | 1000 | 500 | 1.384 | 2.707 | — | — | — | — | — | — |
| er | 1000 | 2000 | 0.959 | 1.646 | — | 127 | — | — | — | — |
| er | 2000 | 1000 | 16.6 | 47.8 | — | — | — | — | — | — |
| er | 2000 | 4000 | 8.005 | 24.7 | — | — | — | — | — | — |

### Seconds for the smallest penalty alone, one thread

| model | p | n | ParametricSimplex | DualSimplex | HiGHS | PDHG |
|---|---:|---:|---:|---:|---:|---:|
| ar1 | 100 | 50 | 0.0079 | 0.0078 | 0.093 | 0.029 |
| ar1 | 100 | 200 | 0.0061 | 0.0060 | 0.087 | 0.033 |
| ar1 | 200 | 100 | 0.0081 | 0.0095 | 0.531 | 0.129 |
| ar1 | 200 | 400 | 0.0096 | 0.0093 | 0.543 | 0.175 |
| ar1 | 500 | 250 | 0.043 | 0.042 | 7.553 | 1.407 |
| ar1 | 500 | 1000 | 0.062 | 0.061 | 7.817 | 2.155 |
| ar1 | 1000 | 500 | 0.350 | 0.342 | 74.4 | — |
| ar1 | 1000 | 2000 | 0.456 | 0.451 | 74.7 | 14.6 |
| ar1 | 2000 | 1000 | 3.680 | 3.822 | — | — |
| ar1 | 2000 | 4000 | 4.403 | 4.353 | — | — |
| band | 100 | 50 | 0.0051 | 0.0053 | 0.078 | 0.026 |
| band | 100 | 200 | 0.0083 | 0.0071 | 0.118 | 0.105 |
| band | 200 | 100 | 0.015 | 0.014 | 0.613 | 0.196 |
| band | 200 | 400 | 0.019 | 0.020 | 0.712 | 0.311 |
| band | 500 | 250 | 0.164 | 0.170 | 13.4 | 3.172 |
| band | 500 | 1000 | 0.180 | 0.179 | 11.6 | 4.880 |
| band | 1000 | 500 | 1.641 | 2.008 | 228 | — |
| band | 1000 | 2000 | 1.702 | 1.835 | 137 | 42.8 |
| band | 2000 | 1000 | 22.4 | 34.5 | — | — |
| band | 2000 | 4000 | 16.8 | 19.7 | — | — |
| er | 100 | 50 | 0.0087 | 0.0084 | 0.108 | 0.071 |
| er | 100 | 200 | 0.0082 | 0.0078 | 0.143 | 0.276 |
| er | 200 | 100 | 0.014 | 0.015 | 0.706 | 0.772 |
| er | 200 | 400 | 0.018 | 0.018 | 0.748 | 0.685 |
| er | 500 | 250 | 0.223 | 0.473 | 10.8 | 7.589 |
| er | 500 | 1000 | 0.093 | 0.106 | 9.057 | 6.717 |
| er | 1000 | 500 | 1.412 | 3.257 | 111 | — |
| er | 1000 | 2000 | 0.994 | 1.425 | 86.6 | 32.5 |
| er | 2000 | 1000 | 16.7 | 57.8 | — | — |
| er | 2000 | 4000 | 8.652 | 20.3 | — | — |

### Accuracy: largest relative difference of the column ℓ1 norms from ParametricSimplex / largest relative constraint violation

| model | p | n | DualSimplex | HiGHS | PDHG | fastclime | flare | clime_simplex | clime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar1 | 100 | 50 | 5e-15 / 7e-15 | 4e-14 / 3e-14 | 4e-07 / 2e-07 | 6e-15 / 5e-15 | 6e-02 / 1e-02 | 3e-12 / 2e-12 | — / — |
| ar1 | 100 | 200 | 2e-15 / 2e-14 | 2e-14 / 2e-14 | 3e-15 / 3e-15 | 5e-15 / 2e-14 | 2e-02 / 2e-02 | 3e-12 / 8e-12 | 1e-04 / 0 |
| ar1 | 200 | 100 | 3e-14 / 5e-14 | 5e-13 / 6e-13 | 2e-14 / 2e-15 | 6e-07 / 1e-06 | 5e-02 / 5e-02 | — / — | — / — |
| ar1 | 200 | 400 | 8e-15 / 4e-14 | 3e-14 / 3e-13 | 1e-06 / 7e-06 | 7e-08 / 2e-06 | 1e-02 / 2e-02 | — / — | — / — |
| ar1 | 500 | 250 | 8e-15 / 4e-14 | 1e-11 / 2e-11 | 4e-15 / 4e-15 | 2e-06 / 2e-05 | — / — | — / — | — / — |
| ar1 | 500 | 1000 | 4e-15 / 6e-14 | 3e-13 / 3e-12 | 9e-06 / 2e-04 | 2e-06 / 1e-04 | — / — | — / — | — / — |
| ar1 | 1000 | 500 | 1e-14 / 1e-13 | — / — | — / — | — / — | — / — | — / — | — / — |
| ar1 | 1000 | 2000 | 4e-15 / 7e-14 | — / — | 6e-06 / 1e-04 | — / — | — / — | — / — | — / — |
| ar1 | 2000 | 1000 | 5e-14 / 3e-13 | — / — | — / — | — / — | — / — | — / — | — / — |
| ar1 | 2000 | 4000 | 8e-15 / 3e-13 | — / — | — / — | — / — | — / — | — / — | — / — |
| band | 100 | 50 | 3e-14 / 5e-15 | 3e-14 / 8e-15 | 3e-14 / 9e-16 | 1e-14 / 1e-13 | 1e-01 / 4e-02 | 3e-12 / 1e-12 | — / — |
| band | 100 | 200 | 1e-14 / 8e-14 | 6e-14 / 4e-13 | 1e-14 / 5e-15 | 3e-07 / 3e-06 | 1e-01 / 7e-02 | 4e-12 / 6e-12 | 1e-04 / 0 |
| band | 200 | 100 | 2e-14 / 5e-14 | 2e-12 / 2e-12 | 3e-14 / 2e-15 | 6e-13 / 4e-13 | 1e-01 / 6e-02 | — / — | — / — |
| band | 200 | 400 | 6e-15 / 1e-13 | 5e-13 / 1e-12 | 2e-06 / 3e-06 | 1e-06 / 3e-05 | 2e-01 / 8e-02 | — / — | — / — |
| band | 500 | 250 | 3e-14 / 2e-13 | 7e-12 / 3e-11 | 2e-06 / 6e-06 | 6e-06 / 1e-05 | — / — | — / — | — / — |
| band | 500 | 1000 | 2e-14 / 6e-13 | 3e-11 / 5e-10 | 2e-06 / 1e-05 | 1e-06 / 8e-05 | — / — | — / — | — / — |
| band | 1000 | 500 | 5e-14 / 3e-13 | — / — | — / — | — / — | — / — | — / — | — / — |
| band | 1000 | 2000 | 1e-13 / 1e-12 | — / — | 2e-06 / 3e-05 | — / — | — / — | — / — | — / — |
| band | 2000 | 1000 | 5e-14 / 1e-12 | — / — | — / — | — / — | — / — | — / — | — / — |
| band | 2000 | 4000 | 6e-14 / 1e-12 | — / — | — / — | — / — | — / — | — / — | — / — |
| er | 100 | 50 | 1e-14 / 4e-14 | 1e-11 / 7e-12 | 1e-06 / 2e-06 | 9e-14 / 1e-13 | 9e-02 / 4e-02 | 3e-12 / 2e-12 | — / — |
| er | 100 | 200 | 2e-14 / 2e-13 | 4e-12 / 7e-12 | 3e-07 / 3e-06 | 1e-12 / 3e-11 | 1e-01 / 6e-02 | 5e-12 / 2e-11 | 4e-04 / 0 |
| er | 200 | 100 | 3e-14 / 2e-13 | 1e-11 / 4e-11 | 3e-05 / 4e-05 | 2e-13 / 2e-12 | 2e-01 / 4e-02 | — / — | — / — |
| er | 200 | 400 | 4e-14 / 1e-13 | 4e-11 / 3e-11 | 5e-06 / 2e-05 | 5e-13 / 3e-11 | 2e-01 / 5e-02 | — / — | — / — |
| er | 500 | 250 | 1e-13 / 1e-12 | 5e-12 / 6e-11 | 4e-02 / 3e-03 | 5e-10 / 1e-08 | — / — | — / — | — / — |
| er | 500 | 1000 | 3e-14 / 4e-13 | 5e-10 / 2e-09 | 9e-06 / 9e-05 | 2e-06 / 4e-05 | — / — | — / — | — / — |
| er | 1000 | 500 | 2e-14 / 3e-13 | — / — | — / — | — / — | — / — | — / — | — / — |
| er | 1000 | 2000 | 5e-14 / 8e-13 | — / — | 9e-06 / 2e-03 | — / — | — / — | — / — | — / — |
| er | 2000 | 1000 | 2e-14 / 4e-12 | — / — | — / — | — / — | — / — | — / — | — / — |
| er | 2000 | 4000 | 6e-15 / 3e-13 | — / — | — / — | — / — | — / — | — / — | — / — |

The exact solvers (`DualSimplex`, HiGHS, fastclime, clime's simplex) agree with
`ParametricSimplex` to rounding (fastclime to its 1e-5 storage threshold).
PDHG is accurate to its tolerance except on the Erdős–Rényi problem with
p = 500, n = 250, whose hardest columns have thousands of breakpoints. flare
and clime's interior point method do not solve the linear programs exactly. On
the two Erdős–Rényi problems with p = 500, the number of pivots of fastclime per
column (mean 18 and 19, largest 842 and 3015) is the same as the number of
breakpoints of `ParametricSimplex`: both follow the same path, with different
linear algebra.

### ParametricSimplex with several threads: seconds for the path (speedup over one thread)

| model | p | n | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| ar1 | 2000 | 1000 | 3.678 | 1.816 (2.0×) | 0.900 (4.1×) | 0.652 (5.6×) |
| ar1 | 2000 | 4000 | 4.362 | 2.109 (2.1×) | 1.058 (4.1×) | 0.563 (7.8×) |
| ar1 | 4000 | 2000 | 39.7 | 20.2 (2.0×) | 10.2 (3.9×) | 5.387 (7.4×) |
| ar1 | 4000 | 8000 | 42.3 | 20.9 (2.0×) | 10.7 (4.0×) | 6.591 (6.4×) |
| band | 2000 | 1000 | 22.4 | 11.2 (2.0×) | 5.534 (4.0×) | 3.716 (6.0×) |
| er | 2000 | 1000 | 16.6 | 7.960 (2.1×) | 6.867 (2.4×) | 7.063 (2.4×) |

The columns are distributed over the threads, so the time cannot drop below
that of the hardest column. On the Erdős–Rényi problem with p = 2000 and
n = 1000, 10 of the 2000 columns take 93% of the time (the hardest alone 42%,
with 692 nonzero entries at the smallest penalty, close to its feasibility
boundary), and 8 threads are only 2.4 times faster than one.

<!-- REPLICATION -->


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
