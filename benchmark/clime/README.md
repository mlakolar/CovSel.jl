# CLIME solvers

A comparison of the solvers of [`clime`](../../src/clime.jl) with each other
and with the R packages that implement CLIME, a replication of the simulation
of Cai, Liu and Zhou (2016) for ACLIME, and a study of the first step of
ACLIME.

## Recommendations

- **Use `ParametricSimplex()`, the default.** It gives the exact solutions of
  the linear programs, and it was the fastest solver on every problem below, for
  one penalty and for a path: a whole path costs about as much as its smallest
  penalty alone. Start Julia with threads (`julia -t auto`) for the columns to
  be solved in parallel; 8 threads were 7.1–7.8 times faster than one, except
  when a few columns are much harder than the others (below).
- `DualSimplex()` was never much faster: for one penalty between 14% faster and
  3.4 times slower, and for a path up to 2.8 times slower. It is there for
  comparison.
- `LPSolver(optimizer)` makes any MathOptInterface solver available, as an
  independent check or for solvers not tried here, but a generic solver does not
  exploit the structure of the problem: HiGHS was 15–230 times slower for one
  penalty and 410–3800 times slower for a path, and near the feasibility
  boundary of a singular `S` it can stop with an unknown status.
- `PDHG()` is the most effective first order method for this problem in the
  literature we surveyed (restarted, adaptive PDHG as in PDLP, plus polishing),
  and it solves most columns exactly, but on a CPU it was 90–350 times slower
  than `ParametricSimplex` for a path, and on the hardest columns it stops at its
  tolerance. It could pay off on a GPU, which CovarianceSelection does not support.
- Of the R packages, fastclime is exact (to its 1e-5 storage threshold), but it
  was 40–590 times slower, runs on one thread, and its memory grows with p² times
  the length of the longest column path (12.4 GB for p = 500 when one column has
  3015 breakpoints). flare and clime were 3900 to 29000 times slower; flare's
  solutions miss the linear program by up to 20% in ℓ1 norm and violate its
  constraints by up to 8% at its default tolerance, and the interior point
  method of clime is accurate to about 1e-4.

On the statistics side (the replication below): the penalty of CLIME is in the
units of `SΩ`, so standardize the data, and choose λ by `climecv` with the
likelihood loss; the trace loss of Cai, Liu and Luo (2011) chose penalties much
too large. ACLIME's δ = 2 is conservative: choose δ by `aclimecv`. Its first
step, with δ₁ = 2, underestimates the diagonal of Ω by a factor of 2 to 4 in
the design of the paper, but better estimates did not improve the estimate of
Ω once δ is chosen by validation (below). The rule of the paper for large
variances in step 1 (`truncate=true`) depends on the units of the variables
and in the design of the paper applies to most variables; it is off by
default. Without it ACLIME was more accurate for the band model and the
Erdős–Rényi model with p ≤ 100, less for AR(1) and Erdős–Rényi with p = 200.

## Results

USC CARC Discovery, AMD EPYC 7513, one CPU per run unless stated otherwise,
Julia 1.13.2 and R 4.5.3 with single-threaded OpenBLAS. Run
`20261022T2303Z-covsel-clime`: one job of 24 minutes on 8 CPUs with CovarianceSelection at
commit `72a6d55`, a rerun of the two fastclime tasks that needed more memory,
and a rerun of `ParametricSimplex` and `DualSimplex` (8 minutes) with the
safeguards against cycling and loss of accuracy of commit `4d9fe8c`, which
changed their times by a few percent. Times of a few milliseconds
include fixed overheads, which is why one penalty can take longer than a whole
path for the smallest problems. A dash means not run: the slow solvers were
limited to the smaller problems (HiGHS paths to p ≤ 500, PDHG to p ≤ 1000,
fastclime to p ≤ 500, flare to p ≤ 200, clime to p = 100), and the R packages
always compute a whole path.

### Seconds for the whole path of 30 penalties, one thread

| model | p | n | ParametricSimplex | DualSimplex | HiGHS | PDHG | fastclime | flare | clime_simplex | clime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar1 | 100 | 50 | 0.0011 | 0.0015 | 1.307 | 0.160 | 0.067 | 6.630 | 32.0 | — |
| ar1 | 100 | 200 | 0.0012 | 0.0017 | 1.322 | 0.207 | 0.085 | 4.701 | 32.3 | 14.8 |
| ar1 | 200 | 100 | 0.0039 | 0.0061 | 8.713 | 0.769 | 0.854 | 57.1 | — | — |
| ar1 | 200 | 400 | 0.0055 | 0.0079 | 8.871 | 0.840 | 1.240 | 53.7 | — | — |
| ar1 | 500 | 250 | 0.033 | 0.050 | 126 | 5.718 | 19.5 | — | — | — |
| ar1 | 500 | 1000 | 0.058 | 0.071 | 129 | 7.869 | 27.6 | — | — | — |
| ar1 | 1000 | 500 | 0.323 | 0.359 | — | — | — | — | — | — |
| ar1 | 1000 | 2000 | 0.426 | 0.495 | — | 49.0 | — | — | — | — |
| ar1 | 2000 | 1000 | 3.516 | 3.653 | — | — | — | — | — | — |
| ar1 | 2000 | 4000 | 4.270 | 4.393 | — | — | — | — | — | — |
| ar1 | 4000 | 2000 | 38.0 | — | — | — | — | — | — | — |
| ar1 | 4000 | 8000 | 40.7 | — | — | — | — | — | — | — |
| band | 100 | 50 | 0.0011 | 0.0014 | 1.302 | 0.113 | 0.064 | 4.799 | 31.2 | — |
| band | 100 | 200 | 0.0036 | 0.0038 | 1.462 | 0.429 | 0.175 | 19.6 | 33.1 | 17.6 |
| band | 200 | 100 | 0.0099 | 0.0098 | 9.073 | 1.009 | 1.365 | 117 | — | — |
| band | 200 | 400 | 0.014 | 0.015 | 9.509 | 1.806 | 1.689 | 178 | — | — |
| band | 500 | 250 | 0.140 | 0.135 | 143 | 13.8 | 33.2 | — | — | — |
| band | 500 | 1000 | 0.165 | 0.176 | 143 | 20.1 | 36.7 | — | — | — |
| band | 1000 | 500 | 1.568 | 1.733 | — | — | — | — | — | — |
| band | 1000 | 2000 | 1.546 | 1.722 | — | 136 | — | — | — | — |
| band | 2000 | 1000 | 21.1 | 33.2 | — | — | — | — | — | — |
| band | 2000 | 4000 | 16.3 | 19.2 | — | — | — | — | — | — |
| er | 100 | 50 | 0.0017 | 0.0019 | 1.324 | 0.284 | 0.076 | 9.590 | 31.8 | — |
| er | 100 | 200 | 0.0034 | 0.0035 | 1.424 | 1.025 | 0.141 | 20.1 | 32.5 | 16.0 |
| er | 200 | 100 | 0.0088 | 0.010 | 8.786 | 3.089 | 1.065 | 99.8 | — | — |
| er | 200 | 400 | 0.0099 | 0.012 | 9.055 | 3.293 | 1.246 | 127 | — | — |
| er | 500 | 250 | 0.221 | 0.406 | 138 | 33.3 | 31.9 | — | — | — |
| er | 500 | 1000 | 0.093 | 0.128 | 138 | 22.0 | 23.6 | — | — | — |
| er | 1000 | 500 | 1.357 | 2.737 | — | — | — | — | — | — |
| er | 1000 | 2000 | 0.910 | 1.672 | — | 127 | — | — | — | — |
| er | 2000 | 1000 | 18.2 | 46.2 | — | — | — | — | — | — |
| er | 2000 | 4000 | 7.366 | 20.5 | — | — | — | — | — | — |

### Seconds for the smallest penalty alone, one thread

| model | p | n | ParametricSimplex | DualSimplex | HiGHS | PDHG |
|---|---:|---:|---:|---:|---:|---:|
| ar1 | 100 | 50 | 0.0052 | 0.0057 | 0.093 | 0.029 |
| ar1 | 100 | 200 | 0.0056 | 0.0055 | 0.087 | 0.033 |
| ar1 | 200 | 100 | 0.0081 | 0.0079 | 0.531 | 0.129 |
| ar1 | 200 | 400 | 0.0094 | 0.010 | 0.543 | 0.175 |
| ar1 | 500 | 250 | 0.037 | 0.033 | 7.553 | 1.407 |
| ar1 | 500 | 1000 | 0.058 | 0.058 | 7.817 | 2.155 |
| ar1 | 1000 | 500 | 0.319 | 0.302 | 74.4 | — |
| ar1 | 1000 | 2000 | 0.423 | 0.430 | 74.7 | 14.6 |
| ar1 | 2000 | 1000 | 3.361 | 3.469 | — | — |
| ar1 | 2000 | 4000 | 4.054 | 4.254 | — | — |
| band | 100 | 50 | 0.0051 | 0.0051 | 0.078 | 0.026 |
| band | 100 | 200 | 0.0078 | 0.0072 | 0.118 | 0.105 |
| band | 200 | 100 | 0.014 | 0.012 | 0.613 | 0.196 |
| band | 200 | 400 | 0.018 | 0.017 | 0.712 | 0.311 |
| band | 500 | 250 | 0.140 | 0.135 | 13.4 | 3.172 |
| band | 500 | 1000 | 0.164 | 0.165 | 11.6 | 4.880 |
| band | 1000 | 500 | 1.590 | 1.810 | 228 | — |
| band | 1000 | 2000 | 1.557 | 1.671 | 137 | 42.8 |
| band | 2000 | 1000 | 21.0 | 32.1 | — | — |
| band | 2000 | 4000 | 16.1 | 18.9 | — | — |
| er | 100 | 50 | 0.0059 | 0.0071 | 0.108 | 0.071 |
| er | 100 | 200 | 0.0077 | 0.0073 | 0.143 | 0.276 |
| er | 200 | 100 | 0.013 | 0.012 | 0.706 | 0.772 |
| er | 200 | 400 | 0.014 | 0.013 | 0.748 | 0.685 |
| er | 500 | 250 | 0.226 | 0.469 | 10.8 | 7.589 |
| er | 500 | 1000 | 0.093 | 0.105 | 9.057 | 6.717 |
| er | 1000 | 500 | 1.368 | 3.289 | 111 | — |
| er | 1000 | 2000 | 0.910 | 1.367 | 86.6 | 32.5 |
| er | 2000 | 1000 | 17.3 | 58.5 | — | — |
| er | 2000 | 4000 | 7.509 | 18.4 | — | — |

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
| er | 500 | 250 | 1e-13 / 2e-12 | 5e-12 / 6e-11 | 4e-02 / 3e-03 | 5e-10 / 1e-08 | — / — | — / — | — / — |
| er | 500 | 1000 | 3e-14 / 4e-13 | 5e-10 / 2e-09 | 9e-06 / 9e-05 | 2e-06 / 4e-05 | — / — | — / — | — / — |
| er | 1000 | 500 | 2e-14 / 3e-13 | — / — | — / — | — / — | — / — | — / — | — / — |
| er | 1000 | 2000 | 5e-14 / 8e-13 | — / — | 9e-06 / 2e-03 | — / — | — / — | — / — | — / — |
| er | 2000 | 1000 | 2e-14 / 3e-12 | — / — | — / — | — / — | — / — | — / — | — / — |
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
| ar1 | 2000 | 1000 | 3.516 | 1.676 (2.1×) | 0.880 (4.0×) | 0.454 (7.8×) |
| ar1 | 2000 | 4000 | 4.270 | 2.054 (2.1×) | 1.064 (4.0×) | 0.561 (7.6×) |
| ar1 | 4000 | 2000 | 38.0 | 19.9 (1.9×) | 9.979 (3.8×) | 5.277 (7.2×) |
| ar1 | 4000 | 8000 | 40.7 | 21.6 (1.9×) | 10.9 (3.7×) | 5.763 (7.1×) |
| band | 2000 | 1000 | 21.1 | 11.0 (1.9×) | 5.560 (3.8×) | 2.841 (7.4×) |
| er | 2000 | 1000 | 18.2 | 8.060 (2.3×) | 7.714 (2.4×) | 6.327 (2.9×) |

The columns are distributed over the threads, so the time cannot drop below
that of the hardest column. On the Erdős–Rényi problem with p = 2000 and
n = 1000, 10 of the 2000 columns take 93% of the time (the hardest alone 42%,
with 692 nonzero entries at the smallest penalty, close to its feasibility
boundary), and 8 threads are only 2.9 times faster than one.

## Replication of Cai, Liu and Zhou (2016)

[`aclime_table.jl`](aclime_table.jl) repeats the simulation of Section 5 of
the arXiv version (1212.2882) of the paper: `Σ = D Ω₁⁻¹ D` with `D` diagonal
with entries uniform on (1, 5), for the band, AR(1) (`ω₁,ᵢⱼ = 0.6^|i-j|`, a dense
precision matrix) and Erdős–Rényi models; n = 200 observations, and 200 more to
choose δ of ACLIME, from `(1:100)/50`, by the likelihood (δ = 2 in step 1).
The paper chooses the penalty of CLIME "by cross validation", without more
detail; here it is chosen from 40 values between 1 and 0.03 by 5-fold cross
validation on the training sample, with the likelihood and with the trace loss
`Σᵢ ((SΩ)ᵢᵢ - 1)²` of Cai, Liu and Luo (2011). Mean errors over 50
replications (20 for p = 200), standard errors in parentheses, and the values
of Table 1 of the paper (100 replications).

| model | p | norm | paper: ACLIME | paper: CLIME | ACLIME, `truncate=true` | ACLIME (default) | CLIME, likelihood CV | CLIME, trace CV |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| band | 50 | spectral | 0.30 | 0.32 | 0.48 (0.02) | 0.24 (0.01) | 0.40 (0.02) | 0.94 (0.03) |
| band | 50 | matrix ℓ1 | 0.62 | 0.65 | 0.99 (0.05) | 0.40 (0.01) | 0.71 (0.03) | 1.28 (0.03) |
| band | 50 | Frobenius | 0.80 | 0.83 | 0.80 (0.04) | 0.45 (0.01) | 0.69 (0.02) | 1.88 (0.04) |
| band | 100 | spectral | 0.45 | 0.50 | 0.75 (0.02) | 0.33 (0.01) | 0.45 (0.01) | 1.02 (0.03) |
| band | 100 | matrix ℓ1 | 0.79 | 0.86 | 1.24 (0.04) | 0.52 (0.02) | 0.75 (0.02) | 1.37 (0.03) |
| band | 100 | Frobenius | 1.61 | 1.73 | 1.51 (0.03) | 0.80 (0.02) | 1.25 (0.02) | 2.64 (0.04) |
| band | 200 | spectral | 0.65 | 0.72 | 0.95 (0.07) | 0.59 (0.02) | 0.97 (0.04) | 1.15 (0.03) |
| band | 200 | matrix ℓ1 | 0.94 | 0.99 | 1.35 (0.08) | 0.85 (0.02) | 1.27 (0.04) | 1.51 (0.04) |
| band | 200 | Frobenius | 3.11 | 3.29 | 2.58 (0.04) | 1.79 (0.05) | 3.20 (0.09) | 3.80 (0.09) |
| AR(1) | 50 | spectral | 0.75 | 0.73 | 0.35 (0.01) | 0.50 (0.02) | 0.41 (0.01) | 0.95 (0.03) |
| AR(1) | 50 | matrix ℓ1 | 1.19 | 1.17 | 0.82 (0.03) | 0.90 (0.03) | 0.86 (0.02) | 1.51 (0.04) |
| AR(1) | 50 | Frobenius | 1.47 | 1.47 | 0.73 (0.02) | 0.85 (0.02) | 0.84 (0.02) | 1.74 (0.04) |
| AR(1) | 100 | spectral | 1.04 | 1.05 | 0.51 (0.01) | 0.69 (0.02) | 0.71 (0.02) | 1.11 (0.03) |
| AR(1) | 100 | matrix ℓ1 | 1.62 | 1.59 | 1.12 (0.02) | 1.19 (0.02) | 1.21 (0.02) | 1.71 (0.03) |
| AR(1) | 100 | Frobenius | 2.73 | 2.82 | 1.45 (0.02) | 1.53 (0.03) | 1.79 (0.03) | 2.60 (0.05) |
| AR(1) | 200 | spectral | 1.25 | 1.30 | 0.75 (0.03) | 0.96 (0.03) | 1.07 (0.03) | 1.28 (0.03) |
| AR(1) | 200 | matrix ℓ1 | 1.93 | 1.89 | 1.48 (0.05) | 1.53 (0.05) | 1.63 (0.05) | 1.93 (0.05) |
| AR(1) | 200 | Frobenius | 4.72 | 4.97 | 2.53 (0.02) | 2.53 (0.03) | 3.06 (0.04) | 3.62 (0.04) |
| Erdős–Rényi | 50 | spectral | 0.65 | 0.72 | 0.82 (0.05) | 0.62 (0.04) | 0.76 (0.04) | 1.32 (0.05) |
| Erdős–Rényi | 50 | matrix ℓ1 | 1.47 | 1.53 | 1.89 (0.11) | 1.36 (0.08) | 1.31 (0.06) | 1.92 (0.09) |
| Erdős–Rényi | 50 | Frobenius | 1.53 | 1.62 | 1.49 (0.07) | 1.17 (0.06) | 1.31 (0.05) | 2.61 (0.08) |
| Erdős–Rényi | 100 | spectral | 0.95 | 1.21 | 1.13 (0.06) | 0.82 (0.03) | 1.22 (0.05) | 1.91 (0.04) |
| Erdős–Rényi | 100 | matrix ℓ1 | 2.15 | 2.34 | 2.25 (0.11) | 1.60 (0.07) | 2.12 (0.07) | 3.05 (0.08) |
| Erdős–Rényi | 100 | Frobenius | 3.15 | 3.61 | 2.61 (0.10) | 2.03 (0.05) | 2.86 (0.08) | 5.02 (0.09) |
| Erdős–Rényi | 200 | spectral | 2.62 | 2.28 | 1.68 (0.06) | 1.80 (0.05) | 2.32 (0.06) | 2.68 (0.06) |
| Erdős–Rényi | 200 | matrix ℓ1 | 5.47 | 5.20 | 3.66 (0.13) | 3.78 (0.12) | 4.71 (0.14) | 5.21 (0.14) |
| Erdős–Rényi | 200 | Frobenius | 9.89 | 8.86 | 5.50 (0.13) | 5.77 (0.12) | 7.92 (0.18) | 9.37 (0.17) |

Mean tuning parameters:

| model | p | δ, `truncate=true` | δ, default | λ, likelihood CV | λ, trace CV |
|---|---:|---:|---:|---:|---:|
| band | 50 | 0.13 | 0.43 | 0.034 | 0.333 |
| band | 100 | 0.24 | 0.59 | 0.099 | 0.331 |
| band | 200 | 0.47 | 0.76 | 0.238 | 0.333 |
| AR(1) | 50 | 0.24 | 0.56 | 0.058 | 0.258 |
| AR(1) | 100 | 0.39 | 0.67 | 0.118 | 0.268 |
| AR(1) | 200 | 0.50 | 0.76 | 0.186 | 0.278 |
| Erdős–Rényi | 50 | 0.17 | 0.26 | 0.037 | 0.413 |
| Erdős–Rényi | 100 | 0.39 | 0.52 | 0.114 | 0.358 |
| Erdős–Rényi | 200 | 0.69 | 0.75 | 0.244 | 0.311 |

- The errors are of the same order as those of the paper, but they do not
  match closely, and neither do the comparisons between methods: the paper's
  code is not available, and some details are not reported (the cross
  validation of CLIME; the treatment of estimates that are not positive definite
  in the likelihood, which here are excluded).
- ACLIME, with or without the rule of step 1, is the most accurate estimator in
  26 of the 27 cases. In this design most variances exceed `√(n/log p)` (at
  p = 100: 87%, 81% and 43% of the variables for the band, AR(1) and Erdős–Rényi
  models), so the rule replaces most of the estimated diagonal entries by
  `√(log p/n)`, and it matters: without it (the default) ACLIME is much more
  accurate for the band model and for the Erdős–Rényi model with p ≤ 100, with it
  for the AR(1) model, whose precision matrix is dense, and for the Erdős–Rényi
  model with p = 200.
- CLIME with the likelihood is competitive (more accurate than ACLIME with the
  rule for the band model with p ≤ 100); with the trace loss it chooses
  penalties far too large and is the least accurate estimator in every case.
- This table ran on a laptop (it measures accuracy, not time); with the same
  seeds, the Discovery run gave the same numbers to the last digit.

## The first step of ACLIME

Step 1 of ACLIME estimates the diagonal of Ω, which scales the constraints of
step 2, by a linear program with the constant δ₁ = 2, and the theory of the
paper needs `ω̂ᵢᵢ/ωᵢᵢ → 1`. The scripts in [`pilot/`](pilot) measure how good
that estimate is and how much its quality matters, in the design of the
replication above, with the same random draws (50 replications for p = 50 and
100, 20 for p = 200), without the rule for large variances (the default), and
with δ of step 2 chosen by the validation likelihood from `(1:100)/50`. The
estimates of the diagonal ("pilots") compared:

- step 1 of the paper with δ₁ = 0.25, 0.5, 1, 1.5, 2 (the default) and 3;
- step 1 on the standardized data, that is, with the scale `√(Sᵢᵢ Sⱼⱼ)` in
  place of `Sᵢᵢ ∨ Sⱼⱼ`, which makes it invariant to the units of the variables;
- refits: the residual variance of the least squares regression of each
  variable on the support of its column in step 1 (standardized, δ₁ = 2), with
  `n − 1 − |support|` degrees of freedom;
- the diagonal of `S⁻¹` with its bias removed, `(n − p − 2)/(n − 1) diag(S⁻¹)`
  (p < n only);
- `1/Sᵢᵢ`, the value for a variable unrelated to the others (`ωᵢᵢ ≥ 1/σᵢᵢ`);
- the true `ωᵢᵢ`;
- and, for comparison, ACLIME on the correlation matrix `R`, scaled back.

The study ran on Discovery (run `20261023T0701Z-covsel-aclime-pilot`, at
commit `a8aec49`). For p = 200, to fit the time budget, step 2 ran only for the
pilots in the tables below and on δ ≥ 0.1; no chosen δ was at 0.1, so this
changes nothing. With the default pilot the errors equal those of the
replication above.

Median of `ω̂ᵢᵢ/ωᵢᵢ` over the variables (10% and 90% quantiles), mean over the
replications:

| model | p | δ₁ = 0.5 | δ₁ = 1 | δ₁ = 2 | δ₁ = 3 | δ₁ = 2, standardized | δ₁ = 2, standardized, refit | 1/Sᵢᵢ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| band | 50 | 0.56 (0.47–0.65) | 0.39 (0.33–0.47) | 0.28 (0.24–0.33) | 0.24 (0.20–0.28) | 0.32 (0.28–0.37) | 0.78 (0.69–0.91) | 0.34 (0.30–0.40) |
| band | 100 | 0.54 (0.45–0.62) | 0.38 (0.32–0.44) | 0.26 (0.23–0.31) | 0.23 (0.19–0.26) | 0.30 (0.27–0.34) | 0.77 (0.67–0.88) | 0.33 (0.30–0.39) |
| band | 200 | 0.52 (0.44–0.60) | 0.36 (0.30–0.42) | 0.25 (0.22–0.29) | 0.22 (0.19–0.25) | 0.28 (0.26–0.32) | 0.76 (0.66–0.87) | 0.33 (0.29–0.38) |
| AR(1) | 50 | 0.63 (0.55–0.72) | 0.49 (0.42–0.57) | 0.38 (0.33–0.44) | 0.33 (0.29–0.39) | 0.42 (0.38–0.48) | 0.78 (0.69–0.90) | 0.48 (0.42–0.55) |
| AR(1) | 100 | 0.61 (0.53–0.70) | 0.47 (0.41–0.55) | 0.37 (0.31–0.43) | 0.32 (0.28–0.37) | 0.40 (0.36–0.46) | 0.78 (0.68–0.89) | 0.47 (0.42–0.54) |
| AR(1) | 200 | 0.60 (0.52–0.69) | 0.46 (0.40–0.53) | 0.36 (0.31–0.41) | 0.32 (0.27–0.36) | 0.39 (0.35–0.44) | 0.77 (0.67–0.88) | 0.47 (0.42–0.54) |
| Erdős–Rényi | 50 | 0.79 (0.55–0.96) | 0.65 (0.40–0.85) | 0.51 (0.27–0.73) | 0.44 (0.20–0.66) | 0.55 (0.30–0.75) | 0.97 (0.83–1.12) | 0.63 (0.26–0.94) |
| Erdős–Rényi | 100 | 0.74 (0.54–0.90) | 0.59 (0.39–0.76) | 0.46 (0.27–0.64) | 0.40 (0.21–0.57) | 0.49 (0.30–0.65) | 0.89 (0.75–1.04) | 0.59 (0.30–0.84) |
| Erdős–Rényi | 200 | 0.69 (0.54–0.84) | 0.55 (0.39–0.69) | 0.43 (0.28–0.57) | 0.37 (0.23–0.51) | 0.44 (0.30–0.58) | 0.77 (0.64–0.92) | 0.56 (0.34–0.77) |

Error in spectral norm with δ chosen by validation, and the mean paired
difference to the default pilot (* when larger than twice its standard error):

| model | p | δ₁ = 2 (default) | δ₁ = 0.5 | standardized, refit | diag of S⁻¹ | 1/Sᵢᵢ | true ωᵢᵢ | ACLIME on R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| band | 50 | 0.24 | 0.23 (-0.01*) | 0.24 (+0.00) | 0.23 (-0.00) | 0.24 (+0.00) | 0.25 (+0.01) | 0.26 (+0.02*) |
| band | 100 | 0.33 | 0.33 (-0.00) | 0.35 (+0.02*) | 0.34 (+0.00) | 0.36 (+0.02*) | 0.37 (+0.04*) | 0.36 (+0.03*) |
| band | 200 | 0.59 | 0.57 (-0.03*) | 0.62 (+0.03*) | — | 0.62 (+0.02*) | 0.63 (+0.04*) | 0.53 (-0.06*) |
| AR(1) | 50 | 0.50 | 0.51 (+0.01) | 0.52 (+0.01*) | 0.51 (+0.01) | 0.52 (+0.01*) | 0.52 (+0.02*) | 0.51 (+0.00) |
| AR(1) | 100 | 0.69 | 0.69 (-0.01*) | 0.71 (+0.02*) | 0.70 (+0.00) | 0.71 (+0.02*) | 0.71 (+0.02*) | 0.68 (-0.01) |
| AR(1) | 200 | 0.96 | 0.94 (-0.02*) | 0.97 (+0.01) | — | 0.98 (+0.02*) | 0.98 (+0.02*) | 0.87 (-0.08*) |
| Erdős–Rényi | 50 | 0.62 | 0.62 (+0.00) | 0.57 (-0.05*) | 0.55 (-0.07*) | 0.58 (-0.04*) | 0.60 (-0.02) | 0.53 (-0.09*) |
| Erdős–Rényi | 100 | 0.82 | 0.82 (+0.00) | 0.83 (+0.01) | 0.81 (-0.01) | 0.82 (+0.00) | 0.89 (+0.07*) | 0.82 (+0.00) |
| Erdős–Rényi | 200 | 1.80 | 1.69 (-0.11*) | 1.73 (-0.07) | — | 1.94 (+0.14*) | 1.83 (+0.03) | 1.75 (-0.04) |

The δ chosen by validation, and the errors at δ = 2:

| model | p | δ, default pilot | δ, true ωᵢᵢ | spectral at chosen δ | spectral at δ = 2, default pilot | at δ = 2, true ωᵢᵢ |
|---|---:|---:|---:|---:|---:|---:|
| band | 50 | 0.43 | 0.23 | 0.24 | 0.96 | 1.13 |
| band | 100 | 0.59 | 0.31 | 0.33 | 1.08 | 1.24 |
| band | 200 | 0.76 | 0.39 | 0.59 | 1.22 | 1.37 |
| AR(1) | 50 | 0.56 | 0.35 | 0.50 | 1.02 | 1.11 |
| AR(1) | 100 | 0.67 | 0.41 | 0.69 | 1.18 | 1.27 |
| AR(1) | 200 | 0.76 | 0.46 | 0.96 | 1.35 | 1.44 |
| Erdős–Rényi | 50 | 0.26 | 0.21 | 0.62 | 1.24 | 1.65 |
| Erdős–Rényi | 100 | 0.52 | 0.32 | 0.82 | 2.02 | 2.59 |
| Erdős–Rényi | 200 | 0.75 | 0.41 | 1.80 | 3.15 | 3.90 |

The bias as n grows, p = 100, one sample for each n ([`scaling.jl`](pilot/scaling.jl)):

| model | n | λ₁ (δ₁ = 2) | δ₁ = 0.5 | δ₁ = 2 | δ₁ = 0.5, standardized | δ₁ = 2, standardized |
|---|---:|---:|---:|---:|---:|---:|
| band | 200 | 0.303 | 0.54 (0.46–0.63) | 0.26 (0.24–0.30) | 0.64 (0.57–0.71) | 0.30 (0.27–0.34) |
| band | 800 | 0.152 | 0.71 (0.63–0.77) | 0.38 (0.33–0.44) | 0.77 (0.73–0.82) | 0.45 (0.42–0.47) |
| band | 3200 | 0.076 | 0.81 (0.73–0.85) | 0.52 (0.44–0.57) | 0.87 (0.84–0.89) | 0.61 (0.59–0.63) |
| band | 12800 | 0.038 | 0.89 (0.86–0.91) | 0.68 (0.60–0.72) | 0.93 (0.91–0.94) | 0.76 (0.74–0.77) |
| AR(1) | 200 | 0.303 | 0.62 (0.53–0.71) | 0.37 (0.32–0.42) | 0.70 (0.62–0.80) | 0.40 (0.36–0.46) |
| AR(1) | 800 | 0.152 | 0.74 (0.67–0.80) | 0.48 (0.42–0.53) | 0.80 (0.75–0.85) | 0.53 (0.50–0.57) |
| AR(1) | 3200 | 0.076 | 0.82 (0.75–0.86) | 0.60 (0.50–0.63) | 0.88 (0.85–0.90) | 0.66 (0.64–0.68) |
| AR(1) | 12800 | 0.038 | 0.89 (0.86–0.92) | 0.71 (0.65–0.75) | 0.93 (0.91–0.94) | 0.78 (0.77–0.79) |
| Erdős–Rényi | 200 | 0.303 | 0.71 (0.54–0.86) | 0.43 (0.25–0.61) | 0.79 (0.60–0.97) | 0.46 (0.31–0.66) |
| Erdős–Rényi | 800 | 0.152 | 0.84 (0.72–0.94) | 0.60 (0.43–0.73) | 0.89 (0.79–0.96) | 0.64 (0.49–0.76) |
| Erdős–Rényi | 3200 | 0.076 | 0.90 (0.81–0.96) | 0.68 (0.52–0.82) | 0.93 (0.85–0.98) | 0.75 (0.60–0.87) |
| Erdős–Rényi | 12800 | 0.038 | 0.96 (0.90–0.99) | 0.84 (0.70–0.92) | 0.97 (0.94–0.99) | 0.89 (0.80–0.93) |

- **The estimates of step 1 are far too small.** With δ₁ = 2 the median of
  `ω̂ᵢᵢ/ωᵢᵢ` is 0.25–0.51 for n = 200, below even `1/Sᵢᵢ`, and still 0.68–0.84
  for n = 12800; each fourfold increase of n, which halves λ₁, removed only
  16–50% of the remaining bias. The constraint of step 1 for entry i of column
  j, `λ₁ (Sᵢᵢ ∨ Sⱼⱼ) bⱼⱼ`, is wider than the noise of `(Ŝ ωⱼ − eⱼ)ᵢ`, of order
  `λ₁ √(σᵢᵢ ωⱼⱼ)`, by the factor `(σᵢᵢ ∨ σⱼⱼ) √(ωⱼⱼ/σᵢᵢ) ≥ 1`, which is 1 only
  when Σ is a multiple of the identity (the case of Remark 1 of the paper); the
  ℓ1 minimization uses the slack to shrink the regression of variable j on the
  others, which inflates its residual variance `1/ωⱼⱼ`. A smaller δ₁ is less
  biased, standardizing helps little, and the least squares refit removes most
  of the bias.
- **For step 2 this matters little.** With δ chosen by validation, the errors of
  all the pilots are mostly within 5% of each other, and the true diagonal was
  never significantly better than the default in spectral norm (significantly
  worse, by 2–11%, in 6 of the 9 cases). The constraints of step 2 scale with
  `√ω̂ᵢᵢ`, so a pilot that is too small by a common factor is offset by a larger
  δ: with the default pilot the chosen δ is 1.2–2 times that with the true
  diagonal. The largest gains from better pilots were for the Erdős–Rényi
  model: for p = 50 the diagonal of `S⁻¹` and the refit reduced the errors by
  6–14%, depending on the norm, and for p = 200 δ₁ = 0.5 by 6% (8% in
  Frobenius norm). In spectral norm δ₁ = 0.5 was up to 6% better than δ₁ = 2,
  or within 0.5%, except for AR(1) with p = 50 (2% worse). The matrix ℓ1 and
  Frobenius norms give the same picture.
- **At δ = 2 the pilot does matter,** but there the errors are 1.4–4 times
  those at the chosen δ, and larger still with the true diagonal: δ = 2 is
  conservative, and the small pilot partly offsets it: with the true diagonal
  it would act like the constant `δ √(ω̂ᵢᵢ/ωᵢᵢ)`.
- ACLIME on the correlation matrix, which is invariant to the units of the
  variables, was more accurate for the band and AR(1) models with p = 200 and
  the Erdős–Rényi model with p = 50, and less accurate for the band model with
  p ≤ 100. Without standardizing, neither step of ACLIME is invariant:
  rescaling each variable by a factor uniform on (1, 5) changed the estimates
  of step 1 by up to 30% and the estimate by 17%
  ([`invariance.jl`](pilot/invariance.jl)).

So δ₁ = 2 gives poor estimates of the diagonal but a good enough pilot: better
estimates did not give better estimates of Ω once δ is chosen by validation.
The default stays δ₁ = 2, as in the paper; choose δ by `aclimecv`.

## What is compared

Every solver gets the same sample correlation matrix `S` and the same 30
penalties, from 0.8 down to `√(log p / n)`, equally spaced on the log scale.
Each is timed on the whole path and on the smallest penalty alone, on one CPU
(an AMD EPYC 7513 node of USC CARC Discovery), and its column solutions,
before symmetrization, are compared with those of `ParametricSimplex`.

CovarianceSelection:

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
[`summarize.jl`](summarize.jl) makes the tables. The study of the first step
of ACLIME ran as two more jobs, [`pilot/pilot.sbatch`](pilot/pilot.sbatch) and,
for p = 200 one replication per task, [`pilot/shard.sbatch`](pilot/shard.sbatch).
On a laptop, for one problem:

```sh
julia --project=benchmark/clime -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/clime benchmark/clime/prepare.jl /tmp/clime quick
julia --project=benchmark/clime benchmark/clime/run_julia.jl /tmp/clime/ar1_p50_n100_s1001 /tmp/clime-out \
  ParametricSimplex DualSimplex HiGHS PDHG
Rscript benchmark/clime/install_r.R ~/Rlib-clime
Rscript benchmark/clime/bench_r.R /tmp/clime/ar1_p50_n100_s1001/S.bin 50 \
  /tmp/clime/ar1_p50_n100_s1001/lambda.txt fastclime /tmp/clime-out/ar1_p50_n100_s1001__fastclime lib=~/Rlib-clime
```
