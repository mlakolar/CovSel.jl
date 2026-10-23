# slassoinv: sparse matrix inversion with the scaled lasso

[`slassoinv`](../../src/slassoinv.jl) estimates a precision matrix one
column at a time by the scaled lasso of Sun and Zhang (2013): column `j`
regresses variable `j` on the others, estimates the residual standard deviation
`σ̂ⱼ` jointly with the coefficients, with the penalty `λ₀ σ̂ⱼ √Sₖₖ` on
coefficient `k`, and gives `Ωⱼⱼ = 1/σ̂ⱼ²` and `Ωₖⱼ = -β̂ₖ/σ̂ⱼ²`. The penalty level
`λ₀` is a universal constant (by default that of the simulations of Sun and
Zhang), so the estimator has nothing to tune. This directory compares it

- for speed, with the R package scalreg (Sun, 2013), by the authors of the
  method, and with the scaled lasso of
  [CoordinateDescent](https://github.com/mlakolar/CoordinateDescent.jl) applied
  to each column;
- for accuracy, with CLIME and ACLIME whose tuning parameters are chosen on
  validation data, in three designs.

## Summary

- **Exact, and fast.** `slassoinv` follows the lasso path of each column by
  the homotopy method until the penalty `λ₀σ̂` of the scaled lasso, which it
  finds exactly: on each segment of the path the residual variance is a
  quadratic in the penalty. On one CPU it took 0.85–1.7 s for p = 4000, and 8
  threads were 6.0–7.3 times faster. It was 13–112 times faster than the scaled
  lasso of CoordinateDescent column by column, with tight tolerances, and
  1,500–45,000 times faster than scalreg, which computes the whole lars path of
  every column (701 s for p = 500, n = 1000, against 0.026 s). Their estimates
  agree with those of `slassoinv` to 2e-6 (CoordinateDescent) and 2e-4
  (scalreg, whose noise level is iterated to an absolute tolerance of 1e-4).
- **It reproduces Table 1 of Sun and Zhang (2013).** The mean spectral errors
  of the scaled lasso are within 0.02 of those of the paper in 15 of the 18
  cells (three models, p from 30 to 1000), and within 0.08 in all; those of its
  least squares refit match the paper as closely, except for the random graph
  (model 2) with p ≤ 150, where they are up to 0.17 smaller.
- **Accurate without tuning.** CLIME and ACLIME choose their tuning parameter by
  the likelihood of a validation sample as large as the training sample, which
  the scaled lasso does not use. Even so, in spectral norm the scaled lasso was
  - in the design of Sun and Zhang, 3–20% more accurate than the better of the
    two for p ≥ 60, and 8–10% less accurate for p = 30 in models 1 and 3, as Sun
    and Zhang also found;
  - in the design of Cai, Liu and Zhou, 20–37% more accurate for the
    Erdős–Rényi model, and 21% and 5% for the band and AR(1) models with
    p = 200; within 3% with p = 100, and 13–15% less accurate with p = 50.
- **The least squares refit** (`refit=true`) was the most accurate estimator in
  the sparse models (models 1 and 3 of Sun and Zhang, the AR(1) model of Cai,
  Liu and Zhou), 22–37% more accurate than the better of CLIME and ACLIME, but
  less accurate than the scaled lasso for the random graphs (model 2 with
  p ≤ 90, Erdős–Rényi).
- **Hadamard blocks**, where the matrix ℓ1 norm of Ω grows like `√k` while its
  spectral norm stays at 1.6, test the one place where the error bound of Sun
  and Zhang is better than that of ACLIME (the spectral norm of Ω in place of
  its matrix ℓ1 norm). The errors do not follow the bounds: from k = 2 to
  k = 32 the error of the scaled lasso grew 3.8 times, that of ACLIME 1.9 times,
  and the scaled lasso went from 30% more accurate than the better of CLIME and
  ACLIME to 36% less accurate. Its fixed penalty shrinks the k entries of each
  column, of size `ρ/√k`, and this bias adds up over the k entries, while
  ACLIME, whose δ is chosen on validation data, shrinks less as k grows. The
  least squares refit removes the bias: it was the most accurate estimator for
  k = 8 and 16, and as accurate as ACLIME for k = 32.
- **The scaled lasso as the first step of ACLIME.** Replacing the estimates of
  the diagonal of step 1 of ACLIME by the `1/σ̂ⱼ²` of the scaled lasso changed
  the spectral errors by -9% to +4% (smaller in 15 of the 24 cells), in line
  with the study of the first step in [`../clime`](../clime/README.md).

## Speed

USC CARC Discovery, AMD EPYC 7513, Julia 1.13.2 and R 4.5.3 with
single-threaded OpenBLAS; one CPU per run, with 8 runs at a time on a node
(which share its memory bandwidth: on an otherwise idle node `slassoinv` was
up to twice as fast). The data are n draws from the AR(1), band and Erdős–Rényi
precision matrices of [`../clime/problems.jl`](../clime/problems.jl), and
`S = X'X/n`; all methods use the penalty level `λ₀` of `slassoinv`.
CoordinateDescent ran up to p = 1000, scalreg up to p = 500.

Seconds for the whole estimate (in parentheses, relative to `slassoinv` on one thread):

| problem | slassoinv | refit | 8 threads | CoordinateDescent | scalreg | nonzero entries |
|---|---:|---:|---:|---:|---:|---:|
| AR(1), p = 100, n = 50 | 0.0007 | 0.0006 | — | 0.012 (18×) | 1.33 (2039×) | 462 |
| AR(1), p = 100, n = 200 | 0.0006 | 0.0006 | — | 0.015 (23×) | 2.42 (3745×) | 424 |
| AR(1), p = 200, n = 100 | 0.0027 | 0.0027 | — | 0.046 (17×) | 9.03 (3370×) | 898 |
| AR(1), p = 200, n = 400 | 0.0025 | 0.0025 | — | 0.113 (46×) | 18.98 (7725×) | 800 |
| AR(1), p = 500, n = 250 | 0.016 | 0.016 | — | 0.432 (27×) | 199 (12478×) | 2006 |
| AR(1), p = 500, n = 1000 | 0.014 | 0.014 | — | 1.06 (74×) | 651 (45267×) | 1918 |
| AR(1), p = 1000, n = 500 | 0.058 | 0.059 | 0.0096 (6.1×) | 1.94 (33×) | — | 3720 |
| AR(1), p = 1000, n = 2000 | 0.055 | 0.056 | 0.0085 (6.4×) | 6.08 (112×) | — | 3528 |
| AR(1), p = 2000, n = 1000 | 0.229 | 0.229 | 0.036 (6.4×) | — | — | 6864 |
| AR(1), p = 2000, n = 4000 | 0.216 | 0.210 | 0.034 (6.3×) | — | — | 6724 |
| AR(1), p = 4000, n = 2000 | 0.904 | 0.896 | 0.141 (6.4×) | — | — | 13126 |
| AR(1), p = 4000, n = 8000 | 0.852 | 0.852 | 0.142 (6.0×) | — | — | 12896 |
| band, p = 100, n = 50 | 0.0009 | 0.0009 | — | 0.016 (17×) | 1.50 (1603×) | 618 |
| band, p = 100, n = 200 | 0.0011 | 0.0011 | — | 0.019 (17×) | 2.65 (2353×) | 696 |
| band, p = 200, n = 100 | 0.0041 | 0.0040 | — | 0.065 (16×) | 9.90 (2422×) | 1440 |
| band, p = 200, n = 400 | 0.0042 | 0.0042 | — | 0.148 (35×) | 27.33 (6450×) | 1368 |
| band, p = 500, n = 250 | 0.027 | 0.027 | — | 0.509 (19×) | 214 (7901×) | 3600 |
| band, p = 500, n = 1000 | 0.026 | 0.026 | — | 1.18 (46×) | 701 (27109×) | 3370 |
| band, p = 1000, n = 500 | 0.110 | 0.111 | 0.018 (6.3×) | 2.40 (22×) | — | 7022 |
| band, p = 1000, n = 2000 | 0.103 | 0.103 | 0.015 (6.9×) | 7.74 (75×) | — | 6674 |
| band, p = 2000, n = 1000 | 0.415 | 0.418 | 0.058 (7.1×) | — | — | 13524 |
| band, p = 2000, n = 4000 | 0.400 | 0.406 | 0.058 (6.9×) | — | — | 13176 |
| band, p = 4000, n = 2000 | 1.71 | 1.71 | 0.239 (7.2×) | — | — | 26498 |
| band, p = 4000, n = 8000 | 1.65 | 1.66 | 0.244 (6.8×) | — | — | 26124 |
| Erdős–Rényi, p = 100, n = 50 | 0.0010 | 0.0010 | — | 0.014 (13×) | 1.55 (1502×) | 620 |
| Erdős–Rényi, p = 100, n = 200 | 0.0011 | 0.0011 | — | 0.016 (14×) | 2.60 (2265×) | 748 |
| Erdős–Rényi, p = 200, n = 100 | 0.0043 | 0.0043 | — | 0.055 (13×) | 9.32 (2160×) | 1422 |
| Erdős–Rényi, p = 200, n = 400 | 0.0042 | 0.0042 | — | 0.122 (29×) | 24.40 (5838×) | 1554 |
| Erdős–Rényi, p = 500, n = 250 | 0.025 | 0.025 | — | 0.416 (17×) | 189 (7477×) | 3772 |
| Erdős–Rényi, p = 500, n = 1000 | 0.024 | 0.024 | — | 1.04 (43×) | 548 (22869×) | 3710 |
| Erdős–Rényi, p = 1000, n = 500 | 0.096 | 0.099 | 0.013 (7.2×) | 2.08 (22×) | — | 7130 |
| Erdős–Rényi, p = 1000, n = 2000 | 0.096 | 0.095 | 0.014 (7.0×) | 6.81 (71×) | — | 6876 |
| Erdős–Rényi, p = 2000, n = 1000 | 0.363 | 0.367 | 0.055 (6.6×) | — | — | 13302 |
| Erdős–Rényi, p = 2000, n = 4000 | 0.363 | 0.361 | 0.050 (7.3×) | — | — | 13078 |
| Erdős–Rényi, p = 4000, n = 2000 | 1.54 | 1.54 | 0.217 (7.1×) | — | — | 25284 |
| Erdős–Rényi, p = 4000, n = 8000 | 1.46 | 1.50 | 0.213 (6.9×) | — | — | 23574 |

## Accuracy

Mean errors over the replications, with standard errors in parentheses.
SLasso is `slassoinv(S, n)`, SLasso/LSE `slassoinv(S, n; refit=true)`;
CLIME runs on the correlation matrix, its estimate scaled back, with its
penalty chosen from 45 values between 1 and 0.02; ACLIME chooses δ from
`(1:100)/50` for p ≤ 100 in the design of Cai, Liu and Zhou (as in the
replication) and from `(5:100)/50` otherwise; both by the likelihood of the
validation sample, and their times include the whole path of tuning
parameters. "ACLIME (scaled lasso pilot)" uses the diagonal `1/σ̂ⱼ²` of the
scaled lasso in place of step 1. CLIME and ACLIME ran up to p = 300.

### Design of Sun and Zhang (2013)

Their three models, with n = 100 observations and as many for validation: 1,
`Ωᵢⱼ = 0.6^|i-j|`; 2, a random graph with entries 0.5 with probability 0.1,
made to have condition number p and scaled to a unit diagonal; 3, model 1 with
an unequal diagonal. 100 replications (20 for p = 300); the last columns give
the means of Table 1 of the paper (100 replications, GLasso omitted).

Spectral norm:

| model | p | reps | SLasso | SLasso/LSE | CLIME | ACLIME | ACLIME (scaled lasso pilot) | paper: SLasso | paper: SLasso/LSE | paper: CLIME |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model 1 | 30 | 100 | 2.46 (0.01) | 1.77 (0.01) | 2.27 (0.02) | 2.28 (0.02) | 2.26 (0.02) | 2.46 | 1.77 | 2.29 |
| model 1 | 60 | 100 | 2.67 (0.00) | 2.03 (0.01) | 2.78 (0.01) | 2.75 (0.01) | 2.72 (0.01) | 2.68 | 2.04 | 2.68 |
| model 1 | 90 | 100 | 2.76 (0.00) | 2.11 (0.01) | 2.94 (0.01) | 2.92 (0.01) | 2.89 (0.01) | 2.75 | 2.09 | 2.87 |
| model 1 | 150 | 100 | 2.84 (0.00) | 2.19 (0.01) | 3.10 (0.00) | 3.08 (0.00) | 3.04 (0.00) | 2.84 | 2.18 | 3.05 |
| model 1 | 300 | 20 | 2.94 (0.01) | 2.27 (0.02) | 3.28 (0.01) | 3.24 (0.01) | 3.24 (0.00) | 2.93 | 2.25 | NA |
| model 1 | 1000 | 100 | 3.08 (0.00) | 2.38 (0.01) | — | — | — | 3.08 | 2.38 | NA |
| model 2 | 30 | 100 | 0.73 (0.01) | 1.01 (0.02) | 0.79 (0.02) | 0.79 (0.02) | 0.79 (0.01) | 0.72 | 1.08 | 0.81 |
| model 2 | 60 | 100 | 1.07 (0.01) | 1.39 (0.02) | 1.32 (0.02) | 1.25 (0.02) | 1.25 (0.01) | 1.06 | 1.41 | 1.19 |
| model 2 | 90 | 100 | 1.41 (0.01) | 1.57 (0.01) | 1.86 (0.01) | 1.77 (0.01) | 1.69 (0.01) | 1.48 | 1.73 | 1.61 |
| model 2 | 150 | 100 | 1.96 (0.00) | 1.87 (0.02) | 2.45 (0.01) | 2.40 (0.01) | 2.34 (0.01) | 1.96 | 2.04 | 2.06 |
| model 2 | 300 | 20 | 2.96 (0.01) | 2.37 (0.03) | 3.39 (0.01) | 3.37 (0.01) | 3.32 (0.01) | 2.88 | 2.34 | NA |
| model 2 | 1000 | 100 | 5.49 (0.00) | 4.90 (0.03) | — | — | — | 5.46 | 4.88 | NA |
| model 3 | 30 | 100 | 1.83 (0.01) | 1.27 (0.01) | 1.67 (0.02) | 1.69 (0.02) | 1.68 (0.02) | 1.84 | 1.28 | 1.63 |
| model 3 | 60 | 100 | 2.19 (0.01) | 1.59 (0.01) | 2.28 (0.01) | 2.31 (0.01) | 2.29 (0.01) | 2.18 | 1.58 | 2.10 |
| model 3 | 90 | 100 | 2.33 (0.01) | 1.70 (0.01) | 2.50 (0.01) | 2.54 (0.01) | 2.51 (0.01) | 2.34 | 1.71 | 2.38 |
| model 3 | 150 | 100 | 2.51 (0.00) | 1.85 (0.01) | 2.76 (0.01) | 2.80 (0.01) | 2.79 (0.00) | 2.51 | 1.84 | 2.76 |
| model 3 | 300 | 20 | 2.68 (0.01) | 1.98 (0.02) | 3.03 (0.01) | 3.10 (0.01) | 3.11 (0.00) | 2.70 | 1.99 | NA |
| model 3 | 1000 | 100 | 2.93 (0.00) | 2.14 (0.01) | — | — | — | 2.94 | 2.16 | NA |

Frobenius norm:

| model | p | reps | SLasso | SLasso/LSE | CLIME | ACLIME | ACLIME (scaled lasso pilot) | paper: SLasso | paper: SLasso/LSE | paper: CLIME |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model 1 | 30 | 100 | 4.20 (0.01) | 3.38 (0.02) | 3.79 (0.04) | 3.79 (0.04) | 3.75 (0.03) | 4.20 | 3.37 | 3.80 |
| model 1 | 60 | 100 | 6.40 (0.01) | 5.36 (0.01) | 6.61 (0.03) | 6.52 (0.03) | 6.43 (0.03) | 6.41 | 5.35 | 6.32 |
| model 1 | 90 | 100 | 8.10 (0.01) | 6.87 (0.01) | 8.63 (0.02) | 8.57 (0.02) | 8.45 (0.02) | 8.09 | 6.87 | 8.42 |
| model 1 | 150 | 100 | 10.79 (0.01) | 9.36 (0.02) | 11.89 (0.02) | 11.80 (0.02) | 11.65 (0.02) | 10.79 | 9.32 | 11.68 |
| model 1 | 300 | 20 | 15.81 (0.02) | 13.95 (0.03) | 18.18 (0.04) | 17.91 (0.03) | 17.93 (0.03) | 15.83 | 13.89 | NA |
| model 1 | 1000 | 100 | 30.53 (0.01) | 26.74 (0.02) | — | — | — | 30.55 | 26.68 | NA |
| model 2 | 30 | 100 | 1.76 (0.02) | 2.17 (0.02) | 1.84 (0.02) | 1.81 (0.02) | 1.84 (0.02) | 1.86 | 2.40 | 1.87 |
| model 2 | 60 | 100 | 3.21 (0.02) | 4.12 (0.03) | 3.54 (0.04) | 3.41 (0.03) | 3.32 (0.02) | 3.27 | 4.19 | 3.42 |
| model 2 | 90 | 100 | 4.38 (0.01) | 5.82 (0.02) | 5.32 (0.04) | 5.04 (0.04) | 4.72 (0.02) | 4.42 | 6.18 | 4.65 |
| model 2 | 150 | 100 | 6.24 (0.01) | 8.58 (0.03) | 7.78 (0.03) | 7.54 (0.03) | 7.19 (0.03) | 5.95 | 9.17 | 6.33 |
| model 2 | 300 | 20 | 9.60 (0.03) | 13.53 (0.07) | 12.08 (0.08) | 11.89 (0.07) | 11.46 (0.06) | 9.26 | 13.99 | NA |
| model 2 | 1000 | 100 | 19.08 (0.02) | 26.72 (0.04) | — | — | — | 18.85 | 26.08 | NA |
| model 3 | 30 | 100 | 2.69 (0.01) | 2.15 (0.01) | 2.40 (0.02) | 2.42 (0.02) | 2.39 (0.02) | 2.69 | 2.15 | 2.37 |
| model 3 | 60 | 100 | 4.12 (0.01) | 3.41 (0.01) | 4.24 (0.02) | 4.26 (0.02) | 4.21 (0.02) | 4.10 | 3.41 | 3.98 |
| model 3 | 90 | 100 | 5.19 (0.01) | 4.40 (0.01) | 5.53 (0.01) | 5.58 (0.01) | 5.50 (0.01) | 5.19 | 4.40 | 5.37 |
| model 3 | 150 | 100 | 6.93 (0.01) | 5.95 (0.01) | 7.64 (0.01) | 7.68 (0.01) | 7.64 (0.01) | 6.93 | 5.96 | 7.75 |
| model 3 | 300 | 20 | 10.16 (0.01) | 8.92 (0.02) | 11.66 (0.03) | 11.82 (0.02) | 11.85 (0.02) | 10.18 | 8.89 | NA |
| model 3 | 1000 | 100 | 19.63 (0.01) | 17.04 (0.01) | — | — | — | 19.63 | 17.04 | NA |

### Design of Cai, Liu and Zhou (2016)

The design of the replication in [`../clime`](../clime/README.md), with the
same random draws: n = 200 observations and as many for validation, 50
replications (20 for p = 200). The column ACLIME equals the column ACLIME,
`truncate=false` of the replication there.

Spectral norm:

| model | p | reps | SLasso | SLasso/LSE | CLIME | ACLIME | ACLIME (scaled lasso pilot) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR(1) | 50 | 50 | 0.57 (0.02) | 0.37 (0.01) | 0.50 (0.02) | 0.50 (0.02) | 0.52 (0.02) |
| AR(1) | 100 | 50 | 0.70 (0.02) | 0.47 (0.01) | 0.68 (0.02) | 0.69 (0.02) | 0.70 (0.02) |
| AR(1) | 200 | 20 | 0.84 (0.02) | 0.56 (0.01) | 0.88 (0.02) | 0.96 (0.03) | 0.96 (0.03) |
| band | 50 | 50 | 0.27 (0.01) | 0.25 (0.01) | 0.24 (0.01) | 0.24 (0.01) | 0.23 (0.01) |
| band | 100 | 50 | 0.34 (0.01) | 0.34 (0.01) | 0.36 (0.01) | 0.33 (0.01) | 0.35 (0.01) |
| band | 200 | 20 | 0.42 (0.01) | 0.42 (0.02) | 0.53 (0.02) | 0.59 (0.02) | 0.60 (0.02) |
| Erdős–Rényi | 50 | 50 | 0.35 (0.02) | 0.46 (0.03) | 0.57 (0.04) | 0.62 (0.04) | 0.57 (0.03) |
| Erdős–Rényi | 100 | 50 | 0.66 (0.03) | 0.91 (0.04) | 0.84 (0.03) | 0.82 (0.03) | 0.85 (0.03) |
| Erdős–Rényi | 200 | 20 | 1.27 (0.04) | 1.72 (0.08) | 1.87 (0.05) | 1.80 (0.05) | 1.81 (0.06) |

Frobenius norm:

| model | p | reps | SLasso | SLasso/LSE | CLIME | ACLIME | ACLIME (scaled lasso pilot) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR(1) | 50 | 50 | 1.01 (0.02) | 0.73 (0.02) | 0.85 (0.02) | 0.85 (0.02) | 0.87 (0.02) |
| AR(1) | 100 | 50 | 1.60 (0.03) | 1.19 (0.02) | 1.52 (0.03) | 1.53 (0.03) | 1.55 (0.03) |
| AR(1) | 200 | 20 | 2.32 (0.03) | 1.76 (0.02) | 2.38 (0.03) | 2.53 (0.03) | 2.55 (0.03) |
| band | 50 | 50 | 0.54 (0.01) | 0.43 (0.02) | 0.46 (0.01) | 0.45 (0.01) | 0.44 (0.01) |
| band | 100 | 50 | 0.83 (0.01) | 0.69 (0.02) | 0.83 (0.02) | 0.80 (0.02) | 0.81 (0.02) |
| band | 200 | 20 | 1.32 (0.03) | 1.13 (0.03) | 1.65 (0.04) | 1.79 (0.05) | 1.79 (0.05) |
| Erdős–Rényi | 50 | 50 | 0.68 (0.03) | 0.83 (0.03) | 1.05 (0.06) | 1.17 (0.06) | 1.09 (0.05) |
| Erdős–Rényi | 100 | 50 | 1.73 (0.04) | 2.12 (0.06) | 2.07 (0.06) | 2.03 (0.05) | 2.08 (0.05) |
| Erdős–Rényi | 200 | 20 | 4.38 (0.10) | 5.34 (0.14) | 6.20 (0.13) | 5.77 (0.12) | 5.31 (0.11) |

### Hadamard blocks

`Ω` has blocks `I + (ρ/√k) H` of size k on its diagonal, with `H` the Sylvester
Hadamard matrix of order k and ρ = 0.6: its eigenvalues are 0.4 and 1.6 for
every k, while its matrix ℓ1 norm, `1 + ρ√k`, grows from 1.8 for k = 2 to 4.4
for k = 32. p = 256, n = 2000 observations and as many for
validation, 20 replications.

Spectral norm:

| model | p | reps | SLasso | SLasso/LSE | CLIME | ACLIME |
|---|---:|---:|---:|---:|---:|---:|
| k = 2 | 256 | 20 | 0.16 (0.00) | 0.27 (0.01) | 0.23 (0.00) | 0.23 (0.00) |
| k = 4 | 256 | 20 | 0.24 (0.00) | 0.25 (0.00) | 0.28 (0.00) | 0.28 (0.00) |
| k = 8 | 256 | 20 | 0.34 (0.00) | 0.26 (0.00) | 0.33 (0.00) | 0.33 (0.00) |
| k = 16 | 256 | 20 | 0.45 (0.00) | 0.32 (0.01) | 0.38 (0.00) | 0.38 (0.00) |
| k = 32 | 256 | 20 | 0.60 (0.00) | 0.43 (0.01) | 0.45 (0.00) | 0.44 (0.00) |

Frobenius norm:

| model | p | reps | SLasso | SLasso/LSE | CLIME | ACLIME |
|---|---:|---:|---:|---:|---:|---:|
| k = 2 | 256 | 20 | 0.95 (0.01) | 1.41 (0.01) | 1.41 (0.01) | 1.41 (0.01) |
| k = 4 | 256 | 20 | 1.54 (0.01) | 1.37 (0.01) | 1.80 (0.01) | 1.77 (0.01) |
| k = 8 | 256 | 20 | 2.28 (0.01) | 1.54 (0.01) | 2.12 (0.01) | 2.14 (0.01) |
| k = 16 | 256 | 20 | 3.41 (0.01) | 1.87 (0.01) | 2.53 (0.01) | 2.56 (0.01) |
| k = 32 | 256 | 20 | 5.15 (0.01) | 2.49 (0.01) | 3.12 (0.01) | 3.09 (0.01) |

Mean tuning parameters (λ₀ of the scaled lasso, λ of CLIME on the correlation
matrix, δ of ACLIME) and median seconds per replication, for the Hadamard
blocks; the other designs are in the output of `summarize.jl`:

| model | p | SLasso | SLasso/LSE | CLIME | ACLIME |
|---|---:|---:|---:|---:|---:|
| k = 2 | 256 | 0.052, 0.00236 s | 0.052, 0.00233 s | 0.041, 0.313 s | 0.756, 3.89 s |
| k = 4 | 256 | 0.052, 0.00313 s | 0.052, 0.00306 s | 0.037, 0.398 s | 0.666, 3.79 s |
| k = 8 | 256 | 0.052, 0.00489 s | 0.052, 0.00489 s | 0.031, 0.474 s | 0.590, 4.41 s |
| k = 16 | 256 | 0.052, 0.00908 s | 0.052, 0.00914 s | 0.026, 0.53 s | 0.511, 4.88 s |
| k = 32 | 256 | 0.052, 0.0216 s | 0.052, 0.0216 s | 0.022, 0.657 s | 0.420, 5.32 s |

## Running

The campaign ran on Discovery through [`campaign.sbatch`](campaign.sbatch):
[`accuracy.jl`](accuracy.jl) for a range of replications of one design, model
and dimension, [`speed.jl`](speed.jl) for the Julia timings, and
[`bench_r.R`](bench_r.R) for scalreg (1.0.1, archived on CRAN; it needs lars),
under GNU Parallel with one CPU per task, then the threaded timings;
[`summarize.jl`](summarize.jl) makes the tables, and
[`verify.jl`](verify.jl) checks that a version of CovarianceSelection reproduces the
estimates stored by another run. Run `20261023T0951Z-covsel-scaledlasso`, at
commit `95a36d0`; the accuracy of the design of Sun and Zhang with p ≤ 90 and
p = 1000, of the design of Cai, Liu and Zhou with p ≤ 100 and of the Hadamard
blocks comes from a first job at `df4dfd4`, whose `slassoinv` gives the
same estimates to the last bit (`verify.jl`). Later commits take the normal
quantile of the default `λ₀` from SpecialFunctions, which changes `λ₀` by less
than 1e-12. On a laptop:

```sh
julia --project=benchmark/slassoinv -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/slassoinv benchmark/slassoinv/accuracy.jl sz 1 60 1 10 /tmp/sz_1_p60.csv
julia --project=benchmark/slassoinv benchmark/slassoinv/speed.jl ar1 500 1000 /tmp/speed
julia benchmark/slassoinv/summarize.jl /tmp            # with /tmp/accuracy and /tmp/speed
```
