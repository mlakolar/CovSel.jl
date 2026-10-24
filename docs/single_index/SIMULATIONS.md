# Discovery simulation campaign

All numerical tests, simulation fitting, bootstrap sampling, aggregation, and
figure generation run in Slurm allocations on Discovery. Local steps only edit
source, generate configuration text, and transfer/inspect outputs. The user
explicitly authorized the resources required for these experiments.

## Scientific design

The block design follows the active target/nuisance/probe construction in the
SingleIndexGraph manuscript's `A_6_simulation_protocol.tex`, with independent
standard Gaussian covariates as the default score-known design. The protocol's
bounded bump covariates are also implemented, with its stated variance scaling,
score, and rejection sampler. This Gaussian default is a documented departure
from that protocol, not an exact reproduction of its entire simulation suite.

The target pair shares sparse nuisance variables. Its structural residual
variances are 1-0.35^2 and its residual covariance is 0.20*tanh(2*curvature*t).
A separate last-coordinate probe pair has correlation signal*tanh(2*t). The
varying-nuisance scenario multiplies the structural coefficients by
1+0.5*tanh(t), keeping the residual target separate from nuisance curvature.
Analytic covariance is constructed from these structural equations.

Chain and grid graphs provide sparse alternatives inspired by ROCKET's
simulation comparisons. The chain is a stationary AR correlation model with
index-dependent correlation. Grid precision matrices are strictly diagonally
dominant. Covariance-standardized t8 and scale-mixture noise preserve the
conditional covariance, so the same analytic precision is the truth. A t1
example would have no covariance and is deliberately excluded from this target.
We do not use arbitrary marginal transformations, which would change it.

Each dataset is shared across all methods, both index modes and tuning values
within a scenario. Stable integer seeds are `seed_base + 100000*scenario_id +
replication`. Multiplier seeds and tuning-fold seeds have separate fixed
offsets. The confirmation and paper plans have separate seed bases from the
development plan. The same selected entries include one diagonal, the active
pair, and a nonedge. More entries can be added without fitting all entries.

## Stages and planned replication counts

| Plan | Purpose | Replications per scenario | Multipliers |
|---|---|---:|---:|
| `smoke.toml` | End-to-end output canary | 2 | 199 |
| `development.toml` | Final/pilot bandwidth, penalty, varying nuisances, index error, data-driven tuning | 100–200 | 399 |
| `confirmation.toml` | Independent distribution, dimension, graph and tuning checks | 200 | 399 |
| `refinement.toml` | Support refits, fixed-window CV and nested 21/41/81-point grids | 200–300 | 399–999 |
| `paper.toml` | Prespecified larger evidence after development is satisfactory | 500–2000 | 999–1999 |

A prepared plan is not evidence that it was submitted or completed. See
[RESULTS.md](RESULTS.md) and the registry for actual job IDs and output status.
At 95% coverage, 1000 replications give a Monte Carlo standard error of about
0.7 percentage points. Use paired errors for method comparisons. With only
200 replications, differences of a few percentage points are unresolved.

The paper plan covers increasing dimensions and sample sizes, several graph
structures, varying nuisance functions, generated sparse indices, heavy tails,
contamination, and support boundaries. It is a candidate simulation suite, not
a claim of AoS readiness. Before interpreting it as paper evidence, add/check:

* calibrated tuning frozen before independent confirmation;
* adequate coverage for both diagonal and off-diagonal targets, including nulls;
* failure-inclusive curvewise and familywise coverage and informative lengths;
* dense-grid refinement and explicit limits of continuum interpretation;
* larger local dimension relative to sample size and first-stage weak-signal
  failures, not only easy observed-index cases;
* runtime, condition numbers and numerical failure distributions;
* comparisons of oracle versus feasible estimates on the leading scale;
* density variation and boundary sensitivity;
* a scientific application and its preprocessing, when data are available.

## Execution and provenance

Discovery account: `mkolar_1314`; partition: `main`; CPU constraint:
`epyc-7513`. Julia: `/home1/mkolar/juliaPkg/julia-1.13.2/bin/julia`.
Task-owned directories are under `/project2/mkolar_1314/SingleIndexGraph/`.
The private SSH helper session is closed at the end of the interactive task;
Slurm jobs do not rely on it. Project storage is used for source snapshots,
locked Julia environment, raw compact shards, job logs and summaries. Scratch
is not treated as permanent storage.

`canary.sbatch` resolves the required unregistered solver dependencies and runs
the numerical tests. The saved `Manifest.toml` pins the resolved environment.
`campaign.sbatch` runs GNU Parallel inside one allocation and launches exclusive,
exact, one-CPU `srun` steps. BLAS and OpenMP threads are one. A shard contains
several replications in one Julia process; there is no scheduler job per seed.
The launcher stops scheduling after a worker failure and retains partial
outputs and the GNU Parallel job log. It does not automatically resubmit or
reuse possibly invalid shards.

```bash
# From a staged, versioned source directory on Discovery:
sbatch --parsable --job-name=sig-<unique-run-tag> \
  experiments/single_index/campaign.sbatch \
  experiments/single_index/development.toml results
```

The default request is 32 CPUs, 64 GB, four hours. Command-line sbatch options
can reduce it. Actual submitted requests and IDs belong in the registry.
Resolve scheduler uncertainty before retrying any ambiguous submission.
For partial recovery, use the Parallel log and final shard filenames to create
an explicit missing-task list; do not overwrite valid final shards.

## Outputs and validation

Each shard writes a `.partial` file and renames it only after success. Its CSV
contains scenario/replication/seed, method/index mode, tuning settings, each
pointwise result, curvewise and joint-family coverage, lengths and failure
messages, plus full-matrix norm errors at zero. The requested finite grid and
all three entries enter each family maximum. Failures count as noncoverage.
Conditional-on-success coverage is separately labeled.

The aggregator checks expected shard count and duplicate result identities. It
reports Wilson intervals and Monte Carlo standard errors, pointwise bias,
empirical standard deviation, root-mean squared estimated standard error,
length, all requested matrix norms, and paired coverage differences. These
are output-consistency checks, not an independent scientific refit.

Matrix errors are entrywise maximum, entrywise l1 sum, Frobenius, spectral,
induced l1, induced infinity, and relative Frobenius. Norms are reported for
both sparse pilots and one-step matrices. These should not be confused with
errors of a matrix assembled from independently fitted pairwise blocks.

## Refinement after the scout

The 30-replication scout showed severe undercoverage for the default ACLIME
one-step correction. Its results remain in the reporting inputs. This motivated
support-refit comparisons on fresh seeds, rather than dropping the unfavorable
method. The larger baseline development and confirmation studies retain their
original frozen source.

Review also found that the initial CV implementation changed its validation
window with each bandwidth candidate. The final module instead compares
candidates on a common validation window. The `fixed_window_cv` refinement
scenario tests that corrected selector on fresh data. Earlier `data_driven`
scenarios describe the initial selector and must not be cited as validation of
the corrected implementation.

Nested-grid refinement shares each simulated dataset and multiplier stream
across 21, 41 and 81 grid points, using an explicit common `seed_group`. It
assesses numerical stability of critical values and coverage. It does not
supply a grid-to-continuum theorem.
