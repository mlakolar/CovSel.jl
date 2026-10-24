# Validation and findings

The module is implemented and tested. Coverage selection and paper-scale
validation remain an active numerical investigation. The large `paper.toml`
suite is prepared but has not been submitted: the exploratory results do not
yet justify declaring every inference variant satisfactory.

## Software validation

All numerical checks ran in Discovery Slurm allocations.

* Job `13382591`: 53 initial focused tests passed.
* Job `13382607`: the two-replication end-to-end smoke test completed; 496
  output rows, one expected shard and no duplicate result identities.
* Job `13382675`: all 1379 then-current package tests passed after restoring
  an extension directory omitted from the initial staging snapshot. The later
  generator check exposed a test-fixture dictionary type error. Both failed
  validation attempts and their causes remain recorded.
* Job `13382690`: 54 focused tests and 68 independent generator checks passed
  on Julia 1.13.2 and Julia 1.10.12.
* Job `13382773`: all 1388 final package tests passed on Julia 1.13.2,
  followed by 68 independent generator checks. On Julia 1.10.12, all 62 focused
  inference tests and 68 generator checks passed. This validates the final
  support-refit extension and fixed-window tuning implementation.

The tests include polynomial reproduction, an independent direct HC0
residual-product calculation, diagonal inference, all sparse pilots, one-step
algebra, exact reproduction of observation-multiplier critical values, support
refits, input failures, tuning and norm definitions. Generator checks include
analytic inverse identities, structural residual moments, deterministic seeds,
and the Gaussian/bump score identities. They do not prove coverage.

## Completed scout

Job `13382679` completed all 18 shards and 105000 output rows. Its six scenarios
use 30 replications each and common datasets across methods and tuning choices.
[The full scout table](../../experiments/single_index/results/scout/REPORT.md)
retains every method, including failures and adverse results. The reporting job
`13382691` produced summary tables, paired coverage comparisons and PDF figures.

The scout indicates that the default ACLIME inverse is not automatically a good
pilot for direct one-step inference. In the final-bandwidth scenario, with
nominal 95% joint coverage over three entries and 11 grid points, its observed
coverage was 43.3%, 16.7% and 6.7% as the final-bandwidth factor increased from
0.65 to 1 to 1.4. The matching ACLIME block construction covered 76.7% in these
three cases. These are exploratory 30-replication results, not final estimates.

For the penalty scenario, reducing the multiplier from 1.2 to 0.35 improved
ACLIME-block family coverage from 46.7% to 100%; scaled-lasso block coverage
increased from 80.0% to 96.7%. Widths also increased. The intervals and Wilson
bounds in the saved summaries must accompany any interpretation of these
small-sample Monte Carlo results. No optimal multiplier is inferred from them.

The poor direct ACLIME behavior motivates support refits and a closer check of
pilot scale. Its population one-step diagonal bias is quadratic and negative;
see [METHODS.md](METHODS.md). Multiplier resampling does not remove that bias.

## Completed larger studies

* Development: jobs `13382620` and recovery `13382838`; all 110 shards,
  693200 rows, 100–200 replications per scenario.
* Independent baseline confirmation: jobs `13382659` and recovery `13382839`;
  all 240 shards, 582400 rows, 200 replications per scenario.
* Refinement: jobs `13382776` and recovery `13382840`; all 140 shards,
  581200 rows, 200–300 replications per scenario.
* Assessment: job `13382956`; regenerated summaries and figures from saved
  outputs, checked nested-grid critical-value monotonicity, and wrote the
  [full assessment](../../experiments/single_index/results/assessment/ASSESSMENT.md).

All recovery jobs completed successfully. Final shards were reused, interrupted
partial files and original logs were preserved, and only missing shards were
rerun after the worker-memory scheduling repair. Aggregation checked all
expected shards and duplicate identities. The final reporter also checks the
expected pointwise-row count and restores failure-inclusive norm denominators
for early worker formats. No independent refitting audit is claimed.

The independent 200-replication block baseline at sample size 600 gave joint
coverage 93.0% for square-root-lasso block inversion, 92.0% for scaled-lasso
block inversion, 77.5% for ACLIME block inversion, and 31.5% for the direct
ACLIME one-step correction. The oracle covered 94.0%. Thus a default inverse
pilot is not automatically suitable for debiasing.

In the 300-replication support-refit baseline, scaled-lasso-refit block and
square-root-lasso-refit block both covered 95.0%; ACLIME-refit block covered
94.3%. With varying nuisance coefficients, these values were 94.0%, 93.0%,
and 90.7%, respectively. Refitting is promising but is not uniformly beneficial:
scaled-lasso-refit one-step family coverage was 83.0% and 82.0% in the two cases.

The corrected fixed-window CV selector gave 94.0% family coverage in 200 fresh
replications. At zero its diagonal, crossing-edge and null-edge pointwise
coverages were 94.5%, 93.0% and 92.0%. It is a prediction-based heuristic, not a
coverage-optimal rule. The high-dimensional fixed-baseline square-root-lasso
case covered only 88.5% jointly. These limitations prevent a general tuning
recommendation or a claim that the full suite is ready for an AoS paper.

Nested 21/41/81-point grids shared each dataset and multiplier stream. Across
200 replications there were no failed triples or critical-value monotonicity
violations. Increasing 41 to 81 points changed mean critical values by about
0.003 for the oracle/square-root-lasso methods and 0.012 for scaled-lasso-refit
block inversion. Family coverage changed by 0 to -0.5 percentage points.
This supports numerical grid stability in these cases, not continuum validity.

Sparse pilots and full one-step matrices serve different purposes. For example,
in the high-dimensional baseline the mean spectral error was 0.573 for the
scaled-lasso pilot and 4.433 for its full one-step matrix. Norm tables, lengths,
Wilson intervals and adverse findings are retained in the full assessment.

## Paper-focused confirmation and remaining limits

Job `13383011` runs 1000 independent replications per scenario with 1999
multipliers and 21 grid points. It tests wider pilots, the two prespecified
penalty multipliers 0.35 and 0.7, promising support-refit block constructions,
and the corrected CV selector in high dimension and with a generated sparse
index. Its seed namespace is independent of all development and refinement
studies. It is submitted evidence until outputs pass their completion checks.

The broader `paper.toml` plan remains prepared and unsubmitted. The initial CV
runs changed the validation window with the candidate bandwidth; they do not
validate the corrected selector. No general feasible continuum-band or
diagonal-coverage theorem is claimed.

Run records are in the SingleIndexGraph project's `.slurm/runs/`. Source
snapshots, locked environments, raw compact shards, job logs and reporting
inputs are under `/project2/mkolar_1314/SingleIndexGraph/`. No numerical work
has been run locally. GitHub CI is skipped for this task's commits so that
numerical validation remains on Discovery, as requested.
