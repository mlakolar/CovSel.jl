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

## Larger studies and limits

* `13382838` (recovery of `13382620`): development, 100–200 replications per scenario; final bandwidth,
  pilot bandwidth, penalty, varying nuisances and generated-index sensitivities.
* `13382839` (recovery of `13382659`): independent fixed-baseline confirmation, 200 replications per
  scenario; sample sizes, dimensions, block/chain/grid graphs, heavy tails,
  bump covariates, sparse generated indices and initial CV tuning.
* `13382840` (recovery of `13382776`): refinement, 200–300 replications per scenario; support refits,
  corrected fixed-window CV and nested 21/41/81-point grids with shared data
  and multiplier streams. It starts only after final software validation.

The original allocations were cancelled to correct inherited per-step memory,
which had serialized the workers. Completed shards, interrupted partial files,
and original logs were preserved. The recovery jobs reuse the scientific source
and seeds and run only missing final shards.

These are submitted studies. Their completed-output validation and scientific
findings will be recorded separately from submission and scheduler status.
The initial CV runs changed the validation window with the candidate bandwidth.
They are not validation of the corrected selector in the final module.
No general feasible continuum-band or diagonal-coverage theorem is claimed.

Run records are in the SingleIndexGraph project's `.slurm/runs/`. Source
snapshots, locked environments, raw compact shards, job logs and reporting
inputs are under `/project2/mkolar_1314/SingleIndexGraph/`. No numerical work
has been run locally. GitHub CI is skipped for this task's commits so that
numerical validation remains on Discovery, as requested.
