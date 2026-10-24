# Reporting inputs

These tables and selected figures were produced in Discovery Slurm jobs.
CSV line endings are normalized to LF for version control; numerical values
are unchanged. Raw replication shards, source snapshots, original job logs,
interrupted partial files and the reconstruction recipes remain in
`/project2/mkolar_1314/SingleIndexGraph/`.

* `paper_confirmation/PAPER_CONFIRMATION.md`: independent 1000-replication
  confirmation, with pointwise, curvewise and family coverage and selection
  frequencies.
* `assessment/ASSESSMENT.md`: combined development, confirmation and refinement
  assessment, including adverse findings and nested-grid checks.
* `development/`, `confirmation/`, `refinement/`: per-group summary, paired
  method coverage differences, selected tuning values and completion receipts.
* `scout/`: the earlier 30-replication exploration, kept separately.
* `environment/`: the Julia dependency lock and Python plotting versions.

A row of `summary.csv` summarizes replications of one fixed scenario, method,
index mode, tuning setting, inferential target and metric. The `replications`
column is the denominator. Failures count as noncoverage. Conditional coverage,
length and bias use valid fits and are labeled accordingly. Norm failures also
retain the full intended denominator. `paired.csv` compares methods on the
same simulated datasets. `selected_tuning.csv` records the actual chosen
bandwidths and penalties; their frequencies are not optimality claims.

Finite-grid family coverage concerns only the three prespecified entries and
the requested index grid. Numerical grid stability does not establish a
continuum-coverage theorem. See `docs/single_index/RESULTS.md` for interpretation
and the distinction between completed and prepared studies.

For CV scenarios, nominal tuning-factor columns label the requested design.
Actual selected bandwidths and penalties vary across replications and are
recorded in `selected_tuning.csv`.
