# Frozen first-stage confirmation

The 751000000 namespace independently tests the first-stage candidate class.
Its configuration is frozen before any outcome in that namespace. No coverage
result selects a different rule within a scenario.

Compare known indices, the original five-block MOM index with multiplier 0.5
and soft thresholding, and the following alignment-CV index:

* three folds, common validation observations and training-anchor orientation;
* threshold multipliers `[0.25,0.5,0.75,1,1.25]`;
* block counts `[1,5]` and unshrinking flags `[false,true]`;
* maximum held-out alignment, with the documented tie rule;
* a final full-data index fit with the selected settings.

All index modes use the same frozen second-stage guarded CV and the same
prespecified precision entries. Compare the feasible corrected log-diagonal
procedure and the oracle nuisance benchmark. Use 1000 replications, 1999
multipliers, and 21 grid points in each of four scenarios: strong signal with
100 or 200 covariates, weak signal with n=1200, and weak signal with n=2400.

Report direction error, selected index settings, support size, failure-inclusive
pointwise and joint coverage, widths, and matched-index-mode differences.
Improved alignment alone is not evidence of valid precision inference.
