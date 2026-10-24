# First-stage index tuning

The second-stage guarded CV and diagonal correction are fixed. Index modes compare known indices, the original multiplier 0.5 with soft thresholding, and held-out Stein-alignment CV with optional support unshrinking. No true index or coverage enters tuning. Failures count as noncoverage.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | estimated | 300 | 0 | 0.900 | [0.861, 0.929] | 0.626 |
| cv_larger_index | oracle_hc3_log | estimated_cv | 300 | 0 | 0.897 | [0.857, 0.926] | 0.620 |
| cv_larger_index | oracle_hc3_log | known | 300 | 0 | 0.927 | [0.891, 0.951] | 0.621 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.897 | [0.857, 0.926] | 0.627 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 0 | 0.887 | [0.846, 0.918] | 0.619 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.937 | [0.903, 0.959] | 0.620 |
| cv_strong_index | oracle_hc3_log | estimated | 300 | 0 | 0.933 | [0.899, 0.956] | 0.624 |
| cv_strong_index | oracle_hc3_log | estimated_cv | 300 | 0 | 0.960 | [0.931, 0.977] | 0.621 |
| cv_strong_index | oracle_hc3_log | known | 300 | 0 | 0.960 | [0.931, 0.977] | 0.620 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.943 | [0.911, 0.964] | 0.625 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 0 | 0.947 | [0.915, 0.967] | 0.621 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.953 | [0.923, 0.972] | 0.619 |
| cv_weak_index | oracle_hc3_log | estimated | 300 | 0 | 0.807 | [0.758, 0.847] | 0.618 |
| cv_weak_index | oracle_hc3_log | estimated_cv | 300 | 3 | 0.757 | [0.705, 0.802] | 0.616 |
| cv_weak_index | oracle_hc3_log | known | 300 | 0 | 0.947 | [0.915, 0.967] | 0.622 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.790 | [0.740, 0.832] | 0.621 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 3 | 0.763 | [0.712, 0.808] | 0.619 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.953 | [0.923, 0.972] | 0.621 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | 300 | 0 | 0.827 | [0.780, 0.865] | 0.466 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | 300 | 0 | 0.857 | [0.812, 0.892] | 0.466 |
| cv_weak_index_larger_n | oracle_hc3_log | known | 300 | 0 | 0.947 | [0.915, 0.967] | 0.470 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.830 | [0.783, 0.868] | 0.467 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 0 | 0.847 | [0.802, 0.883] | 0.466 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.943 | [0.911, 0.964] | 0.469 |

## Pointwise intervals and index error

Index error means are conditional on a successful fit. Index failures remain in coverage denominators.

| Scenario | Method | Index | Entry | Coverage | Bias | SD | RMS SE | Mean direction error | Failures |
|---|---|---|---|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | estimated | (1,1) | 0.967 | 0.014 | 0.117 | 0.120 | 0.378 | 0 |
| cv_larger_index | oracle_hc3_log | estimated | (1,2) | 0.950 | 0.006 | 0.085 | 0.085 | 0.378 | 0 |
| cv_larger_index | oracle_hc3_log | estimated | (1,78) | 0.953 | -0.003 | 0.079 | 0.078 | 0.378 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,1) | 0.943 | 0.018 | 0.120 | 0.119 | 0.267 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,2) | 0.943 | 0.001 | 0.085 | 0.084 | 0.267 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,78) | 0.943 | -0.002 | 0.080 | 0.078 | 0.267 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,1) | 0.940 | 0.020 | 0.121 | 0.117 | 0.000 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,2) | 0.933 | 0.000 | 0.084 | 0.082 | 0.000 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,78) | 0.937 | 0.002 | 0.083 | 0.077 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.977 | 0.007 | 0.117 | 0.121 | 0.378 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.950 | 0.006 | 0.085 | 0.086 | 0.378 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.957 | -0.003 | 0.078 | 0.079 | 0.378 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.957 | 0.011 | 0.119 | 0.119 | 0.267 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.953 | 0.001 | 0.085 | 0.084 | 0.267 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.947 | -0.003 | 0.079 | 0.078 | 0.267 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.943 | 0.014 | 0.122 | 0.118 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.930 | -0.000 | 0.083 | 0.083 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.940 | 0.001 | 0.083 | 0.077 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,1) | 0.967 | 0.012 | 0.113 | 0.118 | 0.320 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,2) | 0.953 | 0.009 | 0.085 | 0.084 | 0.320 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,78) | 0.963 | 0.001 | 0.077 | 0.079 | 0.320 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,1) | 0.950 | 0.009 | 0.118 | 0.115 | 0.248 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,2) | 0.950 | 0.011 | 0.082 | 0.083 | 0.248 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,78) | 0.967 | 0.007 | 0.075 | 0.077 | 0.248 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,1) | 0.963 | 0.008 | 0.109 | 0.115 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,2) | 0.963 | 0.014 | 0.077 | 0.081 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,78) | 0.957 | 0.011 | 0.074 | 0.077 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.967 | 0.005 | 0.112 | 0.118 | 0.320 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.960 | 0.010 | 0.084 | 0.085 | 0.320 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.967 | 0.001 | 0.075 | 0.079 | 0.320 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.953 | 0.003 | 0.118 | 0.116 | 0.248 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.950 | 0.012 | 0.081 | 0.084 | 0.248 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.970 | 0.007 | 0.074 | 0.078 | 0.248 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.957 | 0.003 | 0.110 | 0.115 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.960 | 0.014 | 0.076 | 0.082 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.953 | 0.010 | 0.073 | 0.077 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | estimated | (1,1) | 0.953 | 0.016 | 0.118 | 0.117 | 0.698 | 0 |
| cv_weak_index | oracle_hc3_log | estimated | (1,2) | 0.943 | 0.003 | 0.088 | 0.086 | 0.698 | 0 |
| cv_weak_index | oracle_hc3_log | estimated | (1,78) | 0.957 | 0.000 | 0.074 | 0.078 | 0.698 | 0 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,1) | 0.937 | 0.018 | 0.117 | 0.118 | 0.720 | 3 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,2) | 0.923 | 0.002 | 0.092 | 0.085 | 0.720 | 3 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,78) | 0.950 | 0.001 | 0.074 | 0.078 | 0.720 | 3 |
| cv_weak_index | oracle_hc3_log | known | (1,1) | 0.963 | 0.014 | 0.116 | 0.115 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,2) | 0.950 | -0.007 | 0.088 | 0.082 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,78) | 0.940 | 0.000 | 0.081 | 0.077 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.957 | 0.010 | 0.119 | 0.118 | 0.698 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.943 | 0.003 | 0.087 | 0.086 | 0.698 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.963 | 0.002 | 0.074 | 0.078 | 0.698 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.937 | 0.012 | 0.117 | 0.118 | 0.720 | 3 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.927 | 0.004 | 0.090 | 0.085 | 0.720 | 3 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.950 | 0.002 | 0.073 | 0.078 | 0.720 | 3 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.960 | 0.009 | 0.116 | 0.116 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.953 | -0.006 | 0.087 | 0.082 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.940 | 0.001 | 0.080 | 0.078 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,1) | 0.930 | -0.006 | 0.089 | 0.086 | 0.477 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,2) | 0.963 | 0.003 | 0.062 | 0.063 | 0.477 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,78) | 0.967 | 0.001 | 0.056 | 0.058 | 0.477 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,1) | 0.943 | -0.002 | 0.083 | 0.086 | 0.455 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,2) | 0.970 | -0.000 | 0.061 | 0.062 | 0.455 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,78) | 0.967 | 0.001 | 0.058 | 0.058 | 0.455 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,1) | 0.940 | -0.001 | 0.087 | 0.086 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,2) | 0.947 | -0.002 | 0.061 | 0.061 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,78) | 0.933 | 0.004 | 0.058 | 0.057 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.930 | -0.009 | 0.088 | 0.086 | 0.477 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.963 | 0.003 | 0.062 | 0.063 | 0.477 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.963 | 0.001 | 0.055 | 0.058 | 0.477 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.947 | -0.005 | 0.083 | 0.086 | 0.455 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.967 | -0.001 | 0.061 | 0.063 | 0.455 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.973 | 0.002 | 0.057 | 0.058 | 0.455 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.943 | -0.004 | 0.087 | 0.086 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.947 | -0.002 | 0.060 | 0.062 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.940 | 0.004 | 0.058 | 0.057 | 0.000 | 0 |

## Observed first-stage selections

| Scenario | Multiplier | Unshrink | Support | Count | Fraction |
|---|---|---|---|---|---|
| cv_larger_index | 1.0 | true | 4 | 76 | 0.253 |
| cv_larger_index | 1.0 | true | 5 | 43 | 0.143 |
| cv_larger_index | 1.25 | true | 4 | 26 | 0.087 |
| cv_larger_index | 0.75 | false | 9 | 18 | 0.060 |
| cv_larger_index | 0.75 | false | 8 | 16 | 0.053 |
| cv_larger_index | 0.75 | false | 11 | 16 | 0.053 |
| cv_larger_index | 0.75 | false | 10 | 14 | 0.047 |
| cv_larger_index | 1.0 | true | 6 | 13 | 0.043 |
| cv_strong_index | 1.0 | true | 4 | 78 | 0.260 |
| cv_strong_index | 1.25 | true | 4 | 38 | 0.127 |
| cv_strong_index | 1.0 | true | 5 | 38 | 0.127 |
| cv_strong_index | 0.75 | false | 7 | 14 | 0.047 |
| cv_strong_index | 0.75 | false | 8 | 13 | 0.043 |
| cv_strong_index | 0.75 | false | 6 | 12 | 0.040 |
| cv_strong_index | 1.25 | true | 3 | 12 | 0.040 |
| cv_strong_index | 1.0 | true | 6 | 11 | 0.037 |
| cv_weak_index | 1.0 | false | 2 | 24 | 0.080 |
| cv_weak_index | 0.75 | false | 6 | 20 | 0.067 |
| cv_weak_index | 1.0 | false | 3 | 12 | 0.040 |
| cv_weak_index | 1.0 | true | 2 | 12 | 0.040 |
| cv_weak_index | 0.5 | false | 21 | 12 | 0.040 |
| cv_weak_index | 0.75 | false | 8 | 11 | 0.037 |
| cv_weak_index | 0.5 | false | 20 | 11 | 0.037 |
| cv_weak_index | 0.75 | false | 5 | 11 | 0.037 |
| cv_weak_index_larger_n | 0.75 | false | 7 | 32 | 0.107 |
| cv_weak_index_larger_n | 0.75 | false | 8 | 27 | 0.090 |
| cv_weak_index_larger_n | 1.0 | true | 4 | 20 | 0.067 |
| cv_weak_index_larger_n | 0.75 | false | 9 | 18 | 0.060 |
| cv_weak_index_larger_n | 0.75 | false | 6 | 17 | 0.057 |
| cv_weak_index_larger_n | 1.0 | true | 3 | 15 | 0.050 |
| cv_weak_index_larger_n | 0.75 | false | 5 | 14 | 0.047 |
| cv_weak_index_larger_n | 1.0 | false | 4 | 12 | 0.040 |

## Paired tuned versus original index comparisons

| Scenario | Method | Kind | NMC | Coverage difference | MCSE |
|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | family | 300 | -0.003 | 0.023 |
| cv_larger_index | oracle_hc3_log | pointwise | 300 | -0.023 | 0.016 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | family | 300 | -0.010 | 0.022 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | pointwise | 300 | -0.020 | 0.014 |
| cv_strong_index | oracle_hc3_log | family | 300 | 0.027 | 0.014 |
| cv_strong_index | oracle_hc3_log | pointwise | 300 | -0.017 | 0.012 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | family | 300 | 0.003 | 0.017 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | pointwise | 300 | -0.013 | 0.012 |
| cv_weak_index | oracle_hc3_log | family | 300 | -0.050 | 0.028 |
| cv_weak_index | oracle_hc3_log | pointwise | 300 | -0.017 | 0.017 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | family | 300 | -0.027 | 0.026 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | pointwise | 300 | -0.020 | 0.017 |
| cv_weak_index_larger_n | oracle_hc3_log | family | 300 | 0.030 | 0.024 |
| cv_weak_index_larger_n | oracle_hc3_log | pointwise | 300 | 0.013 | 0.014 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | family | 300 | 0.017 | 0.024 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | pointwise | 300 | 0.017 | 0.015 |
