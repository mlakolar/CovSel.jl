# First-stage index tuning

The second-stage guarded CV and diagonal correction are fixed. Index modes compare known indices, the original multiplier 0.5 with soft thresholding, and held-out Stein-alignment CV with optional support unshrinking. No true index or coverage enters tuning. Failures count as noncoverage.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | estimated | 300 | 0 | 0.910 | [0.872, 0.937] | 0.625 |
| cv_larger_index | oracle_hc3_log | estimated_cv | 300 | 0 | 0.910 | [0.872, 0.937] | 0.621 |
| cv_larger_index | oracle_hc3_log | known | 300 | 0 | 0.940 | [0.907, 0.962] | 0.620 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.920 | [0.884, 0.946] | 0.626 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 0 | 0.930 | [0.895, 0.954] | 0.620 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.940 | [0.907, 0.962] | 0.618 |
| cv_strong_index | oracle_hc3_log | estimated | 300 | 0 | 0.943 | [0.911, 0.964] | 0.624 |
| cv_strong_index | oracle_hc3_log | estimated_cv | 300 | 0 | 0.980 | [0.957, 0.991] | 0.623 |
| cv_strong_index | oracle_hc3_log | known | 300 | 0 | 0.963 | [0.936, 0.979] | 0.622 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.930 | [0.895, 0.954] | 0.624 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 0 | 0.990 | [0.971, 0.997] | 0.623 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.967 | [0.940, 0.982] | 0.621 |
| cv_weak_index | oracle_hc3_log | estimated | 300 | 1 | 0.810 | [0.762, 0.850] | 0.620 |
| cv_weak_index | oracle_hc3_log | estimated_cv | 300 | 2 | 0.827 | [0.780, 0.865] | 0.618 |
| cv_weak_index | oracle_hc3_log | known | 300 | 0 | 0.963 | [0.936, 0.979] | 0.622 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | 300 | 1 | 0.797 | [0.748, 0.838] | 0.624 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 2 | 0.830 | [0.783, 0.868] | 0.620 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.963 | [0.936, 0.979] | 0.621 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | 300 | 0 | 0.810 | [0.762, 0.850] | 0.469 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | 300 | 0 | 0.893 | [0.853, 0.923] | 0.469 |
| cv_weak_index_larger_n | oracle_hc3_log | known | 300 | 0 | 0.950 | [0.919, 0.969] | 0.470 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | 300 | 0 | 0.830 | [0.783, 0.868] | 0.470 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | 300 | 0 | 0.903 | [0.865, 0.932] | 0.469 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | 300 | 0 | 0.960 | [0.931, 0.977] | 0.469 |

## Pointwise intervals and index error

Index error means are conditional on a successful fit. Index failures remain in coverage denominators.

| Scenario | Method | Index | Entry | Coverage | Bias | SD | RMS SE | Mean direction error | Failures |
|---|---|---|---|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | estimated | (1,1) | 0.953 | 0.010 | 0.122 | 0.118 | 0.382 | 0 |
| cv_larger_index | oracle_hc3_log | estimated | (1,2) | 0.970 | 0.007 | 0.080 | 0.085 | 0.382 | 0 |
| cv_larger_index | oracle_hc3_log | estimated | (1,78) | 0.933 | -0.008 | 0.082 | 0.079 | 0.382 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,1) | 0.950 | 0.011 | 0.116 | 0.116 | 0.176 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,2) | 0.950 | -0.001 | 0.087 | 0.083 | 0.176 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,78) | 0.930 | -0.004 | 0.080 | 0.077 | 0.176 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,1) | 0.957 | 0.008 | 0.115 | 0.115 | 0.000 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,2) | 0.960 | -0.004 | 0.083 | 0.082 | 0.000 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,78) | 0.957 | -0.008 | 0.077 | 0.076 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.953 | 0.004 | 0.124 | 0.119 | 0.382 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.977 | 0.007 | 0.078 | 0.085 | 0.382 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.947 | -0.008 | 0.082 | 0.079 | 0.382 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.950 | 0.004 | 0.114 | 0.116 | 0.176 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.953 | -0.000 | 0.085 | 0.083 | 0.176 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.937 | -0.004 | 0.078 | 0.078 | 0.176 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.963 | 0.001 | 0.113 | 0.115 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.963 | -0.003 | 0.082 | 0.082 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.957 | -0.008 | 0.076 | 0.077 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,1) | 0.963 | 0.000 | 0.115 | 0.117 | 0.318 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,2) | 0.950 | 0.003 | 0.080 | 0.084 | 0.318 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,78) | 0.937 | -0.002 | 0.079 | 0.079 | 0.318 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,1) | 0.943 | 0.002 | 0.115 | 0.116 | 0.166 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,2) | 0.963 | 0.007 | 0.080 | 0.082 | 0.166 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,78) | 0.953 | -0.001 | 0.078 | 0.077 | 0.166 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,1) | 0.957 | 0.004 | 0.110 | 0.115 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,2) | 0.957 | 0.003 | 0.080 | 0.081 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,78) | 0.953 | 0.001 | 0.077 | 0.077 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.963 | -0.006 | 0.115 | 0.118 | 0.318 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.960 | 0.003 | 0.079 | 0.084 | 0.318 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.943 | -0.002 | 0.079 | 0.079 | 0.318 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.940 | -0.004 | 0.115 | 0.116 | 0.166 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.967 | 0.007 | 0.078 | 0.083 | 0.166 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.960 | -0.001 | 0.077 | 0.078 | 0.166 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | -0.004 | 0.111 | 0.115 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.963 | 0.003 | 0.078 | 0.082 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.953 | 0.001 | 0.076 | 0.077 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | estimated | (1,1) | 0.943 | 0.008 | 0.117 | 0.116 | 0.694 | 1 |
| cv_weak_index | oracle_hc3_log | estimated | (1,2) | 0.963 | 0.000 | 0.084 | 0.085 | 0.694 | 1 |
| cv_weak_index | oracle_hc3_log | estimated | (1,78) | 0.947 | -0.006 | 0.078 | 0.079 | 0.694 | 1 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,1) | 0.910 | 0.013 | 0.126 | 0.117 | 0.620 | 2 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,2) | 0.937 | 0.001 | 0.086 | 0.084 | 0.620 | 2 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,78) | 0.950 | -0.007 | 0.075 | 0.078 | 0.620 | 2 |
| cv_weak_index | oracle_hc3_log | known | (1,1) | 0.953 | 0.019 | 0.115 | 0.117 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,2) | 0.967 | 0.002 | 0.079 | 0.082 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,78) | 0.970 | -0.001 | 0.073 | 0.077 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.947 | 0.001 | 0.117 | 0.116 | 0.694 | 1 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.967 | -0.000 | 0.084 | 0.086 | 0.694 | 1 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.950 | -0.005 | 0.078 | 0.079 | 0.694 | 1 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.907 | 0.007 | 0.126 | 0.118 | 0.620 | 2 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.943 | 0.000 | 0.085 | 0.085 | 0.620 | 2 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.953 | -0.006 | 0.074 | 0.078 | 0.620 | 2 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.963 | 0.014 | 0.114 | 0.118 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.970 | 0.000 | 0.079 | 0.082 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.973 | -0.002 | 0.072 | 0.078 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,1) | 0.950 | 0.008 | 0.087 | 0.087 | 0.480 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,2) | 0.950 | -0.001 | 0.062 | 0.062 | 0.480 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,78) | 0.947 | 0.001 | 0.058 | 0.058 | 0.480 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,1) | 0.960 | 0.014 | 0.082 | 0.087 | 0.351 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,2) | 0.947 | 0.002 | 0.062 | 0.062 | 0.351 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,78) | 0.953 | 0.002 | 0.059 | 0.058 | 0.351 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,1) | 0.947 | 0.013 | 0.086 | 0.087 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,2) | 0.937 | 0.003 | 0.061 | 0.061 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,78) | 0.950 | 0.003 | 0.058 | 0.058 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.950 | 0.005 | 0.087 | 0.088 | 0.480 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.950 | -0.001 | 0.061 | 0.063 | 0.480 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.947 | 0.001 | 0.058 | 0.059 | 0.480 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.963 | 0.011 | 0.081 | 0.087 | 0.351 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.950 | 0.002 | 0.061 | 0.062 | 0.351 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.947 | 0.003 | 0.059 | 0.058 | 0.351 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.947 | 0.010 | 0.085 | 0.087 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.937 | 0.004 | 0.061 | 0.061 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.953 | 0.004 | 0.058 | 0.058 | 0.000 | 0 |

## Observed first-stage selections

| Scenario | Multiplier | Unshrink | Support | Blocks | Count | Fraction |
|---|---|---|---|---|---|---|
| cv_larger_index | 1.0 | true | 4 | 1 | 157 | 0.523 |
| cv_larger_index | 0.75 | true | 5 | 1 | 18 | 0.060 |
| cv_larger_index | 1.25 | true | 4 | 1 | 14 | 0.047 |
| cv_larger_index | 0.75 | true | 6 | 1 | 12 | 0.040 |
| cv_larger_index | 1.0 | true | 4 | 5 | 12 | 0.040 |
| cv_larger_index | 0.5 | false | 20 | 1 | 8 | 0.027 |
| cv_larger_index | 1.0 | true | 5 | 1 | 7 | 0.023 |
| cv_larger_index | 0.75 | true | 4 | 1 | 7 | 0.023 |
| cv_strong_index | 1.0 | true | 4 | 1 | 148 | 0.493 |
| cv_strong_index | 1.25 | true | 4 | 1 | 30 | 0.100 |
| cv_strong_index | 0.75 | true | 4 | 1 | 17 | 0.057 |
| cv_strong_index | 0.75 | true | 6 | 1 | 15 | 0.050 |
| cv_strong_index | 0.75 | true | 5 | 1 | 12 | 0.040 |
| cv_strong_index | 1.25 | true | 4 | 5 | 9 | 0.030 |
| cv_strong_index | 1.0 | true | 4 | 5 | 8 | 0.027 |
| cv_strong_index | 1.0 | true | 5 | 1 | 8 | 0.027 |
| cv_weak_index | 0.75 | false | 4 | 1 | 19 | 0.063 |
| cv_weak_index | 0.75 | true | 4 | 1 | 18 | 0.060 |
| cv_weak_index | 0.75 | false | 3 | 1 | 15 | 0.050 |
| cv_weak_index | 0.75 | false | 5 | 1 | 13 | 0.043 |
| cv_weak_index | 1.0 | true | 2 | 1 | 12 | 0.040 |
| cv_weak_index | 0.75 | true | 5 | 1 | 10 | 0.033 |
| cv_weak_index | 0.5 | false | 11 | 1 | 10 | 0.033 |
| cv_weak_index | 0.5 | false | 13 | 1 | 9 | 0.030 |
| cv_weak_index_larger_n | 0.75 | true | 4 | 1 | 34 | 0.113 |
| cv_weak_index_larger_n | 1.0 | true | 4 | 1 | 25 | 0.083 |
| cv_weak_index_larger_n | 0.75 | true | 5 | 1 | 24 | 0.080 |
| cv_weak_index_larger_n | 0.75 | true | 6 | 1 | 18 | 0.060 |
| cv_weak_index_larger_n | 1.0 | true | 3 | 1 | 16 | 0.053 |
| cv_weak_index_larger_n | 0.75 | false | 5 | 1 | 15 | 0.050 |
| cv_weak_index_larger_n | 0.5 | false | 13 | 1 | 15 | 0.050 |
| cv_weak_index_larger_n | 0.75 | false | 4 | 1 | 15 | 0.050 |

## Paired tuned versus original index comparisons

| Scenario | Method | Kind | NMC | Coverage difference | MCSE |
|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | family | 300 | 0.000 | 0.022 |
| cv_larger_index | oracle_hc3_log | pointwise | 300 | -0.003 | 0.017 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | family | 300 | 0.010 | 0.020 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | pointwise | 300 | -0.003 | 0.018 |
| cv_strong_index | oracle_hc3_log | family | 300 | 0.037 | 0.016 |
| cv_strong_index | oracle_hc3_log | pointwise | 300 | -0.020 | 0.016 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | family | 300 | 0.060 | 0.016 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | pointwise | 300 | -0.023 | 0.016 |
| cv_weak_index | oracle_hc3_log | family | 300 | 0.017 | 0.030 |
| cv_weak_index | oracle_hc3_log | pointwise | 300 | -0.033 | 0.020 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | family | 300 | 0.033 | 0.031 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | pointwise | 300 | -0.040 | 0.020 |
| cv_weak_index_larger_n | oracle_hc3_log | family | 300 | 0.083 | 0.026 |
| cv_weak_index_larger_n | oracle_hc3_log | pointwise | 300 | 0.010 | 0.016 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | family | 300 | 0.073 | 0.025 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | pointwise | 300 | 0.013 | 0.015 |
