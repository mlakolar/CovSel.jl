# First-stage index tuning

The second-stage guarded CV and diagonal correction are fixed. Index modes compare known indices, the original multiplier 0.5 with soft thresholding, and held-out Stein-alignment CV with optional support unshrinking. No true index or coverage enters tuning. Failures count as noncoverage.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | estimated | 1000 | 0 | 0.911 | [0.892, 0.927] | 0.626 |
| cv_larger_index | oracle_hc3_log | estimated_cv | 1000 | 0 | 0.940 | [0.924, 0.953] | 0.620 |
| cv_larger_index | oracle_hc3_log | known | 1000 | 0 | 0.936 | [0.919, 0.950] | 0.619 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | 1000 | 0 | 0.922 | [0.904, 0.937] | 0.627 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 1000 | 0 | 0.946 | [0.930, 0.958] | 0.620 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.941 | [0.925, 0.954] | 0.618 |
| cv_strong_index | oracle_hc3_log | estimated | 1000 | 0 | 0.914 | [0.895, 0.930] | 0.623 |
| cv_strong_index | oracle_hc3_log | estimated_cv | 1000 | 0 | 0.923 | [0.905, 0.938] | 0.621 |
| cv_strong_index | oracle_hc3_log | known | 1000 | 0 | 0.929 | [0.911, 0.943] | 0.619 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | 1000 | 0 | 0.908 | [0.888, 0.924] | 0.624 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 1000 | 0 | 0.931 | [0.914, 0.945] | 0.620 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.931 | [0.914, 0.945] | 0.618 |
| cv_weak_index | oracle_hc3_log | estimated | 1000 | 4 | 0.821 | [0.796, 0.844] | 0.619 |
| cv_weak_index | oracle_hc3_log | estimated_cv | 1000 | 0 | 0.820 | [0.795, 0.843] | 0.618 |
| cv_weak_index | oracle_hc3_log | known | 1000 | 0 | 0.937 | [0.920, 0.950] | 0.623 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | 1000 | 4 | 0.834 | [0.810, 0.856] | 0.622 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | 1000 | 0 | 0.841 | [0.817, 0.862] | 0.621 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.945 | [0.929, 0.958] | 0.621 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | 1000 | 0 | 0.859 | [0.836, 0.879] | 0.468 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | 1000 | 0 | 0.894 | [0.873, 0.912] | 0.468 |
| cv_weak_index_larger_n | oracle_hc3_log | known | 1000 | 0 | 0.955 | [0.940, 0.966] | 0.470 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | 1000 | 0 | 0.860 | [0.837, 0.880] | 0.469 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | 1000 | 0 | 0.894 | [0.873, 0.912] | 0.468 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.958 | [0.944, 0.969] | 0.469 |

## Pointwise intervals and index error

Index error means are conditional on a successful fit. Index failures remain in coverage denominators.

| Scenario | Method | Index | Entry | Coverage | Bias | SD | RMS SE | Mean direction error | Failures |
|---|---|---|---|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | estimated | (1,1) | 0.946 | 0.008 | 0.120 | 0.118 | 0.385 | 0 |
| cv_larger_index | oracle_hc3_log | estimated | (1,2) | 0.956 | -0.001 | 0.084 | 0.085 | 0.385 | 0 |
| cv_larger_index | oracle_hc3_log | estimated | (1,78) | 0.957 | -0.005 | 0.077 | 0.079 | 0.385 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,1) | 0.936 | 0.009 | 0.116 | 0.116 | 0.180 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,2) | 0.943 | -0.002 | 0.085 | 0.083 | 0.180 | 0 |
| cv_larger_index | oracle_hc3_log | estimated_cv | (1,78) | 0.944 | -0.004 | 0.079 | 0.077 | 0.180 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,1) | 0.946 | 0.005 | 0.115 | 0.114 | 0.000 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,2) | 0.943 | -0.000 | 0.083 | 0.082 | 0.000 | 0 |
| cv_larger_index | oracle_hc3_log | known | (1,78) | 0.949 | -0.000 | 0.078 | 0.076 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.949 | 0.002 | 0.120 | 0.118 | 0.385 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.961 | -0.001 | 0.082 | 0.086 | 0.385 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.959 | -0.005 | 0.076 | 0.079 | 0.385 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.941 | 0.003 | 0.114 | 0.116 | 0.180 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.946 | -0.002 | 0.083 | 0.083 | 0.180 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.945 | -0.004 | 0.078 | 0.077 | 0.180 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.943 | -0.001 | 0.115 | 0.115 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.948 | -0.000 | 0.082 | 0.082 | 0.000 | 0 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.958 | -0.000 | 0.077 | 0.077 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,1) | 0.943 | 0.016 | 0.124 | 0.118 | 0.321 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,2) | 0.938 | 0.002 | 0.086 | 0.085 | 0.321 | 0 |
| cv_strong_index | oracle_hc3_log | estimated | (1,78) | 0.949 | 0.001 | 0.081 | 0.078 | 0.321 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,1) | 0.952 | 0.008 | 0.115 | 0.115 | 0.171 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,2) | 0.942 | 0.001 | 0.086 | 0.083 | 0.171 | 0 |
| cv_strong_index | oracle_hc3_log | estimated_cv | (1,78) | 0.943 | -0.002 | 0.081 | 0.077 | 0.171 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,1) | 0.953 | 0.007 | 0.113 | 0.114 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,2) | 0.950 | -0.000 | 0.084 | 0.082 | 0.000 | 0 |
| cv_strong_index | oracle_hc3_log | known | (1,78) | 0.937 | -0.002 | 0.081 | 0.077 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.944 | 0.010 | 0.124 | 0.118 | 0.321 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.945 | 0.002 | 0.085 | 0.085 | 0.321 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.954 | 0.001 | 0.081 | 0.079 | 0.321 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.954 | 0.003 | 0.116 | 0.116 | 0.171 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.945 | 0.001 | 0.086 | 0.083 | 0.171 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.943 | -0.002 | 0.080 | 0.077 | 0.171 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.952 | 0.002 | 0.113 | 0.115 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.958 | -0.001 | 0.084 | 0.083 | 0.000 | 0 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.940 | -0.001 | 0.081 | 0.077 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | estimated | (1,1) | 0.944 | 0.015 | 0.119 | 0.118 | 0.696 | 4 |
| cv_weak_index | oracle_hc3_log | estimated | (1,2) | 0.947 | 0.002 | 0.088 | 0.085 | 0.696 | 4 |
| cv_weak_index | oracle_hc3_log | estimated | (1,78) | 0.947 | 0.000 | 0.079 | 0.078 | 0.696 | 4 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,1) | 0.947 | 0.020 | 0.118 | 0.117 | 0.609 | 0 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,2) | 0.947 | 0.002 | 0.087 | 0.085 | 0.609 | 0 |
| cv_weak_index | oracle_hc3_log | estimated_cv | (1,78) | 0.934 | 0.001 | 0.082 | 0.078 | 0.609 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,1) | 0.956 | 0.017 | 0.117 | 0.116 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,2) | 0.946 | 0.002 | 0.083 | 0.082 | 0.000 | 0 |
| cv_weak_index | oracle_hc3_log | known | (1,78) | 0.951 | 0.004 | 0.076 | 0.077 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.951 | 0.008 | 0.119 | 0.118 | 0.696 | 4 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.954 | 0.002 | 0.087 | 0.086 | 0.696 | 4 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.950 | 0.000 | 0.079 | 0.078 | 0.696 | 4 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.946 | 0.013 | 0.117 | 0.118 | 0.609 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.949 | 0.002 | 0.086 | 0.085 | 0.609 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.946 | 0.002 | 0.080 | 0.078 | 0.609 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.952 | 0.011 | 0.117 | 0.117 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.952 | 0.002 | 0.081 | 0.083 | 0.000 | 0 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.952 | 0.003 | 0.076 | 0.077 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,1) | 0.950 | 0.008 | 0.087 | 0.087 | 0.475 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,2) | 0.955 | -0.002 | 0.062 | 0.063 | 0.475 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated | (1,78) | 0.943 | 0.001 | 0.058 | 0.058 | 0.475 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,1) | 0.943 | 0.004 | 0.087 | 0.087 | 0.354 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,2) | 0.955 | -0.001 | 0.062 | 0.062 | 0.354 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | estimated_cv | (1,78) | 0.950 | -0.001 | 0.057 | 0.057 | 0.354 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,1) | 0.958 | 0.004 | 0.084 | 0.086 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,2) | 0.949 | -0.003 | 0.064 | 0.061 | 0.000 | 0 |
| cv_weak_index_larger_n | oracle_hc3_log | known | (1,78) | 0.941 | -0.000 | 0.059 | 0.057 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.948 | 0.004 | 0.087 | 0.087 | 0.475 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.957 | -0.002 | 0.062 | 0.063 | 0.475 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.946 | 0.001 | 0.057 | 0.058 | 0.475 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,1) | 0.943 | 0.000 | 0.087 | 0.087 | 0.354 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,2) | 0.955 | -0.001 | 0.061 | 0.062 | 0.354 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | estimated_cv | (1,78) | 0.952 | -0.001 | 0.057 | 0.058 | 0.354 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.956 | 0.001 | 0.084 | 0.086 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.952 | -0.003 | 0.064 | 0.062 | 0.000 | 0 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.944 | -0.000 | 0.059 | 0.057 | 0.000 | 0 |

## Observed first-stage selections

| Scenario | Multiplier | Unshrink | Support | Blocks | Count | Fraction |
|---|---|---|---|---|---|---|
| cv_larger_index | 1.0 | true | 4 | 1 | 530 | 0.530 |
| cv_larger_index | 1.25 | true | 4 | 1 | 60 | 0.060 |
| cv_larger_index | 0.75 | true | 6 | 1 | 40 | 0.040 |
| cv_larger_index | 0.75 | true | 5 | 1 | 39 | 0.039 |
| cv_larger_index | 0.75 | true | 4 | 1 | 33 | 0.033 |
| cv_larger_index | 1.0 | true | 4 | 5 | 26 | 0.026 |
| cv_larger_index | 0.75 | true | 7 | 1 | 21 | 0.021 |
| cv_larger_index | 1.0 | true | 5 | 1 | 21 | 0.021 |
| cv_strong_index | 1.0 | true | 4 | 1 | 465 | 0.465 |
| cv_strong_index | 1.25 | true | 4 | 1 | 109 | 0.109 |
| cv_strong_index | 0.75 | true | 5 | 1 | 53 | 0.053 |
| cv_strong_index | 1.0 | true | 4 | 5 | 42 | 0.042 |
| cv_strong_index | 0.75 | true | 4 | 1 | 40 | 0.040 |
| cv_strong_index | 0.75 | true | 6 | 1 | 31 | 0.031 |
| cv_strong_index | 1.25 | true | 4 | 5 | 28 | 0.028 |
| cv_strong_index | 1.0 | true | 5 | 1 | 23 | 0.023 |
| cv_weak_index | 0.75 | false | 4 | 1 | 65 | 0.065 |
| cv_weak_index | 0.75 | true | 4 | 1 | 62 | 0.062 |
| cv_weak_index | 0.75 | false | 3 | 1 | 42 | 0.042 |
| cv_weak_index | 0.75 | true | 5 | 1 | 40 | 0.040 |
| cv_weak_index | 0.5 | false | 12 | 1 | 39 | 0.039 |
| cv_weak_index | 0.75 | true | 3 | 1 | 33 | 0.033 |
| cv_weak_index | 0.75 | false | 5 | 1 | 33 | 0.033 |
| cv_weak_index | 1.0 | true | 1 | 1 | 32 | 0.032 |
| cv_weak_index_larger_n | 0.75 | true | 5 | 1 | 92 | 0.092 |
| cv_weak_index_larger_n | 1.0 | true | 4 | 1 | 89 | 0.089 |
| cv_weak_index_larger_n | 0.75 | true | 4 | 1 | 85 | 0.085 |
| cv_weak_index_larger_n | 1.0 | true | 3 | 1 | 76 | 0.076 |
| cv_weak_index_larger_n | 0.75 | true | 6 | 1 | 52 | 0.052 |
| cv_weak_index_larger_n | 0.5 | false | 13 | 1 | 47 | 0.047 |
| cv_weak_index_larger_n | 0.5 | false | 14 | 1 | 46 | 0.046 |
| cv_weak_index_larger_n | 0.75 | false | 4 | 1 | 40 | 0.040 |

## Paired tuned versus original index comparisons

| Scenario | Method | Kind | NMC | Coverage difference | MCSE |
|---|---|---|---|---|---|
| cv_larger_index | oracle_hc3_log | family | 1000 | 0.029 | 0.011 |
| cv_larger_index | oracle_hc3_log | pointwise | 1000 | -0.010 | 0.009 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | family | 1000 | 0.024 | 0.011 |
| cv_larger_index | sqrt_refit_df_hc3_bc_log | pointwise | 1000 | -0.008 | 0.009 |
| cv_strong_index | oracle_hc3_log | family | 1000 | 0.009 | 0.011 |
| cv_strong_index | oracle_hc3_log | pointwise | 1000 | 0.009 | 0.009 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | family | 1000 | 0.023 | 0.011 |
| cv_strong_index | sqrt_refit_df_hc3_bc_log | pointwise | 1000 | 0.010 | 0.009 |
| cv_weak_index | oracle_hc3_log | family | 1000 | -0.001 | 0.016 |
| cv_weak_index | oracle_hc3_log | pointwise | 1000 | 0.003 | 0.009 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | family | 1000 | 0.007 | 0.015 |
| cv_weak_index | sqrt_refit_df_hc3_bc_log | pointwise | 1000 | -0.005 | 0.009 |
| cv_weak_index_larger_n | oracle_hc3_log | family | 1000 | 0.035 | 0.014 |
| cv_weak_index_larger_n | oracle_hc3_log | pointwise | 1000 | -0.007 | 0.010 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | family | 1000 | 0.034 | 0.013 |
| cv_weak_index_larger_n | sqrt_refit_df_hc3_bc_log | pointwise | 1000 | -0.005 | 0.010 |
