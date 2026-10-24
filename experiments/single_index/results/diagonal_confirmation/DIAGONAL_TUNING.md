# Diagonal and CV confirmation

Coverage is unconditional over all requested fits. Failures count as noncoverage. Intervals are nominal 95%. Bands are joint over the three selected entries and the full grid. Wilson intervals quantify Monte Carlo uncertainty, not inferential uncertainty in a dataset.

Method suffixes: `refit` removes coefficient shrinkage; `df` refits the union support and normalizes residual-maker row variance; `hc3` adjusts final smoothing leverage; `bc` subtracts second-order inverse curvature. The components are defined in `docs/single_index/DIAGONAL_TUNING.md`. Actual tuning values are in `selected_tuning.csv`.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_generated_inference | oracle_hc3_log | estimated | 1000 | 0 | 0.925 | [0.907, 0.940] | 0.623 |
| cv_generated_inference | oracle_hc3_log | known | 1000 | 0 | 0.953 | [0.938, 0.964] | 0.620 |
| cv_generated_inference | sqrt_refit_block | estimated | 1000 | 0 | 0.912 | [0.893, 0.928] | 0.618 |
| cv_generated_inference | sqrt_refit_block | known | 1000 | 0 | 0.926 | [0.908, 0.941] | 0.613 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | 1000 | 0 | 0.922 | [0.904, 0.937] | 0.624 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.947 | [0.931, 0.959] | 0.619 |
| cv_generated_prediction | sqrtlasso | estimated | 1000 | 0 | 0.894 | [0.873, 0.912] | 0.527 |
| cv_generated_prediction | sqrtlasso | known | 1000 | 0 | 0.931 | [0.914, 0.945] | 0.524 |
| cv_highdim_inference | oracle_hc3_log | known | 1000 | 0 | 0.946 | [0.930, 0.958] | 0.988 |
| cv_highdim_inference | sqrt_refit_block | known | 1000 | 0 | 0.920 | [0.902, 0.935] | 0.968 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.952 | [0.937, 0.964] | 0.993 |
| cv_highdim_prediction | sqrtlasso | known | 1000 | 0 | 0.904 | [0.884, 0.921] | 0.904 |
| fixed_block_corrections | aclime_refit_block | known | 1000 | 0 | 0.939 | [0.922, 0.952] | 0.813 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | 1000 | 0 | 0.962 | [0.948, 0.972] | 0.827 |
| fixed_block_corrections | oracle | known | 1000 | 0 | 0.948 | [0.932, 0.960] | 0.807 |
| fixed_block_corrections | oracle_hc3_bc_log | known | 1000 | 0 | 0.966 | [0.953, 0.976] | 0.827 |
| fixed_block_corrections | scaled_refit_block | known | 1000 | 0 | 0.961 | [0.947, 0.971] | 0.852 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | 1000 | 0 | 0.956 | [0.941, 0.967] | 0.846 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | 1000 | 0 | 0.957 | [0.943, 0.968] | 0.859 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.952 | [0.937, 0.964] | 0.865 |
| fixed_varying_corrections | aclime_refit_block | known | 1000 | 0 | 0.941 | [0.925, 0.954] | 0.615 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | 1000 | 0 | 0.952 | [0.937, 0.964] | 0.621 |
| fixed_varying_corrections | oracle | known | 1000 | 0 | 0.944 | [0.928, 0.957] | 0.613 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | 1000 | 0 | 0.950 | [0.935, 0.962] | 0.622 |
| fixed_varying_corrections | scaled_refit_block | known | 1000 | 0 | 0.942 | [0.926, 0.955] | 0.642 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | 1000 | 0 | 0.950 | [0.935, 0.962] | 0.639 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | 1000 | 0 | 0.930 | [0.912, 0.944] | 0.644 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.925 | [0.907, 0.940] | 0.649 |

## Pointwise intervals at zero

| Scenario | Method | Index | Entry | Coverage | Bias | Empirical SD | RMS SE | SD/SE | Mean length | Failures |
|---|---|---|---|---|---|---|---|---|---|---|
| cv_generated_inference | oracle_hc3_log | estimated | (1,1) | 0.944 | 0.007 | 0.121 | 0.117 | 1.034 | 0.457 | 0 |
| cv_generated_inference | oracle_hc3_log | estimated | (1,2) | 0.963 | -0.000 | 0.080 | 0.085 | 0.951 | 0.329 | 0 |
| cv_generated_inference | oracle_hc3_log | estimated | (1,78) | 0.955 | 0.000 | 0.077 | 0.078 | 0.993 | 0.303 | 0 |
| cv_generated_inference | oracle_hc3_log | known | (1,1) | 0.949 | 0.011 | 0.119 | 0.115 | 1.033 | 0.451 | 0 |
| cv_generated_inference | oracle_hc3_log | known | (1,2) | 0.949 | 0.002 | 0.082 | 0.082 | 1.003 | 0.320 | 0 |
| cv_generated_inference | oracle_hc3_log | known | (1,78) | 0.947 | 0.004 | 0.078 | 0.077 | 1.015 | 0.298 | 0 |
| cv_generated_inference | sqrt_refit_block | estimated | (1,1) | 0.942 | 0.017 | 0.122 | 0.117 | 1.042 | 0.456 | 0 |
| cv_generated_inference | sqrt_refit_block | estimated | (1,2) | 0.959 | 0.000 | 0.081 | 0.085 | 0.959 | 0.330 | 0 |
| cv_generated_inference | sqrt_refit_block | estimated | (1,78) | 0.949 | 0.000 | 0.078 | 0.078 | 1.001 | 0.304 | 0 |
| cv_generated_inference | sqrt_refit_block | known | (1,1) | 0.954 | 0.022 | 0.120 | 0.116 | 1.038 | 0.450 | 0 |
| cv_generated_inference | sqrt_refit_block | known | (1,2) | 0.945 | 0.002 | 0.083 | 0.082 | 1.013 | 0.321 | 0 |
| cv_generated_inference | sqrt_refit_block | known | (1,78) | 0.945 | 0.004 | 0.078 | 0.077 | 1.017 | 0.299 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.944 | 0.000 | 0.120 | 0.118 | 1.024 | 0.459 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.963 | 0.000 | 0.080 | 0.085 | 0.937 | 0.331 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.956 | 0.000 | 0.077 | 0.078 | 0.982 | 0.305 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.953 | 0.005 | 0.118 | 0.116 | 1.020 | 0.453 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.952 | 0.002 | 0.082 | 0.082 | 0.991 | 0.321 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.949 | 0.004 | 0.076 | 0.077 | 0.994 | 0.300 | 0 |
| cv_generated_prediction | sqrtlasso | estimated | (1,1) | 0.927 | 0.027 | 0.115 | 0.103 | 1.116 | 0.399 | 0 |
| cv_generated_prediction | sqrtlasso | estimated | (1,2) | 0.954 | -0.011 | 0.072 | 0.074 | 0.982 | 0.288 | 0 |
| cv_generated_prediction | sqrtlasso | estimated | (1,78) | 0.945 | -0.000 | 0.069 | 0.068 | 1.007 | 0.266 | 0 |
| cv_generated_prediction | sqrtlasso | known | (1,1) | 0.938 | 0.028 | 0.112 | 0.102 | 1.105 | 0.394 | 0 |
| cv_generated_prediction | sqrtlasso | known | (1,2) | 0.947 | -0.009 | 0.074 | 0.072 | 1.020 | 0.281 | 0 |
| cv_generated_prediction | sqrtlasso | known | (1,78) | 0.947 | 0.003 | 0.071 | 0.068 | 1.045 | 0.263 | 0 |
| cv_highdim_inference | oracle_hc3_log | known | (1,1) | 0.935 | 0.033 | 0.189 | 0.185 | 1.019 | 0.720 | 0 |
| cv_highdim_inference | oracle_hc3_log | known | (1,158) | 0.956 | 0.006 | 0.126 | 0.124 | 1.013 | 0.480 | 0 |
| cv_highdim_inference | oracle_hc3_log | known | (1,2) | 0.952 | 0.002 | 0.134 | 0.133 | 1.007 | 0.515 | 0 |
| cv_highdim_inference | sqrt_refit_block | known | (1,1) | 0.937 | 0.057 | 0.199 | 0.187 | 1.068 | 0.714 | 0 |
| cv_highdim_inference | sqrt_refit_block | known | (1,158) | 0.950 | 0.005 | 0.128 | 0.125 | 1.022 | 0.483 | 0 |
| cv_highdim_inference | sqrt_refit_block | known | (1,2) | 0.950 | -0.000 | 0.138 | 0.134 | 1.030 | 0.517 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.940 | 0.015 | 0.190 | 0.187 | 1.019 | 0.728 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.956 | 0.005 | 0.122 | 0.126 | 0.972 | 0.487 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.962 | 0.001 | 0.132 | 0.135 | 0.979 | 0.521 | 0 |
| cv_highdim_prediction | sqrtlasso | known | (1,1) | 0.912 | 0.058 | 0.209 | 0.177 | 1.181 | 0.676 | 0 |
| cv_highdim_prediction | sqrtlasso | known | (1,158) | 0.953 | 0.005 | 0.124 | 0.120 | 1.033 | 0.462 | 0 |
| cv_highdim_prediction | sqrtlasso | known | (1,2) | 0.929 | -0.049 | 0.127 | 0.126 | 1.008 | 0.487 | 0 |
| fixed_block_corrections | aclime_refit_block | known | (1,1) | 0.943 | 0.032 | 0.161 | 0.154 | 1.047 | 0.594 | 0 |
| fixed_block_corrections | aclime_refit_block | known | (1,2) | 0.939 | 0.000 | 0.118 | 0.110 | 1.065 | 0.429 | 0 |
| fixed_block_corrections | aclime_refit_block | known | (1,38) | 0.952 | -0.001 | 0.106 | 0.103 | 1.022 | 0.401 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.944 | 0.003 | 0.156 | 0.155 | 1.011 | 0.603 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.954 | 0.003 | 0.113 | 0.111 | 1.024 | 0.430 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,38) | 0.960 | -0.001 | 0.103 | 0.104 | 0.986 | 0.403 | 0 |
| fixed_block_corrections | oracle | known | (1,1) | 0.937 | 0.016 | 0.156 | 0.152 | 1.030 | 0.586 | 0 |
| fixed_block_corrections | oracle | known | (1,2) | 0.937 | 0.004 | 0.116 | 0.109 | 1.061 | 0.424 | 0 |
| fixed_block_corrections | oracle | known | (1,38) | 0.951 | -0.001 | 0.105 | 0.102 | 1.024 | 0.396 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,1) | 0.945 | -0.004 | 0.153 | 0.153 | 1.000 | 0.598 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,2) | 0.950 | 0.004 | 0.112 | 0.110 | 1.021 | 0.428 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,38) | 0.959 | -0.001 | 0.102 | 0.103 | 0.986 | 0.400 | 0 |
| fixed_block_corrections | scaled_refit_block | known | (1,1) | 0.939 | 0.076 | 0.169 | 0.160 | 1.059 | 0.617 | 0 |
| fixed_block_corrections | scaled_refit_block | known | (1,2) | 0.941 | 0.003 | 0.123 | 0.115 | 1.071 | 0.446 | 0 |
| fixed_block_corrections | scaled_refit_block | known | (1,38) | 0.951 | -0.001 | 0.111 | 0.108 | 1.028 | 0.418 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | (1,1) | 0.946 | 0.037 | 0.163 | 0.159 | 1.024 | 0.613 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | (1,2) | 0.945 | 0.003 | 0.118 | 0.114 | 1.030 | 0.443 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | (1,38) | 0.952 | -0.001 | 0.107 | 0.107 | 0.997 | 0.416 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.942 | 0.037 | 0.163 | 0.159 | 1.024 | 0.621 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.945 | 0.003 | 0.118 | 0.114 | 1.030 | 0.443 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,38) | 0.952 | -0.001 | 0.107 | 0.107 | 0.997 | 0.416 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.938 | 0.046 | 0.165 | 0.160 | 1.029 | 0.625 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.944 | 0.004 | 0.119 | 0.115 | 1.032 | 0.446 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.951 | -0.001 | 0.107 | 0.108 | 0.993 | 0.418 | 0 |
| fixed_varying_corrections | aclime_refit_block | known | (1,1) | 0.944 | 0.021 | 0.117 | 0.116 | 1.009 | 0.449 | 0 |
| fixed_varying_corrections | aclime_refit_block | known | (1,2) | 0.933 | -0.002 | 0.088 | 0.082 | 1.069 | 0.321 | 0 |
| fixed_varying_corrections | aclime_refit_block | known | (1,78) | 0.947 | 0.001 | 0.079 | 0.077 | 1.028 | 0.299 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.946 | 0.006 | 0.115 | 0.116 | 0.995 | 0.452 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.942 | 0.000 | 0.086 | 0.083 | 1.040 | 0.322 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,78) | 0.949 | 0.001 | 0.078 | 0.077 | 1.006 | 0.300 | 0 |
| fixed_varying_corrections | oracle | known | (1,1) | 0.946 | 0.014 | 0.115 | 0.115 | 1.003 | 0.445 | 0 |
| fixed_varying_corrections | oracle | known | (1,2) | 0.935 | 0.000 | 0.087 | 0.082 | 1.064 | 0.319 | 0 |
| fixed_varying_corrections | oracle | known | (1,78) | 0.945 | 0.001 | 0.079 | 0.076 | 1.029 | 0.298 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,1) | 0.951 | 0.002 | 0.114 | 0.115 | 0.986 | 0.451 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,2) | 0.942 | 0.000 | 0.086 | 0.082 | 1.043 | 0.321 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,78) | 0.950 | 0.001 | 0.077 | 0.077 | 1.007 | 0.299 | 0 |
| fixed_varying_corrections | scaled_refit_block | known | (1,1) | 0.927 | 0.060 | 0.122 | 0.120 | 1.023 | 0.464 | 0 |
| fixed_varying_corrections | scaled_refit_block | known | (1,2) | 0.930 | -0.002 | 0.091 | 0.085 | 1.065 | 0.332 | 0 |
| fixed_varying_corrections | scaled_refit_block | known | (1,78) | 0.945 | 0.001 | 0.082 | 0.080 | 1.023 | 0.311 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | (1,1) | 0.943 | 0.038 | 0.120 | 0.119 | 1.005 | 0.462 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | (1,2) | 0.940 | -0.001 | 0.088 | 0.085 | 1.040 | 0.331 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | (1,78) | 0.950 | 0.001 | 0.080 | 0.079 | 1.004 | 0.309 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.930 | 0.038 | 0.120 | 0.119 | 1.005 | 0.465 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.940 | -0.001 | 0.088 | 0.085 | 1.040 | 0.331 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,78) | 0.950 | 0.001 | 0.080 | 0.079 | 1.004 | 0.309 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.919 | 0.048 | 0.122 | 0.120 | 1.014 | 0.469 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.940 | -0.001 | 0.089 | 0.086 | 1.043 | 0.333 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.948 | 0.000 | 0.080 | 0.080 | 1.004 | 0.311 | 0 |

## Curvewise bands

| Scenario | Method | Index | Entry | Coverage | Mean width | Failures |
|---|---|---|---|---|---|---|
| cv_generated_inference | oracle_hc3_log | estimated | (1,1) | 0.938 | 0.698 | 0 |
| cv_generated_inference | oracle_hc3_log | estimated | (1,2) | 0.916 | 0.500 | 0 |
| cv_generated_inference | oracle_hc3_log | estimated | (1,78) | 0.953 | 0.456 | 0 |
| cv_generated_inference | oracle_hc3_log | known | (1,1) | 0.944 | 0.696 | 0 |
| cv_generated_inference | oracle_hc3_log | known | (1,2) | 0.937 | 0.497 | 0 |
| cv_generated_inference | oracle_hc3_log | known | (1,78) | 0.954 | 0.453 | 0 |
| cv_generated_inference | sqrt_refit_block | estimated | (1,1) | 0.912 | 0.684 | 0 |
| cv_generated_inference | sqrt_refit_block | estimated | (1,2) | 0.912 | 0.502 | 0 |
| cv_generated_inference | sqrt_refit_block | estimated | (1,78) | 0.949 | 0.456 | 0 |
| cv_generated_inference | sqrt_refit_block | known | (1,1) | 0.923 | 0.680 | 0 |
| cv_generated_inference | sqrt_refit_block | known | (1,2) | 0.935 | 0.499 | 0 |
| cv_generated_inference | sqrt_refit_block | known | (1,78) | 0.952 | 0.451 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.936 | 0.695 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.916 | 0.503 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.956 | 0.457 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | 0.690 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.944 | 0.499 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.960 | 0.453 | 0 |
| cv_generated_prediction | sqrtlasso | estimated | (1,1) | 0.907 | 0.580 | 0 |
| cv_generated_prediction | sqrtlasso | estimated | (1,2) | 0.881 | 0.424 | 0 |
| cv_generated_prediction | sqrtlasso | estimated | (1,78) | 0.945 | 0.387 | 0 |
| cv_generated_prediction | sqrtlasso | known | (1,1) | 0.925 | 0.577 | 0 |
| cv_generated_prediction | sqrtlasso | known | (1,2) | 0.932 | 0.422 | 0 |
| cv_generated_prediction | sqrtlasso | known | (1,78) | 0.948 | 0.384 | 0 |
| cv_highdim_inference | oracle_hc3_log | known | (1,1) | 0.928 | 1.103 | 0 |
| cv_highdim_inference | oracle_hc3_log | known | (1,158) | 0.954 | 0.715 | 0 |
| cv_highdim_inference | oracle_hc3_log | known | (1,2) | 0.955 | 0.785 | 0 |
| cv_highdim_inference | sqrt_refit_block | known | (1,1) | 0.894 | 1.058 | 0 |
| cv_highdim_inference | sqrt_refit_block | known | (1,158) | 0.959 | 0.714 | 0 |
| cv_highdim_inference | sqrt_refit_block | known | (1,2) | 0.951 | 0.791 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.919 | 1.101 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.975 | 0.720 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.966 | 0.797 | 0 |
| cv_highdim_prediction | sqrtlasso | known | (1,1) | 0.878 | 0.981 | 0 |
| cv_highdim_prediction | sqrtlasso | known | (1,158) | 0.961 | 0.672 | 0 |
| cv_highdim_prediction | sqrtlasso | known | (1,2) | 0.930 | 0.732 | 0 |
| fixed_block_corrections | aclime_refit_block | known | (1,1) | 0.916 | 0.893 | 0 |
| fixed_block_corrections | aclime_refit_block | known | (1,2) | 0.960 | 0.661 | 0 |
| fixed_block_corrections | aclime_refit_block | known | (1,38) | 0.958 | 0.600 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.940 | 0.918 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.973 | 0.663 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,38) | 0.964 | 0.603 | 0 |
| fixed_block_corrections | oracle | known | (1,1) | 0.921 | 0.891 | 0 |
| fixed_block_corrections | oracle | known | (1,2) | 0.959 | 0.652 | 0 |
| fixed_block_corrections | oracle | known | (1,38) | 0.960 | 0.596 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,1) | 0.948 | 0.924 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,2) | 0.971 | 0.659 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,38) | 0.973 | 0.602 | 0 |
| fixed_block_corrections | scaled_refit_block | known | (1,1) | 0.949 | 0.938 | 0 |
| fixed_block_corrections | scaled_refit_block | known | (1,2) | 0.963 | 0.690 | 0 |
| fixed_block_corrections | scaled_refit_block | known | (1,38) | 0.961 | 0.631 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | (1,1) | 0.938 | 0.932 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | (1,2) | 0.977 | 0.686 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | known | (1,38) | 0.970 | 0.627 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.929 | 0.957 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.977 | 0.686 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,38) | 0.970 | 0.627 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.929 | 0.966 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.973 | 0.692 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.963 | 0.631 | 0 |
| fixed_varying_corrections | aclime_refit_block | known | (1,1) | 0.942 | 0.682 | 0 |
| fixed_varying_corrections | aclime_refit_block | known | (1,2) | 0.943 | 0.499 | 0 |
| fixed_varying_corrections | aclime_refit_block | known | (1,78) | 0.952 | 0.452 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.948 | 0.694 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.947 | 0.500 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,78) | 0.958 | 0.453 | 0 |
| fixed_varying_corrections | oracle | known | (1,1) | 0.934 | 0.684 | 0 |
| fixed_varying_corrections | oracle | known | (1,2) | 0.945 | 0.495 | 0 |
| fixed_varying_corrections | oracle | known | (1,78) | 0.947 | 0.451 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,1) | 0.953 | 0.698 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,2) | 0.952 | 0.498 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,78) | 0.952 | 0.454 | 0 |
| fixed_varying_corrections | scaled_refit_block | known | (1,1) | 0.940 | 0.714 | 0 |
| fixed_varying_corrections | scaled_refit_block | known | (1,2) | 0.951 | 0.519 | 0 |
| fixed_varying_corrections | scaled_refit_block | known | (1,78) | 0.946 | 0.474 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | (1,1) | 0.955 | 0.710 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | (1,2) | 0.952 | 0.517 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | known | (1,78) | 0.954 | 0.472 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.916 | 0.721 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.952 | 0.517 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,78) | 0.954 | 0.472 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.909 | 0.729 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.953 | 0.521 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.956 | 0.475 | 0 |

## Selected tuning

Frequencies are descriptive. No truth or coverage entered selection.

| Scenario | Index | Final h | Pilot h | Penalty multiplier | Count | Fraction |
|---|---|---|---|---|---|---|
| cv_generated_inference | estimated | 0.605 | 1.500 | 1.000 | 772 | 0.772 |
| cv_generated_inference | estimated | 0.605 | 1.200 | 1.000 | 227 | 0.227 |
| cv_generated_inference | estimated | 0.605 | 1.500 | 0.700 | 1 | 0.001 |
| cv_generated_inference | known | 0.605 | 1.500 | 1.000 | 754 | 0.754 |
| cv_generated_inference | known | 0.605 | 1.200 | 1.000 | 244 | 0.244 |
| cv_generated_inference | known | 0.605 | 1.500 | 0.700 | 2 | 0.002 |
| cv_generated_prediction | estimated | 0.842 | 1.200 | 0.700 | 700 | 0.700 |
| cv_generated_prediction | estimated | 0.842 | 1.200 | 0.400 | 215 | 0.215 |
| cv_generated_prediction | estimated | 0.631 | 1.200 | 0.700 | 58 | 0.058 |
| cv_generated_prediction | estimated | 0.631 | 1.200 | 0.400 | 19 | 0.019 |
| cv_generated_prediction | estimated | 0.421 | 1.200 | 0.700 | 6 | 0.006 |
| cv_generated_prediction | known | 0.842 | 1.200 | 0.700 | 705 | 0.705 |
| cv_generated_prediction | known | 0.842 | 1.200 | 0.400 | 193 | 0.193 |
| cv_generated_prediction | known | 0.631 | 1.200 | 0.700 | 66 | 0.066 |
| cv_generated_prediction | known | 0.631 | 1.200 | 0.400 | 29 | 0.029 |
| cv_generated_prediction | known | 0.421 | 1.200 | 0.400 | 3 | 0.003 |
| cv_highdim_inference | known | 0.754 | 1.500 | 1.000 | 863 | 0.863 |
| cv_highdim_inference | known | 0.754 | 1.200 | 1.000 | 126 | 0.126 |
| cv_highdim_inference | known | 0.754 | 1.500 | 0.700 | 10 | 0.010 |
| cv_highdim_inference | known | 0.754 | 1.200 | 0.700 | 1 | 0.001 |
| cv_highdim_prediction | known | 0.889 | 1.200 | 0.700 | 838 | 0.838 |
| cv_highdim_prediction | known | 0.667 | 1.200 | 0.700 | 77 | 0.077 |
| cv_highdim_prediction | known | 0.889 | 1.200 | 0.400 | 22 | 0.022 |
| cv_highdim_prediction | known | 0.889 | 1.200 | 1.000 | 18 | 0.018 |
| cv_highdim_prediction | known | 0.889 | 0.900 | 0.700 | 11 | 0.011 |

## Paired rule comparisons

The seed identifies the same generated dataset. The comparison includes support selection, bandwidth selection, and index estimation. Width ratios use the same replications.

| Scenario | Method | Reference | Index | Kind | NMC | Coverage difference | MCSE | Width ratio |
|---|---|---|---|---|---|---|---|---|
| cv_generated_inference | oracle_hc3_log | cv_generated_prediction/sqrtlasso | known | pointwise | 1000 | 0.011 | 0.007 | 1.144 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | pointwise | 1000 | 0.016 | 0.007 | 1.141 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | pointwise | 1000 | 0.015 | 0.007 | 1.149 |
| cv_highdim_inference | oracle_hc3_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 1000 | 0.023 | 0.008 | 1.065 |
| cv_highdim_inference | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | pointwise | 1000 | 0.025 | 0.007 | 1.057 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 1000 | 0.028 | 0.007 | 1.076 |
| fixed_block_corrections | aclime_refit_block | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | 0.004 | 0.006 | 0.963 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | 0.005 | 0.005 | 0.977 |
| fixed_block_corrections | oracle | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | -0.002 | 0.007 | 0.949 |
| fixed_block_corrections | oracle_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | 0.006 | 0.006 | 0.970 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | 0.007 | 0.005 | 0.994 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | 0.003 | 0.003 | 1.006 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | pointwise | 1000 | -0.001 | 0.003 | 1.014 |
| fixed_varying_corrections | aclime_refit_block | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | 0.017 | 0.007 | 0.967 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | 0.019 | 0.006 | 0.975 |
| fixed_varying_corrections | oracle | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | 0.019 | 0.007 | 0.960 |
| fixed_varying_corrections | oracle_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | 0.024 | 0.007 | 0.971 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | 0.016 | 0.005 | 0.996 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | 0.003 | 0.003 | 1.003 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | pointwise | 1000 | -0.008 | 0.004 | 1.011 |
| cv_generated_inference | oracle_hc3_log | cv_generated_prediction/sqrtlasso | known | family | 1000 | 0.022 | 0.007 | 1.184 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | family | 1000 | -0.005 | 0.007 | 1.171 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | family | 1000 | 0.016 | 0.007 | 1.182 |
| cv_highdim_inference | oracle_hc3_log | cv_highdim_prediction/sqrtlasso | known | family | 1000 | 0.042 | 0.010 | 1.093 |
| cv_highdim_inference | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | family | 1000 | 0.016 | 0.009 | 1.071 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | family | 1000 | 0.048 | 0.010 | 1.099 |
| fixed_block_corrections | aclime_refit_block | fixed_block_corrections/scaled_refit_block | known | family | 1000 | -0.022 | 0.006 | 0.954 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | family | 1000 | 0.001 | 0.005 | 0.970 |
| fixed_block_corrections | oracle | fixed_block_corrections/scaled_refit_block | known | family | 1000 | -0.013 | 0.006 | 0.947 |
| fixed_block_corrections | oracle_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | family | 1000 | 0.005 | 0.006 | 0.971 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | fixed_block_corrections/scaled_refit_block | known | family | 1000 | -0.005 | 0.005 | 0.993 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | family | 1000 | -0.004 | 0.005 | 1.008 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | fixed_block_corrections/scaled_refit_block | known | family | 1000 | -0.009 | 0.006 | 1.016 |
| fixed_varying_corrections | aclime_refit_block | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | -0.001 | 0.006 | 0.957 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | 0.010 | 0.006 | 0.967 |
| fixed_varying_corrections | oracle | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | 0.002 | 0.005 | 0.955 |
| fixed_varying_corrections | oracle_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | 0.008 | 0.006 | 0.969 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | 0.008 | 0.004 | 0.995 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | -0.012 | 0.005 | 1.003 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | fixed_varying_corrections/scaled_refit_block | known | family | 1000 | -0.017 | 0.006 | 1.012 |
| cv_generated_inference | oracle_hc3_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 1000 | 0.017 | 0.007 | 1.144 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | pointwise | 1000 | 0.015 | 0.007 | 1.140 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 1000 | 0.017 | 0.007 | 1.149 |
| cv_generated_inference | oracle_hc3_log | cv_generated_prediction/sqrtlasso | estimated | family | 1000 | 0.031 | 0.009 | 1.183 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | family | 1000 | 0.018 | 0.008 | 1.172 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | family | 1000 | 0.028 | 0.008 | 1.183 |
