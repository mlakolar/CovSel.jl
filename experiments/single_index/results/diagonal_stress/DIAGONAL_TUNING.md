# Diagonal and CV stress

Coverage is unconditional over all requested fits. Failures count as noncoverage. Intervals are nominal 95%. Bands are joint over the three selected entries and the full grid. Wilson intervals quantify Monte Carlo uncertainty, not inferential uncertainty in a dataset.

Method suffixes: `refit` removes coefficient shrinkage; `df` refits the union support and normalizes residual-maker row variance; `hc3` adjusts final smoothing leverage; `bc` subtracts second-order inverse curvature. The components are defined in `docs/single_index/DIAGONAL_TUNING.md`. Actual tuning values are in `selected_tuning.csv`.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_boundary_inference | oracle_hc3_log | known | 1000 | 0 | 0.950 | [0.935, 0.962] | 0.812 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.924 | [0.906, 0.939] | 0.797 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.956 | [0.941, 0.967] | 0.811 |
| cv_chain_inference | oracle_hc3_log | known | 1000 | 0 | 0.957 | [0.943, 0.968] | 0.790 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.942 | [0.926, 0.955] | 0.788 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.959 | [0.945, 0.970] | 0.799 |
| cv_curvature_inference | oracle_hc3_log | known | 1000 | 0 | 0.934 | [0.917, 0.948] | 0.636 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.905 | [0.885, 0.922] | 0.625 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.941 | [0.925, 0.954] | 0.630 |
| cv_dimension_inference | oracle_hc3_log | known | 500 | 0 | 0.914 | [0.886, 0.936] | 0.734 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | 500 | 0 | 0.934 | [0.909, 0.953] | 0.726 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | 500 | 0 | 0.946 | [0.923, 0.963] | 0.735 |
| cv_grid_inference | oracle_hc3_log | known | 1000 | 0 | 0.945 | [0.929, 0.958] | 0.919 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.945 | [0.929, 0.958] | 0.912 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.957 | [0.943, 0.968] | 0.924 |
| cv_mixture_inference | oracle_hc3_log | known | 1000 | 0 | 0.752 | [0.724, 0.778] | 1.236 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.876 | [0.854, 0.895] | 1.114 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.718 | [0.689, 0.745] | 1.143 |
| cv_t8_inference | oracle_hc3_log | known | 1000 | 0 | 0.891 | [0.870, 0.909] | 0.998 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.932 | [0.915, 0.946] | 0.980 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.910 | [0.891, 0.926] | 1.001 |
| cv_varying_inference | oracle_hc3_log | known | 1000 | 0 | 0.951 | [0.936, 0.963] | 0.621 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | 1000 | 0 | 0.951 | [0.936, 0.963] | 0.615 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | 1000 | 0 | 0.959 | [0.945, 0.970] | 0.620 |
| cv_weakindex_inference | oracle_hc3_log | estimated | 500 | 3 | 0.832 | [0.797, 0.862] | 0.618 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | 500 | 3 | 0.810 | [0.773, 0.842] | 0.616 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | 500 | 3 | 0.828 | [0.792, 0.859] | 0.621 |

## Pointwise intervals at zero

| Scenario | Method | Index | Entry | Coverage | Bias | Empirical SD | RMS SE | SD/SE | Mean length | Failures |
|---|---|---|---|---|---|---|---|---|---|---|
| cv_boundary_inference | oracle_hc3_log | known | (1,1) | 0.942 | 0.019 | 0.139 | 0.135 | 1.024 | 0.528 | 0 |
| cv_boundary_inference | oracle_hc3_log | known | (1,2) | 0.963 | 0.002 | 0.094 | 0.097 | 0.969 | 0.377 | 0 |
| cv_boundary_inference | oracle_hc3_log | known | (1,38) | 0.937 | -0.001 | 0.095 | 0.090 | 1.054 | 0.352 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.951 | 0.010 | 0.137 | 0.136 | 1.005 | 0.527 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.965 | 0.002 | 0.092 | 0.097 | 0.944 | 0.379 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | (1,38) | 0.946 | -0.002 | 0.093 | 0.091 | 1.023 | 0.354 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.947 | 0.010 | 0.137 | 0.136 | 1.005 | 0.532 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.965 | 0.002 | 0.092 | 0.097 | 0.944 | 0.379 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.946 | -0.002 | 0.093 | 0.091 | 1.023 | 0.354 | 0 |
| cv_chain_inference | oracle_hc3_log | known | (1,1) | 0.955 | 0.010 | 0.141 | 0.144 | 0.982 | 0.562 | 0 |
| cv_chain_inference | oracle_hc3_log | known | (1,2) | 0.939 | -0.010 | 0.112 | 0.109 | 1.029 | 0.424 | 0 |
| cv_chain_inference | oracle_hc3_log | known | (1,38) | 0.957 | -0.002 | 0.104 | 0.106 | 0.983 | 0.410 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.947 | -0.001 | 0.141 | 0.145 | 0.975 | 0.559 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.950 | -0.003 | 0.110 | 0.110 | 1.002 | 0.427 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | (1,38) | 0.957 | -0.001 | 0.103 | 0.107 | 0.965 | 0.414 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.949 | -0.001 | 0.141 | 0.145 | 0.975 | 0.566 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | -0.003 | 0.110 | 0.110 | 1.002 | 0.427 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.957 | -0.001 | 0.103 | 0.107 | 0.965 | 0.414 | 0 |
| cv_curvature_inference | oracle_hc3_log | known | (1,1) | 0.930 | 0.016 | 0.120 | 0.116 | 1.033 | 0.454 | 0 |
| cv_curvature_inference | oracle_hc3_log | known | (1,2) | 0.940 | -0.001 | 0.083 | 0.082 | 1.009 | 0.321 | 0 |
| cv_curvature_inference | oracle_hc3_log | known | (1,78) | 0.962 | -0.001 | 0.075 | 0.077 | 0.978 | 0.300 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.933 | 0.011 | 0.120 | 0.117 | 1.030 | 0.454 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.946 | -0.000 | 0.082 | 0.083 | 0.992 | 0.323 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.968 | -0.000 | 0.074 | 0.077 | 0.962 | 0.301 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.934 | 0.011 | 0.120 | 0.117 | 1.030 | 0.457 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.946 | -0.000 | 0.082 | 0.083 | 0.992 | 0.323 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.968 | -0.000 | 0.074 | 0.077 | 0.962 | 0.301 | 0 |
| cv_dimension_inference | oracle_hc3_log | known | (1,1) | 0.942 | 0.012 | 0.138 | 0.138 | 1.000 | 0.537 | 0 |
| cv_dimension_inference | oracle_hc3_log | known | (1,2) | 0.962 | 0.009 | 0.098 | 0.098 | 0.995 | 0.382 | 0 |
| cv_dimension_inference | oracle_hc3_log | known | (1,318) | 0.932 | -0.004 | 0.097 | 0.091 | 1.065 | 0.355 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.960 | 0.006 | 0.137 | 0.139 | 0.982 | 0.537 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.958 | 0.008 | 0.097 | 0.099 | 0.980 | 0.384 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | (1,318) | 0.944 | -0.005 | 0.096 | 0.092 | 1.041 | 0.358 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.952 | 0.006 | 0.137 | 0.139 | 0.982 | 0.542 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.958 | 0.008 | 0.097 | 0.099 | 0.980 | 0.384 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | (1,318) | 0.944 | -0.005 | 0.096 | 0.092 | 1.041 | 0.358 | 0 |
| cv_grid_inference | oracle_hc3_log | known | (1,1) | 0.949 | 0.020 | 0.170 | 0.168 | 1.014 | 0.655 | 0 |
| cv_grid_inference | oracle_hc3_log | known | (1,2) | 0.950 | 0.000 | 0.130 | 0.126 | 1.025 | 0.490 | 0 |
| cv_grid_inference | oracle_hc3_log | known | (1,38) | 0.954 | -0.005 | 0.129 | 0.126 | 1.030 | 0.487 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.939 | 0.001 | 0.171 | 0.169 | 1.011 | 0.651 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.958 | 0.001 | 0.128 | 0.127 | 1.005 | 0.493 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | (1,38) | 0.960 | -0.006 | 0.126 | 0.126 | 1.004 | 0.488 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.936 | 0.001 | 0.171 | 0.169 | 1.011 | 0.659 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.958 | 0.001 | 0.128 | 0.127 | 1.005 | 0.493 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.960 | -0.006 | 0.126 | 0.126 | 1.004 | 0.488 | 0 |
| cv_mixture_inference | oracle_hc3_log | known | (1,1) | 0.888 | 0.054 | 0.269 | 0.246 | 1.091 | 0.968 | 0 |
| cv_mixture_inference | oracle_hc3_log | known | (1,2) | 0.920 | -0.001 | 0.164 | 0.153 | 1.067 | 0.585 | 0 |
| cv_mixture_inference | oracle_hc3_log | known | (1,78) | 0.913 | 0.012 | 0.161 | 0.146 | 1.106 | 0.554 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.853 | 0.175 | 0.249 | 0.222 | 1.126 | 0.847 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.938 | -0.007 | 0.151 | 0.147 | 1.023 | 0.567 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.924 | 0.009 | 0.150 | 0.138 | 1.089 | 0.530 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.794 | 0.175 | 0.249 | 0.222 | 1.126 | 0.865 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.938 | -0.007 | 0.151 | 0.147 | 1.023 | 0.567 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.924 | 0.009 | 0.150 | 0.138 | 1.089 | 0.530 | 0 |
| cv_t8_inference | oracle_hc3_log | known | (1,1) | 0.923 | 0.050 | 0.214 | 0.196 | 1.093 | 0.763 | 0 |
| cv_t8_inference | oracle_hc3_log | known | (1,2) | 0.950 | 0.004 | 0.133 | 0.131 | 1.017 | 0.503 | 0 |
| cv_t8_inference | oracle_hc3_log | known | (1,78) | 0.924 | -0.007 | 0.134 | 0.123 | 1.094 | 0.472 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.934 | 0.062 | 0.213 | 0.196 | 1.090 | 0.750 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.953 | 0.005 | 0.132 | 0.132 | 0.994 | 0.510 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.943 | -0.007 | 0.130 | 0.124 | 1.051 | 0.478 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.917 | 0.062 | 0.213 | 0.196 | 1.090 | 0.763 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.953 | 0.005 | 0.132 | 0.132 | 0.994 | 0.510 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.943 | -0.007 | 0.130 | 0.124 | 1.051 | 0.478 | 0 |
| cv_varying_inference | oracle_hc3_log | known | (1,1) | 0.962 | 0.013 | 0.109 | 0.116 | 0.942 | 0.452 | 0 |
| cv_varying_inference | oracle_hc3_log | known | (1,2) | 0.953 | 0.005 | 0.081 | 0.082 | 0.981 | 0.320 | 0 |
| cv_varying_inference | oracle_hc3_log | known | (1,78) | 0.947 | 0.004 | 0.078 | 0.077 | 1.015 | 0.299 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.962 | 0.006 | 0.108 | 0.116 | 0.927 | 0.451 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.953 | 0.005 | 0.080 | 0.082 | 0.964 | 0.321 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.956 | 0.003 | 0.077 | 0.077 | 0.997 | 0.301 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.963 | 0.006 | 0.108 | 0.116 | 0.927 | 0.454 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.953 | 0.005 | 0.080 | 0.082 | 0.964 | 0.321 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.956 | 0.003 | 0.077 | 0.077 | 0.997 | 0.301 | 0 |
| cv_weakindex_inference | oracle_hc3_log | estimated | (1,1) | 0.944 | 0.008 | 0.117 | 0.116 | 1.004 | 0.454 | 3 |
| cv_weakindex_inference | oracle_hc3_log | estimated | (1,2) | 0.938 | 0.001 | 0.090 | 0.085 | 1.055 | 0.332 | 3 |
| cv_weakindex_inference | oracle_hc3_log | estimated | (1,78) | 0.950 | -0.002 | 0.074 | 0.077 | 0.958 | 0.302 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | (1,1) | 0.954 | 0.000 | 0.118 | 0.117 | 1.009 | 0.453 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | (1,2) | 0.938 | 0.001 | 0.089 | 0.086 | 1.039 | 0.333 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | (1,78) | 0.958 | -0.003 | 0.073 | 0.078 | 0.946 | 0.303 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.948 | 0.000 | 0.118 | 0.117 | 1.009 | 0.456 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.938 | 0.001 | 0.089 | 0.086 | 1.039 | 0.333 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.958 | -0.003 | 0.073 | 0.078 | 0.946 | 0.303 | 3 |

## Curvewise bands

| Scenario | Method | Index | Entry | Coverage | Mean width | Failures |
|---|---|---|---|---|---|---|
| cv_boundary_inference | oracle_hc3_log | known | (1,1) | 0.929 | 0.916 | 0 |
| cv_boundary_inference | oracle_hc3_log | known | (1,2) | 0.951 | 0.653 | 0 |
| cv_boundary_inference | oracle_hc3_log | known | (1,38) | 0.954 | 0.592 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.892 | 0.880 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.960 | 0.659 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc | known | (1,38) | 0.963 | 0.591 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.945 | 0.909 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.960 | 0.659 | 0 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.963 | 0.591 | 0 |
| cv_chain_inference | oracle_hc3_log | known | (1,1) | 0.932 | 0.845 | 0 |
| cv_chain_inference | oracle_hc3_log | known | (1,2) | 0.957 | 0.633 | 0 |
| cv_chain_inference | oracle_hc3_log | known | (1,38) | 0.959 | 0.612 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.918 | 0.833 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.964 | 0.639 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc | known | (1,38) | 0.972 | 0.618 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.946 | 0.855 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.964 | 0.639 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.972 | 0.618 | 0 |
| cv_curvature_inference | oracle_hc3_log | known | (1,1) | 0.927 | 0.715 | 0 |
| cv_curvature_inference | oracle_hc3_log | known | (1,2) | 0.929 | 0.513 | 0 |
| cv_curvature_inference | oracle_hc3_log | known | (1,78) | 0.948 | 0.460 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.895 | 0.689 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.935 | 0.515 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.951 | 0.456 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.932 | 0.700 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.935 | 0.515 | 0 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.951 | 0.456 | 0 |
| cv_dimension_inference | oracle_hc3_log | known | (1,1) | 0.932 | 0.823 | 0 |
| cv_dimension_inference | oracle_hc3_log | known | (1,2) | 0.960 | 0.585 | 0 |
| cv_dimension_inference | oracle_hc3_log | known | (1,318) | 0.958 | 0.534 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.900 | 0.802 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.958 | 0.590 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc | known | (1,318) | 0.962 | 0.535 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.918 | 0.819 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.958 | 0.590 | 0 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | known | (1,318) | 0.962 | 0.535 | 0 |
| cv_grid_inference | oracle_hc3_log | known | (1,1) | 0.930 | 0.980 | 0 |
| cv_grid_inference | oracle_hc3_log | known | (1,2) | 0.945 | 0.726 | 0 |
| cv_grid_inference | oracle_hc3_log | known | (1,38) | 0.963 | 0.720 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.919 | 0.962 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.957 | 0.731 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc | known | (1,38) | 0.978 | 0.724 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.942 | 0.987 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.957 | 0.731 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.978 | 0.724 | 0 |
| cv_mixture_inference | oracle_hc3_log | known | (1,1) | 0.738 | 1.506 | 0 |
| cv_mixture_inference | oracle_hc3_log | known | (1,2) | 0.881 | 0.916 | 0 |
| cv_mixture_inference | oracle_hc3_log | known | (1,78) | 0.864 | 0.834 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.841 | 1.271 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.924 | 0.888 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.919 | 0.801 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.641 | 1.331 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.924 | 0.888 | 0 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.919 | 0.801 | 0 |
| cv_t8_inference | oracle_hc3_log | known | (1,1) | 0.897 | 1.165 | 0 |
| cv_t8_inference | oracle_hc3_log | known | (1,2) | 0.932 | 0.771 | 0 |
| cv_t8_inference | oracle_hc3_log | known | (1,78) | 0.929 | 0.700 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.916 | 1.107 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.927 | 0.785 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.956 | 0.708 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.899 | 1.151 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.927 | 0.785 | 0 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.956 | 0.708 | 0 |
| cv_varying_inference | oracle_hc3_log | known | (1,1) | 0.951 | 0.697 | 0 |
| cv_varying_inference | oracle_hc3_log | known | (1,2) | 0.949 | 0.496 | 0 |
| cv_varying_inference | oracle_hc3_log | known | (1,78) | 0.956 | 0.453 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | (1,1) | 0.932 | 0.681 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | (1,2) | 0.953 | 0.499 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc | known | (1,78) | 0.956 | 0.453 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.958 | 0.692 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.953 | 0.499 | 0 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.956 | 0.453 | 0 |
| cv_weakindex_inference | oracle_hc3_log | estimated | (1,1) | 0.934 | 0.688 | 3 |
| cv_weakindex_inference | oracle_hc3_log | estimated | (1,2) | 0.734 | 0.498 | 3 |
| cv_weakindex_inference | oracle_hc3_log | estimated | (1,78) | 0.950 | 0.453 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | (1,1) | 0.918 | 0.681 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | (1,2) | 0.732 | 0.500 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc | estimated | (1,78) | 0.948 | 0.455 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.932 | 0.691 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.732 | 0.500 | 3 |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.948 | 0.455 | 3 |

## Selected tuning

Frequencies are descriptive. No truth or coverage entered selection.

| Scenario | Index | Final h | Pilot h | Penalty multiplier | Count | Fraction |
|---|---|---|---|---|---|---|
| cv_boundary_inference | known | 0.696 | 1.500 | 1.000 | 804 | 0.804 |
| cv_boundary_inference | known | 0.696 | 1.200 | 1.000 | 182 | 0.182 |
| cv_boundary_inference | known | 0.696 | 1.500 | 0.700 | 11 | 0.011 |
| cv_boundary_inference | known | 0.696 | 1.200 | 0.700 | 3 | 0.003 |
| cv_chain_inference | known | 0.696 | 1.500 | 1.000 | 850 | 0.850 |
| cv_chain_inference | known | 0.696 | 1.200 | 1.000 | 117 | 0.117 |
| cv_chain_inference | known | 0.696 | 1.500 | 0.700 | 30 | 0.030 |
| cv_chain_inference | known | 0.696 | 1.200 | 0.700 | 3 | 0.003 |
| cv_curvature_inference | known | 0.605 | 1.500 | 1.000 | 678 | 0.678 |
| cv_curvature_inference | known | 0.605 | 1.200 | 1.000 | 320 | 0.320 |
| cv_curvature_inference | known | 0.605 | 1.500 | 0.700 | 1 | 0.001 |
| cv_curvature_inference | known | 0.605 | 1.200 | 0.700 | 1 | 0.001 |
| cv_dimension_inference | known | 0.657 | 1.500 | 1.000 | 460 | 0.920 |
| cv_dimension_inference | known | 0.657 | 1.200 | 1.000 | 40 | 0.080 |
| cv_grid_inference | known | 0.696 | 1.500 | 1.000 | 685 | 0.685 |
| cv_grid_inference | known | 0.696 | 1.200 | 1.000 | 297 | 0.297 |
| cv_grid_inference | known | 0.696 | 1.500 | 0.700 | 17 | 0.017 |
| cv_grid_inference | known | 0.696 | 1.200 | 0.700 | 1 | 0.001 |
| cv_mixture_inference | known | 0.696 | 1.500 | 1.000 | 691 | 0.691 |
| cv_mixture_inference | known | 0.696 | 1.200 | 1.000 | 220 | 0.220 |
| cv_mixture_inference | known | 0.696 | 1.500 | 0.700 | 62 | 0.062 |
| cv_mixture_inference | known | 0.696 | 1.200 | 0.700 | 19 | 0.019 |
| cv_mixture_inference | known | 0.696 | 1.500 | 0.400 | 7 | 0.007 |
| cv_t8_inference | known | 0.696 | 1.500 | 1.000 | 860 | 0.860 |
| cv_t8_inference | known | 0.696 | 1.200 | 1.000 | 126 | 0.126 |
| cv_t8_inference | known | 0.696 | 1.500 | 0.700 | 13 | 0.013 |
| cv_t8_inference | known | 0.696 | 1.200 | 0.700 | 1 | 0.001 |
| cv_varying_inference | known | 0.605 | 1.500 | 1.000 | 741 | 0.741 |
| cv_varying_inference | known | 0.605 | 1.200 | 1.000 | 255 | 0.255 |
| cv_varying_inference | known | 0.605 | 1.500 | 0.700 | 4 | 0.004 |
| cv_weakindex_inference | estimated | 0.605 | 1.500 | 1.000 | 413 | 0.826 |
| cv_weakindex_inference | estimated | 0.605 | 1.200 | 1.000 | 84 | 0.168 |
| cv_weakindex_inference | estimated | 0.605 | 0.969 | 0.700 | 3 | 0.006 |

## Paired rule comparisons

The seed identifies the same generated dataset. The comparison includes support selection, bandwidth selection, and index estimation. Width ratios use the same replications.

| Scenario | Method | Reference | Index | Kind | NMC | Coverage difference | MCSE | Width ratio |
|---|---|---|---|---|---|---|---|---|
| cv_boundary_inference | oracle_hc3_log | cv_boundary_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.009 | 0.006 | 1.002 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | cv_boundary_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.004 | 0.005 | 1.009 |
| cv_chain_inference | oracle_hc3_log | cv_chain_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.008 | 0.006 | 1.004 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | cv_chain_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.002 | 0.005 | 1.012 |
| cv_curvature_inference | oracle_hc3_log | cv_curvature_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.003 | 0.006 | 1.002 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | cv_curvature_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.001 | 0.005 | 1.007 |
| cv_dimension_inference | oracle_hc3_log | cv_dimension_inference/sqrt_refit_df_hc3_bc | known | pointwise | 500 | -0.018 | 0.010 | 1.000 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | cv_dimension_inference/sqrt_refit_df_hc3_bc | known | pointwise | 500 | -0.008 | 0.008 | 1.010 |
| cv_grid_inference | oracle_hc3_log | cv_grid_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.010 | 0.007 | 1.007 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | cv_grid_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.003 | 0.005 | 1.012 |
| cv_mixture_inference | oracle_hc3_log | cv_mixture_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.035 | 0.010 | 1.143 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | cv_mixture_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.059 | 0.008 | 1.021 |
| cv_t8_inference | oracle_hc3_log | cv_t8_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.011 | 0.007 | 1.018 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | cv_t8_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | -0.017 | 0.007 | 1.018 |
| cv_varying_inference | oracle_hc3_log | cv_varying_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.000 | 0.005 | 1.002 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | cv_varying_inference/sqrt_refit_df_hc3_bc | known | pointwise | 1000 | 0.001 | 0.005 | 1.007 |
| cv_boundary_inference | oracle_hc3_log | cv_boundary_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.026 | 0.009 | 1.018 |
| cv_boundary_inference | sqrt_refit_df_hc3_bc_log | cv_boundary_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.032 | 0.007 | 1.017 |
| cv_chain_inference | oracle_hc3_log | cv_chain_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.015 | 0.007 | 1.003 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | cv_chain_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.017 | 0.006 | 1.014 |
| cv_curvature_inference | oracle_hc3_log | cv_curvature_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.029 | 0.009 | 1.017 |
| cv_curvature_inference | sqrt_refit_df_hc3_bc_log | cv_curvature_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.036 | 0.007 | 1.008 |
| cv_dimension_inference | oracle_hc3_log | cv_dimension_inference/sqrt_refit_df_hc3_bc | known | family | 500 | -0.020 | 0.014 | 1.011 |
| cv_dimension_inference | sqrt_refit_df_hc3_bc_log | cv_dimension_inference/sqrt_refit_df_hc3_bc | known | family | 500 | 0.012 | 0.011 | 1.011 |
| cv_grid_inference | oracle_hc3_log | cv_grid_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.000 | 0.007 | 1.007 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | cv_grid_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.012 | 0.005 | 1.014 |
| cv_mixture_inference | oracle_hc3_log | cv_mixture_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | -0.124 | 0.015 | 1.110 |
| cv_mixture_inference | sqrt_refit_df_hc3_bc_log | cv_mixture_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | -0.158 | 0.012 | 1.026 |
| cv_t8_inference | oracle_hc3_log | cv_t8_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | -0.041 | 0.010 | 1.019 |
| cv_t8_inference | sqrt_refit_df_hc3_bc_log | cv_t8_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | -0.022 | 0.007 | 1.022 |
| cv_varying_inference | oracle_hc3_log | cv_varying_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.000 | 0.007 | 1.010 |
| cv_varying_inference | sqrt_refit_df_hc3_bc_log | cv_varying_inference/sqrt_refit_df_hc3_bc | known | family | 1000 | 0.008 | 0.004 | 1.008 |
| cv_weakindex_inference | oracle_hc3_log | cv_weakindex_inference/sqrt_refit_df_hc3_bc | estimated | pointwise | 500 | -0.010 | 0.009 | nan |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | cv_weakindex_inference/sqrt_refit_df_hc3_bc | estimated | pointwise | 500 | -0.006 | 0.007 | nan |
| cv_weakindex_inference | oracle_hc3_log | cv_weakindex_inference/sqrt_refit_df_hc3_bc | estimated | family | 500 | 0.022 | 0.011 | nan |
| cv_weakindex_inference | sqrt_refit_df_hc3_bc_log | cv_weakindex_inference/sqrt_refit_df_hc3_bc | estimated | family | 500 | 0.018 | 0.007 | nan |
