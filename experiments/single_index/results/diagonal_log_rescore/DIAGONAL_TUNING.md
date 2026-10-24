# Diagonal and CV development-log

Coverage is unconditional over all requested fits. Failures count as noncoverage. Intervals are nominal 95%. Bands are joint over the three selected entries and the full grid. Wilson intervals quantify Monte Carlo uncertainty, not inferential uncertainty in a dataset.

Method suffixes: `refit` removes coefficient shrinkage; `df` refits the union support and normalizes residual-maker row variance; `hc3` adjusts final smoothing leverage; `bc` subtracts second-order inverse curvature. The components are defined in `docs/single_index/DIAGONAL_TUNING.md`. Actual tuning values are in `selected_tuning.csv`.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_chain_inference | sqrt_refit_block_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.788 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.794 |
| cv_generated_inference | sqrt_refit_block_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.621 |
| cv_generated_inference | sqrt_refit_block_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.619 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | 200 | 0 | 0.930 | [0.886, 0.958] | 0.622 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.620 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.621 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.620 |
| cv_generated_inference | sqrt_refit_df_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.618 |
| cv_generated_inference | sqrt_refit_df_log | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.617 |
| cv_generated_inference | sqrtlasso_log | estimated | 200 | 0 | 0.905 | [0.856, 0.938] | 0.604 |
| cv_generated_inference | sqrtlasso_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.602 |
| cv_generated_onese | sqrt_refit_block_log | estimated | 200 | 0 | 0.910 | [0.862, 0.942] | 0.425 |
| cv_generated_onese | sqrt_refit_block_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.424 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | 200 | 0 | 0.920 | [0.874, 0.950] | 0.424 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.960 | [0.923, 0.980] | 0.423 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | 200 | 0 | 0.915 | [0.868, 0.946] | 0.424 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.423 |
| cv_generated_onese | sqrt_refit_df_log | estimated | 200 | 0 | 0.910 | [0.862, 0.942] | 0.422 |
| cv_generated_onese | sqrt_refit_df_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.421 |
| cv_generated_onese | sqrtlasso_log | estimated | 200 | 0 | 0.755 | [0.691, 0.809] | 0.405 |
| cv_generated_onese | sqrtlasso_log | known | 200 | 0 | 0.830 | [0.772, 0.876] | 0.403 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | 200 | 0 | 0.840 | [0.783, 0.884] | 0.557 |
| cv_generated_prediction | sqrt_refit_block_log | known | 200 | 0 | 0.760 | [0.696, 0.814] | 0.561 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | 200 | 0 | 0.910 | [0.862, 0.942] | 0.546 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.875 | [0.822, 0.914] | 0.549 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | 200 | 0 | 0.900 | [0.851, 0.934] | 0.546 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.850 | [0.794, 0.893] | 0.549 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | 200 | 0 | 0.900 | [0.851, 0.934] | 0.543 |
| cv_generated_prediction | sqrt_refit_df_log | known | 200 | 0 | 0.850 | [0.794, 0.893] | 0.546 |
| cv_generated_prediction | sqrtlasso_log | estimated | 200 | 0 | 0.915 | [0.868, 0.946] | 0.524 |
| cv_generated_prediction | sqrtlasso_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.527 |
| cv_grid_inference | sqrt_refit_block_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.907 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.914 |
| cv_highdim_inference | sqrt_refit_block_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.972 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.978 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.977 |
| cv_highdim_inference | sqrt_refit_df_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.962 |
| cv_highdim_inference | sqrtlasso_log | known | 200 | 0 | 0.880 | [0.828, 0.918] | 0.892 |
| cv_highdim_onese | sqrt_refit_block_log | known | 200 | 0 | 0.905 | [0.856, 0.938] | 0.739 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.742 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.742 |
| cv_highdim_onese | sqrt_refit_df_log | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.735 |
| cv_highdim_onese | sqrtlasso_log | known | 200 | 0 | 0.760 | [0.696, 0.814] | 0.674 |
| cv_highdim_prediction | sqrt_refit_block_log | known | 200 | 0 | 0.635 | [0.566, 0.699] | 1.136 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.855 | [0.800, 0.897] | 1.081 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.805 | [0.745, 0.854] | 1.080 |
| cv_highdim_prediction | sqrt_refit_df_log | known | 200 | 0 | 0.785 | [0.723, 0.836] | 1.065 |
| cv_highdim_prediction | sqrtlasso_log | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.902 |
| fixed_block_corrections | aclime_refit_block_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.828 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.832 |
| fixed_block_corrections | oracle_hc3_bc_log | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.833 |
| fixed_block_corrections | oracle_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.823 |
| fixed_block_corrections | scaled_refit_block_log | known | 200 | 0 | 0.870 | [0.816, 0.910] | 0.867 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.863 |
| fixed_block_corrections | sqrt_refit_block_log | known | 200 | 0 | 0.855 | [0.800, 0.897] | 0.880 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.930 | [0.886, 0.958] | 0.869 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.885 | [0.833, 0.922] | 0.869 |
| fixed_block_corrections | sqrt_refit_df_log | known | 200 | 0 | 0.875 | [0.822, 0.914] | 0.859 |
| fixed_block_corrections | sqrtlasso_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.820 |
| fixed_varying_corrections | aclime_refit_block_log | known | 200 | 0 | 0.975 | [0.943, 0.989] | 0.617 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | 200 | 0 | 0.975 | [0.943, 0.989] | 0.618 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | 200 | 0 | 0.960 | [0.923, 0.980] | 0.618 |
| fixed_varying_corrections | oracle_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.614 |
| fixed_varying_corrections | scaled_refit_block_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.644 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | 200 | 0 | 0.975 | [0.943, 0.989] | 0.641 |
| fixed_varying_corrections | sqrt_refit_block_log | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.652 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.646 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.646 |
| fixed_varying_corrections | sqrt_refit_df_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.642 |
| fixed_varying_corrections | sqrtlasso_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.615 |

## Pointwise intervals at zero

| Scenario | Method | Index | Entry | Coverage | Bias | Empirical SD | RMS SE | SD/SE | Mean length | Failures |
|---|---|---|---|---|---|---|---|---|---|---|
| cv_chain_inference | sqrt_refit_block_log | known | (1,1) | 0.945 | 0.012 | 0.148 | 0.142 | 1.046 | 0.554 | 0 |
| cv_chain_inference | sqrt_refit_block_log | known | (1,2) | 0.920 | 0.008 | 0.110 | 0.108 | 1.021 | 0.418 | 0 |
| cv_chain_inference | sqrt_refit_block_log | known | (1,38) | 0.935 | -0.003 | 0.111 | 0.106 | 1.048 | 0.412 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | -0.012 | 0.144 | 0.142 | 1.012 | 0.557 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.925 | 0.016 | 0.106 | 0.108 | 0.979 | 0.420 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.950 | -0.002 | 0.107 | 0.106 | 1.008 | 0.412 | 0 |
| cv_generated_inference | sqrt_refit_block_log | estimated | (1,1) | 0.920 | 0.017 | 0.118 | 0.117 | 1.007 | 0.458 | 0 |
| cv_generated_inference | sqrt_refit_block_log | estimated | (1,2) | 0.960 | -0.002 | 0.082 | 0.084 | 0.979 | 0.328 | 0 |
| cv_generated_inference | sqrt_refit_block_log | estimated | (1,78) | 0.935 | 0.001 | 0.083 | 0.077 | 1.081 | 0.300 | 0 |
| cv_generated_inference | sqrt_refit_block_log | known | (1,1) | 0.920 | 0.009 | 0.118 | 0.114 | 1.040 | 0.445 | 0 |
| cv_generated_inference | sqrt_refit_block_log | known | (1,2) | 0.960 | -0.002 | 0.078 | 0.082 | 0.951 | 0.320 | 0 |
| cv_generated_inference | sqrt_refit_block_log | known | (1,78) | 0.910 | -0.003 | 0.080 | 0.076 | 1.052 | 0.297 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.925 | -0.000 | 0.116 | 0.117 | 0.988 | 0.458 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.965 | -0.002 | 0.081 | 0.084 | 0.963 | 0.329 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.935 | 0.001 | 0.082 | 0.077 | 1.057 | 0.301 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.925 | -0.007 | 0.116 | 0.114 | 1.021 | 0.445 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.960 | -0.002 | 0.077 | 0.082 | 0.933 | 0.320 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.920 | -0.003 | 0.079 | 0.076 | 1.037 | 0.298 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.930 | 0.012 | 0.117 | 0.117 | 0.999 | 0.458 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.965 | -0.002 | 0.082 | 0.084 | 0.975 | 0.329 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.935 | 0.001 | 0.083 | 0.077 | 1.073 | 0.301 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | (1,1) | 0.925 | 0.004 | 0.118 | 0.114 | 1.033 | 0.445 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | (1,2) | 0.960 | -0.002 | 0.078 | 0.082 | 0.947 | 0.320 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | (1,78) | 0.910 | -0.003 | 0.081 | 0.076 | 1.055 | 0.298 | 0 |
| cv_generated_inference | sqrt_refit_df_log | estimated | (1,1) | 0.925 | 0.012 | 0.117 | 0.117 | 1.005 | 0.455 | 0 |
| cv_generated_inference | sqrt_refit_df_log | estimated | (1,2) | 0.960 | -0.002 | 0.082 | 0.084 | 0.981 | 0.327 | 0 |
| cv_generated_inference | sqrt_refit_df_log | estimated | (1,78) | 0.930 | 0.001 | 0.083 | 0.077 | 1.079 | 0.300 | 0 |
| cv_generated_inference | sqrt_refit_df_log | known | (1,1) | 0.925 | 0.004 | 0.118 | 0.113 | 1.039 | 0.443 | 0 |
| cv_generated_inference | sqrt_refit_df_log | known | (1,2) | 0.960 | -0.002 | 0.078 | 0.082 | 0.952 | 0.319 | 0 |
| cv_generated_inference | sqrt_refit_df_log | known | (1,78) | 0.910 | -0.003 | 0.081 | 0.076 | 1.060 | 0.296 | 0 |
| cv_generated_inference | sqrtlasso_log | estimated | (1,1) | 0.920 | -0.015 | 0.117 | 0.114 | 1.025 | 0.445 | 0 |
| cv_generated_inference | sqrtlasso_log | estimated | (1,2) | 0.940 | -0.028 | 0.080 | 0.082 | 0.974 | 0.320 | 0 |
| cv_generated_inference | sqrtlasso_log | estimated | (1,78) | 0.930 | 0.001 | 0.081 | 0.076 | 1.067 | 0.295 | 0 |
| cv_generated_inference | sqrtlasso_log | known | (1,1) | 0.940 | -0.022 | 0.113 | 0.111 | 1.023 | 0.433 | 0 |
| cv_generated_inference | sqrtlasso_log | known | (1,2) | 0.950 | -0.026 | 0.075 | 0.080 | 0.942 | 0.312 | 0 |
| cv_generated_inference | sqrtlasso_log | known | (1,78) | 0.915 | -0.003 | 0.079 | 0.075 | 1.054 | 0.293 | 0 |
| cv_generated_onese | sqrt_refit_block_log | estimated | (1,1) | 0.965 | 0.010 | 0.084 | 0.084 | 0.994 | 0.330 | 0 |
| cv_generated_onese | sqrt_refit_block_log | estimated | (1,2) | 0.965 | -0.003 | 0.061 | 0.061 | 0.992 | 0.239 | 0 |
| cv_generated_onese | sqrt_refit_block_log | estimated | (1,78) | 0.975 | 0.001 | 0.054 | 0.056 | 0.962 | 0.218 | 0 |
| cv_generated_onese | sqrt_refit_block_log | known | (1,1) | 0.925 | 0.015 | 0.090 | 0.084 | 1.072 | 0.328 | 0 |
| cv_generated_onese | sqrt_refit_block_log | known | (1,2) | 0.960 | -0.001 | 0.058 | 0.060 | 0.962 | 0.235 | 0 |
| cv_generated_onese | sqrt_refit_block_log | known | (1,78) | 0.945 | 0.000 | 0.056 | 0.056 | 1.008 | 0.217 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.965 | -0.002 | 0.083 | 0.084 | 0.984 | 0.329 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.970 | -0.003 | 0.060 | 0.061 | 0.982 | 0.238 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.980 | 0.001 | 0.053 | 0.056 | 0.953 | 0.218 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.935 | 0.002 | 0.088 | 0.084 | 1.059 | 0.327 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.965 | -0.001 | 0.057 | 0.060 | 0.951 | 0.234 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.945 | 0.000 | 0.056 | 0.056 | 1.000 | 0.217 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.965 | 0.004 | 0.083 | 0.084 | 0.989 | 0.329 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.965 | -0.003 | 0.060 | 0.061 | 0.990 | 0.238 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.975 | 0.001 | 0.054 | 0.056 | 0.961 | 0.218 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | (1,1) | 0.930 | 0.008 | 0.089 | 0.084 | 1.065 | 0.327 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | (1,2) | 0.960 | -0.001 | 0.057 | 0.060 | 0.959 | 0.234 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | (1,78) | 0.945 | 0.000 | 0.056 | 0.056 | 1.009 | 0.217 | 0 |
| cv_generated_onese | sqrt_refit_df_log | estimated | (1,1) | 0.965 | 0.004 | 0.083 | 0.084 | 0.992 | 0.328 | 0 |
| cv_generated_onese | sqrt_refit_df_log | estimated | (1,2) | 0.965 | -0.003 | 0.060 | 0.061 | 0.993 | 0.238 | 0 |
| cv_generated_onese | sqrt_refit_df_log | estimated | (1,78) | 0.975 | 0.001 | 0.054 | 0.056 | 0.964 | 0.217 | 0 |
| cv_generated_onese | sqrt_refit_df_log | known | (1,1) | 0.930 | 0.008 | 0.089 | 0.083 | 1.068 | 0.326 | 0 |
| cv_generated_onese | sqrt_refit_df_log | known | (1,2) | 0.960 | -0.001 | 0.057 | 0.060 | 0.962 | 0.234 | 0 |
| cv_generated_onese | sqrt_refit_df_log | known | (1,78) | 0.945 | 0.000 | 0.056 | 0.055 | 1.012 | 0.216 | 0 |
| cv_generated_onese | sqrtlasso_log | estimated | (1,1) | 0.915 | -0.038 | 0.079 | 0.081 | 0.980 | 0.316 | 0 |
| cv_generated_onese | sqrtlasso_log | estimated | (1,2) | 0.875 | -0.045 | 0.057 | 0.059 | 0.982 | 0.229 | 0 |
| cv_generated_onese | sqrtlasso_log | estimated | (1,78) | 0.965 | 0.002 | 0.052 | 0.054 | 0.956 | 0.213 | 0 |
| cv_generated_onese | sqrtlasso_log | known | (1,1) | 0.925 | -0.036 | 0.084 | 0.080 | 1.050 | 0.313 | 0 |
| cv_generated_onese | sqrtlasso_log | known | (1,2) | 0.890 | -0.043 | 0.054 | 0.058 | 0.943 | 0.225 | 0 |
| cv_generated_onese | sqrtlasso_log | known | (1,78) | 0.935 | 0.000 | 0.055 | 0.054 | 1.020 | 0.212 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | (1,1) | 0.855 | 0.077 | 0.130 | 0.107 | 1.217 | 0.417 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | (1,2) | 0.955 | -0.003 | 0.074 | 0.077 | 0.961 | 0.300 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | (1,78) | 0.930 | 0.002 | 0.076 | 0.071 | 1.074 | 0.274 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | known | (1,1) | 0.840 | 0.070 | 0.131 | 0.105 | 1.251 | 0.409 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | known | (1,2) | 0.945 | -0.002 | 0.077 | 0.075 | 1.020 | 0.293 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | known | (1,78) | 0.925 | -0.002 | 0.074 | 0.070 | 1.064 | 0.273 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.890 | 0.045 | 0.118 | 0.105 | 1.119 | 0.410 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.960 | -0.004 | 0.071 | 0.075 | 0.945 | 0.293 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.930 | 0.002 | 0.073 | 0.069 | 1.058 | 0.270 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.885 | 0.038 | 0.119 | 0.103 | 1.157 | 0.402 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | -0.003 | 0.074 | 0.074 | 1.004 | 0.287 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.930 | -0.002 | 0.072 | 0.069 | 1.055 | 0.267 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.870 | 0.054 | 0.119 | 0.105 | 1.130 | 0.410 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.960 | -0.004 | 0.072 | 0.075 | 0.955 | 0.293 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.930 | 0.002 | 0.074 | 0.069 | 1.071 | 0.270 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | (1,1) | 0.880 | 0.047 | 0.121 | 0.103 | 1.168 | 0.402 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | -0.003 | 0.075 | 0.074 | 1.016 | 0.287 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | (1,78) | 0.920 | -0.002 | 0.073 | 0.069 | 1.068 | 0.267 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | (1,1) | 0.870 | 0.054 | 0.119 | 0.105 | 1.135 | 0.408 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | (1,2) | 0.960 | -0.004 | 0.072 | 0.075 | 0.959 | 0.292 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | (1,78) | 0.925 | 0.002 | 0.074 | 0.069 | 1.075 | 0.268 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | known | (1,1) | 0.880 | 0.047 | 0.121 | 0.103 | 1.172 | 0.401 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | known | (1,2) | 0.950 | -0.003 | 0.075 | 0.073 | 1.020 | 0.286 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | known | (1,78) | 0.915 | -0.002 | 0.073 | 0.068 | 1.072 | 0.266 | 0 |
| cv_generated_prediction | sqrtlasso_log | estimated | (1,1) | 0.920 | 0.021 | 0.114 | 0.102 | 1.123 | 0.396 | 0 |
| cv_generated_prediction | sqrtlasso_log | estimated | (1,2) | 0.960 | -0.016 | 0.071 | 0.073 | 0.977 | 0.285 | 0 |
| cv_generated_prediction | sqrtlasso_log | estimated | (1,78) | 0.945 | 0.002 | 0.071 | 0.067 | 1.063 | 0.262 | 0 |
| cv_generated_prediction | sqrtlasso_log | known | (1,1) | 0.905 | 0.018 | 0.116 | 0.100 | 1.158 | 0.390 | 0 |
| cv_generated_prediction | sqrtlasso_log | known | (1,2) | 0.945 | -0.014 | 0.071 | 0.072 | 0.994 | 0.280 | 0 |
| cv_generated_prediction | sqrtlasso_log | known | (1,78) | 0.925 | -0.002 | 0.070 | 0.067 | 1.045 | 0.261 | 0 |
| cv_grid_inference | sqrt_refit_block_log | known | (1,1) | 0.960 | 0.010 | 0.164 | 0.165 | 0.996 | 0.643 | 0 |
| cv_grid_inference | sqrt_refit_block_log | known | (1,2) | 0.930 | 0.002 | 0.132 | 0.124 | 1.060 | 0.483 | 0 |
| cv_grid_inference | sqrt_refit_block_log | known | (1,38) | 0.945 | 0.006 | 0.128 | 0.123 | 1.036 | 0.477 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.955 | -0.014 | 0.161 | 0.166 | 0.968 | 0.648 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.945 | 0.006 | 0.127 | 0.125 | 1.017 | 0.486 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.955 | 0.006 | 0.124 | 0.124 | 1.005 | 0.480 | 0 |
| cv_highdim_inference | sqrt_refit_block_log | known | (1,1) | 0.910 | 0.059 | 0.183 | 0.184 | 0.996 | 0.717 | 0 |
| cv_highdim_inference | sqrt_refit_block_log | known | (1,158) | 0.960 | 0.002 | 0.128 | 0.126 | 1.013 | 0.487 | 0 |
| cv_highdim_inference | sqrt_refit_block_log | known | (1,2) | 0.900 | -0.008 | 0.166 | 0.134 | 1.245 | 0.516 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | 0.016 | 0.175 | 0.184 | 0.949 | 0.718 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.970 | 0.002 | 0.121 | 0.127 | 0.946 | 0.490 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.910 | -0.005 | 0.161 | 0.134 | 1.202 | 0.519 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | (1,1) | 0.925 | 0.044 | 0.179 | 0.184 | 0.973 | 0.718 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | (1,158) | 0.965 | 0.002 | 0.126 | 0.127 | 0.992 | 0.490 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | (1,2) | 0.895 | -0.005 | 0.168 | 0.134 | 1.249 | 0.519 | 0 |
| cv_highdim_inference | sqrt_refit_df_log | known | (1,1) | 0.925 | 0.044 | 0.179 | 0.182 | 0.986 | 0.708 | 0 |
| cv_highdim_inference | sqrt_refit_df_log | known | (1,158) | 0.960 | 0.002 | 0.126 | 0.126 | 1.005 | 0.484 | 0 |
| cv_highdim_inference | sqrt_refit_df_log | known | (1,2) | 0.895 | -0.005 | 0.168 | 0.133 | 1.266 | 0.512 | 0 |
| cv_highdim_inference | sqrtlasso_log | known | (1,1) | 0.950 | -0.051 | 0.155 | 0.167 | 0.931 | 0.652 | 0 |
| cv_highdim_inference | sqrtlasso_log | known | (1,158) | 0.965 | 0.006 | 0.119 | 0.120 | 0.993 | 0.462 | 0 |
| cv_highdim_inference | sqrtlasso_log | known | (1,2) | 0.850 | -0.097 | 0.145 | 0.123 | 1.181 | 0.474 | 0 |
| cv_highdim_onese | sqrt_refit_block_log | known | (1,1) | 0.940 | 0.011 | 0.150 | 0.143 | 1.050 | 0.559 | 0 |
| cv_highdim_onese | sqrt_refit_block_log | known | (1,158) | 0.960 | -0.001 | 0.098 | 0.098 | 1.005 | 0.380 | 0 |
| cv_highdim_onese | sqrt_refit_block_log | known | (1,2) | 0.900 | -0.028 | 0.123 | 0.103 | 1.189 | 0.402 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.935 | -0.017 | 0.144 | 0.143 | 1.009 | 0.558 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.965 | -0.001 | 0.096 | 0.098 | 0.975 | 0.382 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.895 | -0.027 | 0.123 | 0.104 | 1.179 | 0.405 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | (1,1) | 0.940 | 0.000 | 0.147 | 0.143 | 1.026 | 0.558 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | (1,158) | 0.960 | -0.001 | 0.098 | 0.098 | 0.999 | 0.382 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | (1,2) | 0.895 | -0.028 | 0.126 | 0.104 | 1.209 | 0.405 | 0 |
| cv_highdim_onese | sqrt_refit_df_log | known | (1,1) | 0.940 | 0.000 | 0.147 | 0.142 | 1.035 | 0.553 | 0 |
| cv_highdim_onese | sqrt_refit_df_log | known | (1,158) | 0.960 | -0.001 | 0.098 | 0.098 | 1.008 | 0.379 | 0 |
| cv_highdim_onese | sqrt_refit_df_log | known | (1,2) | 0.895 | -0.028 | 0.126 | 0.103 | 1.220 | 0.402 | 0 |
| cv_highdim_onese | sqrtlasso_log | known | (1,1) | 0.895 | -0.082 | 0.129 | 0.131 | 0.985 | 0.514 | 0 |
| cv_highdim_onese | sqrtlasso_log | known | (1,158) | 0.960 | 0.003 | 0.092 | 0.093 | 0.985 | 0.362 | 0 |
| cv_highdim_onese | sqrtlasso_log | known | (1,2) | 0.770 | -0.112 | 0.108 | 0.096 | 1.127 | 0.373 | 0 |
| cv_highdim_prediction | sqrt_refit_block_log | known | (1,1) | 0.675 | 0.313 | 0.335 | 0.218 | 1.541 | 0.827 | 0 |
| cv_highdim_prediction | sqrt_refit_block_log | known | (1,158) | 0.965 | 0.003 | 0.164 | 0.148 | 1.107 | 0.556 | 0 |
| cv_highdim_prediction | sqrt_refit_block_log | known | (1,2) | 0.880 | 0.002 | 0.201 | 0.158 | 1.268 | 0.598 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.820 | 0.213 | 0.260 | 0.206 | 1.265 | 0.791 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.960 | 0.006 | 0.142 | 0.139 | 1.021 | 0.531 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.895 | -0.001 | 0.174 | 0.148 | 1.174 | 0.569 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | (1,1) | 0.760 | 0.242 | 0.267 | 0.206 | 1.301 | 0.790 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | (1,158) | 0.960 | 0.006 | 0.148 | 0.139 | 1.066 | 0.531 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | (1,2) | 0.890 | -0.000 | 0.181 | 0.148 | 1.219 | 0.569 | 0 |
| cv_highdim_prediction | sqrt_refit_df_log | known | (1,1) | 0.755 | 0.242 | 0.267 | 0.203 | 1.317 | 0.781 | 0 |
| cv_highdim_prediction | sqrt_refit_df_log | known | (1,158) | 0.945 | 0.006 | 0.148 | 0.137 | 1.080 | 0.525 | 0 |
| cv_highdim_prediction | sqrt_refit_df_log | known | (1,2) | 0.885 | -0.000 | 0.181 | 0.147 | 1.234 | 0.562 | 0 |
| cv_highdim_prediction | sqrtlasso_log | known | (1,1) | 0.920 | 0.058 | 0.196 | 0.175 | 1.117 | 0.678 | 0 |
| cv_highdim_prediction | sqrtlasso_log | known | (1,158) | 0.970 | 0.005 | 0.125 | 0.123 | 1.018 | 0.468 | 0 |
| cv_highdim_prediction | sqrtlasso_log | known | (1,2) | 0.870 | -0.060 | 0.158 | 0.127 | 1.244 | 0.488 | 0 |
| fixed_block_corrections | aclime_refit_block_log | known | (1,1) | 0.920 | 0.052 | 0.177 | 0.157 | 1.125 | 0.612 | 0 |
| fixed_block_corrections | aclime_refit_block_log | known | (1,2) | 0.955 | 0.006 | 0.106 | 0.113 | 0.939 | 0.438 | 0 |
| fixed_block_corrections | aclime_refit_block_log | known | (1,38) | 0.955 | 0.001 | 0.110 | 0.104 | 1.062 | 0.403 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.930 | 0.025 | 0.174 | 0.158 | 1.103 | 0.614 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.975 | 0.010 | 0.102 | 0.113 | 0.896 | 0.440 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,38) | 0.955 | 0.001 | 0.108 | 0.104 | 1.032 | 0.405 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,1) | 0.950 | 0.012 | 0.167 | 0.156 | 1.068 | 0.608 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,2) | 0.970 | 0.008 | 0.100 | 0.113 | 0.888 | 0.437 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,38) | 0.970 | 0.002 | 0.103 | 0.103 | 0.998 | 0.402 | 0 |
| fixed_block_corrections | oracle_log | known | (1,1) | 0.945 | 0.032 | 0.170 | 0.155 | 1.100 | 0.601 | 0 |
| fixed_block_corrections | oracle_log | known | (1,2) | 0.950 | 0.008 | 0.103 | 0.112 | 0.925 | 0.433 | 0 |
| fixed_block_corrections | oracle_log | known | (1,38) | 0.965 | 0.002 | 0.106 | 0.102 | 1.036 | 0.398 | 0 |
| fixed_block_corrections | scaled_refit_block_log | known | (1,1) | 0.875 | 0.097 | 0.187 | 0.163 | 1.148 | 0.635 | 0 |
| fixed_block_corrections | scaled_refit_block_log | known | (1,2) | 0.975 | 0.006 | 0.110 | 0.117 | 0.940 | 0.453 | 0 |
| fixed_block_corrections | scaled_refit_block_log | known | (1,38) | 0.960 | 0.004 | 0.115 | 0.108 | 1.060 | 0.419 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.905 | 0.058 | 0.181 | 0.162 | 1.117 | 0.631 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.980 | 0.007 | 0.106 | 0.116 | 0.909 | 0.450 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,38) | 0.965 | 0.005 | 0.111 | 0.107 | 1.032 | 0.416 | 0 |
| fixed_block_corrections | sqrt_refit_block_log | known | (1,1) | 0.855 | 0.111 | 0.192 | 0.166 | 1.162 | 0.643 | 0 |
| fixed_block_corrections | sqrt_refit_block_log | known | (1,2) | 0.970 | 0.006 | 0.113 | 0.119 | 0.956 | 0.460 | 0 |
| fixed_block_corrections | sqrt_refit_block_log | known | (1,38) | 0.960 | 0.003 | 0.115 | 0.109 | 1.053 | 0.423 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.900 | 0.066 | 0.182 | 0.164 | 1.115 | 0.636 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.980 | 0.005 | 0.108 | 0.117 | 0.920 | 0.455 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.965 | 0.004 | 0.110 | 0.108 | 1.022 | 0.418 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | (1,1) | 0.875 | 0.087 | 0.186 | 0.164 | 1.137 | 0.636 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | (1,2) | 0.980 | 0.006 | 0.111 | 0.117 | 0.949 | 0.455 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | (1,38) | 0.955 | 0.003 | 0.113 | 0.108 | 1.050 | 0.418 | 0 |
| fixed_block_corrections | sqrt_refit_df_log | known | (1,1) | 0.875 | 0.087 | 0.186 | 0.162 | 1.148 | 0.630 | 0 |
| fixed_block_corrections | sqrt_refit_df_log | known | (1,2) | 0.980 | 0.006 | 0.111 | 0.116 | 0.958 | 0.450 | 0 |
| fixed_block_corrections | sqrt_refit_df_log | known | (1,38) | 0.955 | 0.003 | 0.113 | 0.107 | 1.061 | 0.414 | 0 |
| fixed_block_corrections | sqrtlasso_log | known | (1,1) | 0.930 | 0.041 | 0.175 | 0.157 | 1.115 | 0.608 | 0 |
| fixed_block_corrections | sqrtlasso_log | known | (1,2) | 0.960 | -0.018 | 0.103 | 0.112 | 0.920 | 0.433 | 0 |
| fixed_block_corrections | sqrtlasso_log | known | (1,38) | 0.955 | 0.002 | 0.108 | 0.104 | 1.038 | 0.403 | 0 |
| fixed_varying_corrections | aclime_refit_block_log | known | (1,1) | 0.940 | 0.011 | 0.114 | 0.115 | 0.993 | 0.449 | 0 |
| fixed_varying_corrections | aclime_refit_block_log | known | (1,2) | 0.945 | 0.002 | 0.084 | 0.083 | 1.020 | 0.322 | 0 |
| fixed_varying_corrections | aclime_refit_block_log | known | (1,78) | 0.945 | -0.000 | 0.080 | 0.077 | 1.037 | 0.300 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.935 | -0.004 | 0.113 | 0.115 | 0.982 | 0.449 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.004 | 0.082 | 0.083 | 0.996 | 0.322 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,78) | 0.955 | -0.001 | 0.079 | 0.077 | 1.022 | 0.301 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,1) | 0.930 | -0.008 | 0.112 | 0.115 | 0.976 | 0.447 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,2) | 0.950 | 0.004 | 0.081 | 0.083 | 0.986 | 0.322 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,78) | 0.940 | -0.000 | 0.079 | 0.077 | 1.023 | 0.299 | 0 |
| fixed_varying_corrections | oracle_log | known | (1,1) | 0.935 | 0.003 | 0.113 | 0.114 | 0.994 | 0.445 | 0 |
| fixed_varying_corrections | oracle_log | known | (1,2) | 0.950 | 0.004 | 0.083 | 0.082 | 1.011 | 0.320 | 0 |
| fixed_varying_corrections | oracle_log | known | (1,78) | 0.935 | -0.000 | 0.080 | 0.076 | 1.046 | 0.298 | 0 |
| fixed_varying_corrections | scaled_refit_block_log | known | (1,1) | 0.925 | 0.051 | 0.125 | 0.119 | 1.044 | 0.465 | 0 |
| fixed_varying_corrections | scaled_refit_block_log | known | (1,2) | 0.940 | 0.003 | 0.087 | 0.086 | 1.020 | 0.334 | 0 |
| fixed_varying_corrections | scaled_refit_block_log | known | (1,78) | 0.940 | -0.001 | 0.084 | 0.080 | 1.054 | 0.311 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.028 | 0.121 | 0.119 | 1.016 | 0.463 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.945 | 0.004 | 0.085 | 0.085 | 0.995 | 0.332 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,78) | 0.940 | -0.000 | 0.083 | 0.079 | 1.040 | 0.309 | 0 |
| fixed_varying_corrections | sqrt_refit_block_log | known | (1,1) | 0.915 | 0.066 | 0.130 | 0.121 | 1.076 | 0.471 | 0 |
| fixed_varying_corrections | sqrt_refit_block_log | known | (1,2) | 0.940 | 0.003 | 0.088 | 0.087 | 1.013 | 0.338 | 0 |
| fixed_varying_corrections | sqrt_refit_block_log | known | (1,78) | 0.940 | -0.001 | 0.085 | 0.080 | 1.059 | 0.314 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.038 | 0.124 | 0.120 | 1.037 | 0.467 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.940 | 0.003 | 0.085 | 0.086 | 0.986 | 0.334 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.930 | -0.000 | 0.083 | 0.080 | 1.040 | 0.311 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | (1,1) | 0.920 | 0.050 | 0.126 | 0.120 | 1.051 | 0.467 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | (1,2) | 0.940 | 0.003 | 0.086 | 0.086 | 1.006 | 0.334 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | (1,78) | 0.930 | -0.000 | 0.084 | 0.080 | 1.058 | 0.311 | 0 |
| fixed_varying_corrections | sqrt_refit_df_log | known | (1,1) | 0.915 | 0.050 | 0.126 | 0.119 | 1.057 | 0.465 | 0 |
| fixed_varying_corrections | sqrt_refit_df_log | known | (1,2) | 0.940 | 0.003 | 0.086 | 0.085 | 1.012 | 0.332 | 0 |
| fixed_varying_corrections | sqrt_refit_df_log | known | (1,78) | 0.930 | -0.000 | 0.084 | 0.079 | 1.063 | 0.309 | 0 |
| fixed_varying_corrections | sqrtlasso_log | known | (1,1) | 0.950 | 0.008 | 0.116 | 0.115 | 1.012 | 0.448 | 0 |
| fixed_varying_corrections | sqrtlasso_log | known | (1,2) | 0.950 | -0.013 | 0.082 | 0.082 | 0.994 | 0.320 | 0 |
| fixed_varying_corrections | sqrtlasso_log | known | (1,78) | 0.940 | -0.001 | 0.080 | 0.077 | 1.043 | 0.300 | 0 |

## Curvewise bands

| Scenario | Method | Index | Entry | Coverage | Mean width | Failures |
|---|---|---|---|---|---|---|
| cv_chain_inference | sqrt_refit_block_log | known | (1,1) | 0.905 | 0.842 | 0 |
| cv_chain_inference | sqrt_refit_block_log | known | (1,2) | 0.950 | 0.628 | 0 |
| cv_chain_inference | sqrt_refit_block_log | known | (1,38) | 0.930 | 0.618 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.895 | 0.848 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.945 | 0.633 | 0 |
| cv_chain_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.940 | 0.620 | 0 |
| cv_generated_inference | sqrt_refit_block_log | estimated | (1,1) | 0.950 | 0.696 | 0 |
| cv_generated_inference | sqrt_refit_block_log | estimated | (1,2) | 0.940 | 0.503 | 0 |
| cv_generated_inference | sqrt_refit_block_log | estimated | (1,78) | 0.940 | 0.452 | 0 |
| cv_generated_inference | sqrt_refit_block_log | known | (1,1) | 0.860 | 0.691 | 0 |
| cv_generated_inference | sqrt_refit_block_log | known | (1,2) | 0.945 | 0.501 | 0 |
| cv_generated_inference | sqrt_refit_block_log | known | (1,78) | 0.960 | 0.453 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.960 | 0.697 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.955 | 0.503 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.945 | 0.453 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.895 | 0.692 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.502 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.960 | 0.455 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.955 | 0.697 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.935 | 0.503 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.940 | 0.453 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | (1,1) | 0.870 | 0.691 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | (1,2) | 0.935 | 0.502 | 0 |
| cv_generated_inference | sqrt_refit_df_hc3_log | known | (1,78) | 0.960 | 0.455 | 0 |
| cv_generated_inference | sqrt_refit_df_log | estimated | (1,1) | 0.950 | 0.692 | 0 |
| cv_generated_inference | sqrt_refit_df_log | estimated | (1,2) | 0.930 | 0.500 | 0 |
| cv_generated_inference | sqrt_refit_df_log | estimated | (1,78) | 0.935 | 0.451 | 0 |
| cv_generated_inference | sqrt_refit_df_log | known | (1,1) | 0.870 | 0.687 | 0 |
| cv_generated_inference | sqrt_refit_df_log | known | (1,2) | 0.935 | 0.499 | 0 |
| cv_generated_inference | sqrt_refit_df_log | known | (1,78) | 0.960 | 0.452 | 0 |
| cv_generated_inference | sqrtlasso_log | estimated | (1,1) | 0.920 | 0.673 | 0 |
| cv_generated_inference | sqrtlasso_log | estimated | (1,2) | 0.885 | 0.487 | 0 |
| cv_generated_inference | sqrtlasso_log | estimated | (1,78) | 0.930 | 0.443 | 0 |
| cv_generated_inference | sqrtlasso_log | known | (1,1) | 0.900 | 0.668 | 0 |
| cv_generated_inference | sqrtlasso_log | known | (1,2) | 0.920 | 0.485 | 0 |
| cv_generated_inference | sqrtlasso_log | known | (1,78) | 0.950 | 0.445 | 0 |
| cv_generated_onese | sqrt_refit_block_log | estimated | (1,1) | 0.945 | 0.466 | 0 |
| cv_generated_onese | sqrt_refit_block_log | estimated | (1,2) | 0.900 | 0.340 | 0 |
| cv_generated_onese | sqrt_refit_block_log | estimated | (1,78) | 0.925 | 0.306 | 0 |
| cv_generated_onese | sqrt_refit_block_log | known | (1,1) | 0.930 | 0.466 | 0 |
| cv_generated_onese | sqrt_refit_block_log | known | (1,2) | 0.955 | 0.338 | 0 |
| cv_generated_onese | sqrt_refit_block_log | known | (1,78) | 0.955 | 0.306 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.945 | 0.465 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.895 | 0.338 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.935 | 0.306 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.940 | 0.464 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.960 | 0.337 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.965 | 0.306 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.945 | 0.465 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.890 | 0.338 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.930 | 0.306 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | (1,1) | 0.945 | 0.464 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | (1,2) | 0.955 | 0.337 | 0 |
| cv_generated_onese | sqrt_refit_df_hc3_log | known | (1,78) | 0.960 | 0.306 | 0 |
| cv_generated_onese | sqrt_refit_df_log | estimated | (1,1) | 0.945 | 0.463 | 0 |
| cv_generated_onese | sqrt_refit_df_log | estimated | (1,2) | 0.885 | 0.337 | 0 |
| cv_generated_onese | sqrt_refit_df_log | estimated | (1,78) | 0.930 | 0.305 | 0 |
| cv_generated_onese | sqrt_refit_df_log | known | (1,1) | 0.940 | 0.463 | 0 |
| cv_generated_onese | sqrt_refit_df_log | known | (1,2) | 0.955 | 0.336 | 0 |
| cv_generated_onese | sqrt_refit_df_log | known | (1,78) | 0.955 | 0.305 | 0 |
| cv_generated_onese | sqrtlasso_log | estimated | (1,1) | 0.865 | 0.440 | 0 |
| cv_generated_onese | sqrtlasso_log | estimated | (1,2) | 0.645 | 0.322 | 0 |
| cv_generated_onese | sqrtlasso_log | estimated | (1,78) | 0.950 | 0.296 | 0 |
| cv_generated_onese | sqrtlasso_log | known | (1,1) | 0.855 | 0.438 | 0 |
| cv_generated_onese | sqrtlasso_log | known | (1,2) | 0.810 | 0.321 | 0 |
| cv_generated_onese | sqrtlasso_log | known | (1,78) | 0.930 | 0.295 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | (1,1) | 0.795 | 0.621 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | (1,2) | 0.935 | 0.446 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | estimated | (1,78) | 0.950 | 0.405 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | known | (1,1) | 0.690 | 0.624 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | known | (1,2) | 0.940 | 0.449 | 0 |
| cv_generated_prediction | sqrt_refit_block_log | known | (1,78) | 0.940 | 0.408 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.910 | 0.610 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.920 | 0.437 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.945 | 0.396 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.775 | 0.613 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | 0.438 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.945 | 0.400 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.870 | 0.610 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.920 | 0.437 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.940 | 0.396 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | (1,1) | 0.770 | 0.612 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | (1,2) | 0.945 | 0.438 | 0 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | known | (1,78) | 0.945 | 0.400 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | (1,1) | 0.870 | 0.607 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | (1,2) | 0.920 | 0.435 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | estimated | (1,78) | 0.940 | 0.394 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | known | (1,1) | 0.760 | 0.610 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | known | (1,2) | 0.945 | 0.436 | 0 |
| cv_generated_prediction | sqrt_refit_df_log | known | (1,78) | 0.945 | 0.398 | 0 |
| cv_generated_prediction | sqrtlasso_log | estimated | (1,1) | 0.910 | 0.581 | 0 |
| cv_generated_prediction | sqrtlasso_log | estimated | (1,2) | 0.920 | 0.421 | 0 |
| cv_generated_prediction | sqrtlasso_log | estimated | (1,78) | 0.935 | 0.382 | 0 |
| cv_generated_prediction | sqrtlasso_log | known | (1,1) | 0.870 | 0.583 | 0 |
| cv_generated_prediction | sqrtlasso_log | known | (1,2) | 0.955 | 0.422 | 0 |
| cv_generated_prediction | sqrtlasso_log | known | (1,78) | 0.950 | 0.385 | 0 |
| cv_grid_inference | sqrt_refit_block_log | known | (1,1) | 0.910 | 0.969 | 0 |
| cv_grid_inference | sqrt_refit_block_log | known | (1,2) | 0.940 | 0.713 | 0 |
| cv_grid_inference | sqrt_refit_block_log | known | (1,38) | 0.950 | 0.715 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.925 | 0.979 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | 0.719 | 0 |
| cv_grid_inference | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.965 | 0.720 | 0 |
| cv_highdim_inference | sqrt_refit_block_log | known | (1,1) | 0.930 | 1.078 | 0 |
| cv_highdim_inference | sqrt_refit_block_log | known | (1,158) | 0.955 | 0.708 | 0 |
| cv_highdim_inference | sqrt_refit_block_log | known | (1,2) | 0.930 | 0.780 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | 1.083 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.975 | 0.715 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.786 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | (1,1) | 0.940 | 1.081 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | (1,158) | 0.970 | 0.715 | 0 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | known | (1,2) | 0.925 | 0.786 | 0 |
| cv_highdim_inference | sqrt_refit_df_log | known | (1,1) | 0.930 | 1.063 | 0 |
| cv_highdim_inference | sqrt_refit_df_log | known | (1,158) | 0.960 | 0.704 | 0 |
| cv_highdim_inference | sqrt_refit_df_log | known | (1,2) | 0.910 | 0.774 | 0 |
| cv_highdim_inference | sqrtlasso_log | known | (1,1) | 0.900 | 0.971 | 0 |
| cv_highdim_inference | sqrtlasso_log | known | (1,158) | 0.955 | 0.667 | 0 |
| cv_highdim_inference | sqrtlasso_log | known | (1,2) | 0.895 | 0.711 | 0 |
| cv_highdim_onese | sqrt_refit_block_log | known | (1,1) | 0.905 | 0.812 | 0 |
| cv_highdim_onese | sqrt_refit_block_log | known | (1,158) | 0.960 | 0.539 | 0 |
| cv_highdim_onese | sqrt_refit_block_log | known | (1,2) | 0.875 | 0.590 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.905 | 0.811 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.970 | 0.542 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.885 | 0.595 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | (1,1) | 0.910 | 0.810 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | (1,158) | 0.960 | 0.542 | 0 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | known | (1,2) | 0.885 | 0.595 | 0 |
| cv_highdim_onese | sqrt_refit_df_log | known | (1,1) | 0.905 | 0.802 | 0 |
| cv_highdim_onese | sqrt_refit_df_log | known | (1,158) | 0.950 | 0.537 | 0 |
| cv_highdim_onese | sqrt_refit_df_log | known | (1,2) | 0.870 | 0.589 | 0 |
| cv_highdim_onese | sqrtlasso_log | known | (1,1) | 0.820 | 0.725 | 0 |
| cv_highdim_onese | sqrtlasso_log | known | (1,158) | 0.955 | 0.503 | 0 |
| cv_highdim_onese | sqrtlasso_log | known | (1,2) | 0.695 | 0.535 | 0 |
| cv_highdim_prediction | sqrt_refit_block_log | known | (1,1) | 0.530 | 1.268 | 0 |
| cv_highdim_prediction | sqrt_refit_block_log | known | (1,158) | 0.970 | 0.824 | 0 |
| cv_highdim_prediction | sqrt_refit_block_log | known | (1,2) | 0.920 | 0.909 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.740 | 1.207 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.980 | 0.787 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.935 | 0.859 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | (1,1) | 0.635 | 1.205 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | (1,158) | 0.975 | 0.787 | 0 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | known | (1,2) | 0.930 | 0.859 | 0 |
| cv_highdim_prediction | sqrt_refit_df_log | known | (1,1) | 0.625 | 1.188 | 0 |
| cv_highdim_prediction | sqrt_refit_df_log | known | (1,158) | 0.970 | 0.777 | 0 |
| cv_highdim_prediction | sqrt_refit_df_log | known | (1,2) | 0.915 | 0.847 | 0 |
| cv_highdim_prediction | sqrtlasso_log | known | (1,1) | 0.920 | 0.991 | 0 |
| cv_highdim_prediction | sqrtlasso_log | known | (1,158) | 0.965 | 0.665 | 0 |
| cv_highdim_prediction | sqrtlasso_log | known | (1,2) | 0.925 | 0.720 | 0 |
| fixed_block_corrections | aclime_refit_block_log | known | (1,1) | 0.910 | 0.926 | 0 |
| fixed_block_corrections | aclime_refit_block_log | known | (1,2) | 0.945 | 0.664 | 0 |
| fixed_block_corrections | aclime_refit_block_log | known | (1,38) | 0.960 | 0.603 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.928 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.665 | 0 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | known | (1,38) | 0.970 | 0.606 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,1) | 0.935 | 0.935 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,2) | 0.975 | 0.663 | 0 |
| fixed_block_corrections | oracle_hc3_bc_log | known | (1,38) | 0.980 | 0.605 | 0 |
| fixed_block_corrections | oracle_log | known | (1,1) | 0.910 | 0.924 | 0 |
| fixed_block_corrections | oracle_log | known | (1,2) | 0.960 | 0.656 | 0 |
| fixed_block_corrections | oracle_log | known | (1,38) | 0.965 | 0.598 | 0 |
| fixed_block_corrections | scaled_refit_block_log | known | (1,1) | 0.870 | 0.972 | 0 |
| fixed_block_corrections | scaled_refit_block_log | known | (1,2) | 0.955 | 0.691 | 0 |
| fixed_block_corrections | scaled_refit_block_log | known | (1,38) | 0.955 | 0.633 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.905 | 0.966 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.970 | 0.688 | 0 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | known | (1,38) | 0.960 | 0.629 | 0 |
| fixed_block_corrections | sqrt_refit_block_log | known | (1,1) | 0.850 | 0.986 | 0 |
| fixed_block_corrections | sqrt_refit_block_log | known | (1,2) | 0.960 | 0.704 | 0 |
| fixed_block_corrections | sqrt_refit_block_log | known | (1,38) | 0.945 | 0.640 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.895 | 0.974 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.975 | 0.695 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.955 | 0.632 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | (1,1) | 0.875 | 0.972 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | (1,2) | 0.960 | 0.695 | 0 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | known | (1,38) | 0.950 | 0.632 | 0 |
| fixed_block_corrections | sqrt_refit_df_log | known | (1,1) | 0.870 | 0.962 | 0 |
| fixed_block_corrections | sqrt_refit_df_log | known | (1,2) | 0.960 | 0.687 | 0 |
| fixed_block_corrections | sqrt_refit_df_log | known | (1,38) | 0.950 | 0.625 | 0 |
| fixed_block_corrections | sqrtlasso_log | known | (1,1) | 0.905 | 0.914 | 0 |
| fixed_block_corrections | sqrtlasso_log | known | (1,2) | 0.955 | 0.656 | 0 |
| fixed_block_corrections | sqrtlasso_log | known | (1,38) | 0.955 | 0.600 | 0 |
| fixed_varying_corrections | aclime_refit_block_log | known | (1,1) | 0.930 | 0.687 | 0 |
| fixed_varying_corrections | aclime_refit_block_log | known | (1,2) | 0.915 | 0.500 | 0 |
| fixed_varying_corrections | aclime_refit_block_log | known | (1,78) | 0.970 | 0.450 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,1) | 0.925 | 0.687 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,2) | 0.940 | 0.502 | 0 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | known | (1,78) | 0.990 | 0.451 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,1) | 0.935 | 0.692 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,2) | 0.925 | 0.499 | 0 |
| fixed_varying_corrections | oracle_hc3_bc_log | known | (1,78) | 0.980 | 0.452 | 0 |
| fixed_varying_corrections | oracle_log | known | (1,1) | 0.940 | 0.687 | 0 |
| fixed_varying_corrections | oracle_log | known | (1,2) | 0.925 | 0.496 | 0 |
| fixed_varying_corrections | oracle_log | known | (1,78) | 0.980 | 0.449 | 0 |
| fixed_varying_corrections | scaled_refit_block_log | known | (1,1) | 0.880 | 0.721 | 0 |
| fixed_varying_corrections | scaled_refit_block_log | known | (1,2) | 0.935 | 0.519 | 0 |
| fixed_varying_corrections | scaled_refit_block_log | known | (1,78) | 0.970 | 0.471 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,1) | 0.930 | 0.718 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,2) | 0.940 | 0.516 | 0 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | known | (1,78) | 0.975 | 0.469 | 0 |
| fixed_varying_corrections | sqrt_refit_block_log | known | (1,1) | 0.840 | 0.732 | 0 |
| fixed_varying_corrections | sqrt_refit_block_log | known | (1,2) | 0.940 | 0.527 | 0 |
| fixed_varying_corrections | sqrt_refit_block_log | known | (1,78) | 0.975 | 0.476 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.724 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.940 | 0.521 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.980 | 0.471 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | (1,1) | 0.895 | 0.724 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | (1,2) | 0.935 | 0.521 | 0 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | known | (1,78) | 0.980 | 0.471 | 0 |
| fixed_varying_corrections | sqrt_refit_df_log | known | (1,1) | 0.895 | 0.720 | 0 |
| fixed_varying_corrections | sqrt_refit_df_log | known | (1,2) | 0.930 | 0.518 | 0 |
| fixed_varying_corrections | sqrt_refit_df_log | known | (1,78) | 0.975 | 0.468 | 0 |
| fixed_varying_corrections | sqrtlasso_log | known | (1,1) | 0.960 | 0.685 | 0 |
| fixed_varying_corrections | sqrtlasso_log | known | (1,2) | 0.905 | 0.497 | 0 |
| fixed_varying_corrections | sqrtlasso_log | known | (1,78) | 0.970 | 0.450 | 0 |

## Paired rule comparisons

The seed identifies the same generated dataset. The comparison includes support selection, bandwidth selection, and index estimation. Width ratios use the same replications.

| Scenario | Method | Reference | Index | Kind | NMC | Coverage difference | MCSE | Width ratio |
|---|---|---|---|---|---|---|---|---|
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.025 | 0.018 | 1.139 |
| cv_generated_inference | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.025 | 0.017 | 1.146 |
| cv_generated_inference | sqrt_refit_df | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.040 | 0.019 | 1.133 |
| cv_generated_inference | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.040 | 0.019 | 1.140 |
| cv_generated_inference | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.050 | 0.020 | 1.140 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.017 | 1.147 |
| cv_generated_inference | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.016 | 1.147 |
| cv_generated_inference | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.016 | 1.141 |
| cv_generated_inference | sqrtlasso | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.035 | 0.019 | 1.107 |
| cv_generated_inference | sqrtlasso_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.045 | 0.018 | 1.114 |
| cv_generated_onese | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.040 | 0.021 | 0.841 |
| cv_generated_onese | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.020 | 0.844 |
| cv_generated_onese | sqrt_refit_df | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.021 | 0.836 |
| cv_generated_onese | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.035 | 0.022 | 0.839 |
| cv_generated_onese | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.021 | 0.839 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.040 | 0.021 | 0.842 |
| cv_generated_onese | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.035 | 0.021 | 0.841 |
| cv_generated_onese | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.035 | 0.021 | 0.839 |
| cv_generated_onese | sqrtlasso | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.000 | 0.027 | 0.804 |
| cv_generated_onese | sqrtlasso_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.030 | 0.023 | 0.806 |
| cv_generated_prediction | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | -0.035 | 0.019 | 1.047 |
| cv_generated_prediction | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | -0.055 | 0.023 | 1.052 |
| cv_generated_prediction | sqrt_refit_df | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.000 | 0.016 | 1.027 |
| cv_generated_prediction | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.005 | 0.015 | 1.031 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.005 | 0.013 | 1.031 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | -0.010 | 0.017 | 1.036 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | -0.015 | 0.018 | 1.036 |
| cv_generated_prediction | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | -0.015 | 0.018 | 1.032 |
| cv_generated_prediction | sqrtlasso_log | cv_generated_prediction/sqrtlasso | known | pointwise | 200 | 0.010 | 0.007 | 1.005 |
| cv_highdim_inference | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.015 | 0.018 | 1.056 |
| cv_highdim_inference | sqrt_refit_block_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.015 | 0.021 | 1.072 |
| cv_highdim_inference | sqrt_refit_df | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.035 | 0.016 | 1.042 |
| cv_highdim_inference | sqrt_refit_df_hc3 | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.035 | 0.016 | 1.056 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.025 | 0.017 | 1.056 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.025 | 0.018 | 1.074 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.000 | 0.019 | 1.073 |
| cv_highdim_inference | sqrt_refit_df_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.000 | 0.019 | 1.058 |
| cv_highdim_inference | sqrtlasso | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.005 | 0.022 | 0.959 |
| cv_highdim_inference | sqrtlasso_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.025 | 0.017 | 0.974 |
| cv_highdim_onese | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.005 | 0.018 | 0.827 |
| cv_highdim_onese | sqrt_refit_block_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.015 | 0.017 | 0.835 |
| cv_highdim_onese | sqrt_refit_df | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.010 | 0.017 | 0.818 |
| cv_highdim_onese | sqrt_refit_df_hc3 | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.010 | 0.017 | 0.826 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.010 | 0.019 | 0.826 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.010 | 0.017 | 0.834 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.015 | 0.015 | 0.834 |
| cv_highdim_onese | sqrt_refit_df_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | 0.015 | 0.015 | 0.827 |
| cv_highdim_onese | sqrtlasso | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.070 | 0.028 | 0.761 |
| cv_highdim_onese | sqrtlasso_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.030 | 0.023 | 0.768 |
| cv_highdim_prediction | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.160 | 0.030 | 1.219 |
| cv_highdim_prediction | sqrt_refit_block_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.250 | 0.035 | 1.236 |
| cv_highdim_prediction | sqrt_refit_df | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.110 | 0.025 | 1.152 |
| cv_highdim_prediction | sqrt_refit_df_hc3 | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.105 | 0.025 | 1.165 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.085 | 0.023 | 1.165 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.105 | 0.027 | 1.182 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.165 | 0.031 | 1.182 |
| cv_highdim_prediction | sqrt_refit_df_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.170 | 0.031 | 1.167 |
| cv_highdim_prediction | sqrtlasso_log | cv_highdim_prediction/sqrtlasso | known | pointwise | 200 | -0.005 | 0.011 | 1.014 |
| fixed_block_corrections | aclime_refit_block | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.045 | 0.019 | 0.951 |
| fixed_block_corrections | aclime_refit_block_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.030 | 0.012 | 0.962 |
| fixed_block_corrections | aclime_refit_df_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.050 | 0.022 | 0.954 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.040 | 0.017 | 0.965 |
| fixed_block_corrections | oracle | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.045 | 0.019 | 0.935 |
| fixed_block_corrections | oracle_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.050 | 0.023 | 0.944 |
| fixed_block_corrections | oracle_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.060 | 0.018 | 0.956 |
| fixed_block_corrections | oracle_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.055 | 0.016 | 0.946 |
| fixed_block_corrections | scaled_refit_block | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.020 | 0.010 | 0.987 |
| fixed_block_corrections | scaled_refit_block_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | -0.015 | 0.013 | 0.998 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.055 | 0.018 | 0.980 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.009 | 0.992 |
| fixed_block_corrections | sqrt_refit_block_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | -0.035 | 0.016 | 1.011 |
| fixed_block_corrections | sqrt_refit_df | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.020 | 0.012 | 0.979 |
| fixed_block_corrections | sqrt_refit_df_hc3 | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.025 | 0.013 | 0.989 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.050 | 0.017 | 0.989 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.010 | 0.007 | 1.000 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | -0.015 | 0.011 | 1.000 |
| fixed_block_corrections | sqrt_refit_df_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | -0.015 | 0.011 | 0.990 |
| fixed_block_corrections | sqrtlasso | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.055 | 0.020 | 0.945 |
| fixed_block_corrections | sqrtlasso_log | fixed_block_corrections/sqrt_refit_block | known | pointwise | 200 | 0.040 | 0.017 | 0.956 |
| fixed_varying_corrections | aclime_refit_block | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.020 | 0.019 | 0.952 |
| fixed_varying_corrections | aclime_refit_block_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.020 | 0.016 | 0.958 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.019 | 0.953 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.018 | 0.959 |
| fixed_varying_corrections | oracle | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.010 | 0.020 | 0.944 |
| fixed_varying_corrections | oracle_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.021 | 0.949 |
| fixed_varying_corrections | oracle_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.010 | 0.017 | 0.955 |
| fixed_varying_corrections | oracle_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.017 | 0.950 |
| fixed_varying_corrections | scaled_refit_block | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.010 | 0.010 | 0.988 |
| fixed_varying_corrections | scaled_refit_block_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.005 | 0.009 | 0.994 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.017 | 0.983 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.000 | 0.007 | 0.990 |
| fixed_varying_corrections | sqrt_refit_block_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | -0.005 | 0.013 | 1.006 |
| fixed_varying_corrections | sqrt_refit_df | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.009 | 0.986 |
| fixed_varying_corrections | sqrt_refit_df_hc3 | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.020 | 0.010 | 0.991 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.020 | 0.012 | 0.991 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.000 | 0.000 | 0.998 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.000 | 0.010 | 0.998 |
| fixed_varying_corrections | sqrt_refit_df_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | -0.005 | 0.009 | 0.992 |
| fixed_varying_corrections | sqrtlasso | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.015 | 0.019 | 0.950 |
| fixed_varying_corrections | sqrtlasso_log | fixed_varying_corrections/sqrt_refit_block | known | pointwise | 200 | 0.030 | 0.014 | 0.957 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.010 | 0.014 | 1.172 |
| cv_generated_inference | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.000 | 0.017 | 1.181 |
| cv_generated_inference | sqrt_refit_df | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.025 | 0.017 | 1.167 |
| cv_generated_inference | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.015 | 0.015 | 1.174 |
| cv_generated_inference | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.025 | 0.018 | 1.174 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.010 | 0.016 | 1.183 |
| cv_generated_inference | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.010 | 0.017 | 1.183 |
| cv_generated_inference | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.015 | 0.018 | 1.176 |
| cv_generated_inference | sqrtlasso | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.040 | 0.020 | 1.139 |
| cv_generated_inference | sqrtlasso_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.010 | 0.016 | 1.147 |
| cv_generated_onese | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.040 | 0.017 | 0.805 |
| cv_generated_onese | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.025 | 0.018 | 0.807 |
| cv_generated_onese | sqrt_refit_df | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.030 | 0.019 | 0.800 |
| cv_generated_onese | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.030 | 0.019 | 0.803 |
| cv_generated_onese | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.025 | 0.019 | 0.803 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.035 | 0.016 | 0.806 |
| cv_generated_onese | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.025 | 0.018 | 0.806 |
| cv_generated_onese | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.025 | 0.018 | 0.803 |
| cv_generated_onese | sqrtlasso | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.155 | 0.030 | 0.766 |
| cv_generated_onese | sqrtlasso_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.095 | 0.028 | 0.769 |
| cv_generated_prediction | sqrt_refit_block | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.060 | 0.025 | 1.063 |
| cv_generated_prediction | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.165 | 0.033 | 1.069 |
| cv_generated_prediction | sqrt_refit_df | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.015 | 0.018 | 1.036 |
| cv_generated_prediction | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.015 | 0.018 | 1.040 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.010 | 0.014 | 1.040 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.050 | 0.025 | 1.046 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.075 | 0.027 | 1.046 |
| cv_generated_prediction | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | known | family | 200 | -0.075 | 0.027 | 1.042 |
| cv_generated_prediction | sqrtlasso_log | cv_generated_prediction/sqrtlasso | known | family | 200 | 0.000 | 0.012 | 1.006 |
| cv_highdim_inference | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.030 | 0.019 | 1.075 |
| cv_highdim_inference | sqrt_refit_block_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.055 | 0.019 | 1.094 |
| cv_highdim_inference | sqrt_refit_df | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.015 | 0.019 | 1.063 |
| cv_highdim_inference | sqrt_refit_df_hc3 | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.025 | 0.018 | 1.079 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.015 | 0.019 | 1.079 |
| cv_highdim_inference | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.045 | 0.018 | 1.101 |
| cv_highdim_inference | sqrt_refit_df_hc3_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.045 | 0.019 | 1.099 |
| cv_highdim_inference | sqrt_refit_df_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.035 | 0.019 | 1.082 |
| cv_highdim_inference | sqrtlasso | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.105 | 0.024 | 0.987 |
| cv_highdim_inference | sqrtlasso_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.030 | 0.017 | 1.004 |
| cv_highdim_onese | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.060 | 0.023 | 0.823 |
| cv_highdim_onese | sqrt_refit_block_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.005 | 0.021 | 0.832 |
| cv_highdim_onese | sqrt_refit_df | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.075 | 0.023 | 0.818 |
| cv_highdim_onese | sqrt_refit_df_hc3 | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.060 | 0.022 | 0.826 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.070 | 0.024 | 0.826 |
| cv_highdim_onese | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.010 | 0.020 | 0.836 |
| cv_highdim_onese | sqrt_refit_df_hc3_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.000 | 0.020 | 0.835 |
| cv_highdim_onese | sqrt_refit_df_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.010 | 0.021 | 0.827 |
| cv_highdim_onese | sqrtlasso | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.220 | 0.031 | 0.751 |
| cv_highdim_onese | sqrtlasso_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.150 | 0.028 | 0.759 |
| cv_highdim_prediction | sqrt_refit_block | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.030 | 0.026 | 1.257 |
| cv_highdim_prediction | sqrt_refit_block_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.275 | 0.039 | 1.279 |
| cv_highdim_prediction | sqrt_refit_df | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.015 | 0.023 | 1.180 |
| cv_highdim_prediction | sqrt_refit_df_hc3 | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.025 | 0.023 | 1.195 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.035 | 0.022 | 1.195 |
| cv_highdim_prediction | sqrt_refit_df_hc3_bc_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.055 | 0.031 | 1.217 |
| cv_highdim_prediction | sqrt_refit_df_hc3_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.105 | 0.034 | 1.216 |
| cv_highdim_prediction | sqrt_refit_df_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | -0.125 | 0.035 | 1.199 |
| cv_highdim_prediction | sqrtlasso_log | cv_highdim_prediction/sqrtlasso | known | family | 200 | 0.000 | 0.016 | 1.015 |
| fixed_block_corrections | aclime_refit_block | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.025 | 0.019 | 0.941 |
| fixed_block_corrections | aclime_refit_block_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.045 | 0.019 | 0.954 |
| fixed_block_corrections | aclime_refit_df_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.025 | 0.023 | 0.946 |
| fixed_block_corrections | aclime_refit_df_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.045 | 0.018 | 0.959 |
| fixed_block_corrections | oracle | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.035 | 0.018 | 0.936 |
| fixed_block_corrections | oracle_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.040 | 0.020 | 0.946 |
| fixed_block_corrections | oracle_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.055 | 0.019 | 0.959 |
| fixed_block_corrections | oracle_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.040 | 0.020 | 0.948 |
| fixed_block_corrections | scaled_refit_block | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.010 | 0.012 | 0.986 |
| fixed_block_corrections | scaled_refit_block_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | -0.030 | 0.017 | 0.999 |
| fixed_block_corrections | scaled_refit_df_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.050 | 0.018 | 0.980 |
| fixed_block_corrections | scaled_refit_df_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.035 | 0.015 | 0.994 |
| fixed_block_corrections | sqrt_refit_block_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | -0.045 | 0.018 | 1.014 |
| fixed_block_corrections | sqrt_refit_df | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.020 | 0.012 | 0.976 |
| fixed_block_corrections | sqrt_refit_df_hc3 | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.025 | 0.013 | 0.987 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.050 | 0.017 | 0.987 |
| fixed_block_corrections | sqrt_refit_df_hc3_bc_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.030 | 0.016 | 1.001 |
| fixed_block_corrections | sqrt_refit_df_hc3_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | -0.015 | 0.015 | 1.001 |
| fixed_block_corrections | sqrt_refit_df_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | -0.025 | 0.015 | 0.990 |
| fixed_block_corrections | sqrtlasso | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.015 | 0.018 | 0.932 |
| fixed_block_corrections | sqrtlasso_log | fixed_block_corrections/sqrt_refit_block | known | family | 200 | 0.035 | 0.018 | 0.944 |
| fixed_varying_corrections | aclime_refit_block | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.005 | 0.013 | 0.946 |
| fixed_varying_corrections | aclime_refit_block_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.015 | 0.009 | 0.954 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.015 | 0.015 | 0.948 |
| fixed_varying_corrections | aclime_refit_df_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.015 | 0.009 | 0.955 |
| fixed_varying_corrections | oracle | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.015 | 0.015 | 0.942 |
| fixed_varying_corrections | oracle_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.010 | 0.014 | 0.948 |
| fixed_varying_corrections | oracle_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.000 | 0.010 | 0.955 |
| fixed_varying_corrections | oracle_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.010 | 0.012 | 0.949 |
| fixed_varying_corrections | scaled_refit_block | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.000 | 0.007 | 0.987 |
| fixed_varying_corrections | scaled_refit_block_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.035 | 0.016 | 0.995 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.005 | 0.013 | 0.982 |
| fixed_varying_corrections | scaled_refit_df_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.015 | 0.011 | 0.990 |
| fixed_varying_corrections | sqrt_refit_block_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.050 | 0.018 | 1.008 |
| fixed_varying_corrections | sqrt_refit_df | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.005 | 0.009 | 0.984 |
| fixed_varying_corrections | sqrt_refit_df_hc3 | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.005 | 0.009 | 0.990 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.015 | 0.009 | 0.990 |
| fixed_varying_corrections | sqrt_refit_df_hc3_bc_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.005 | 0.009 | 0.998 |
| fixed_varying_corrections | sqrt_refit_df_hc3_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.015 | 0.015 | 0.998 |
| fixed_varying_corrections | sqrt_refit_df_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.020 | 0.016 | 0.992 |
| fixed_varying_corrections | sqrtlasso | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | -0.010 | 0.014 | 0.943 |
| fixed_varying_corrections | sqrtlasso_log | fixed_varying_corrections/sqrt_refit_block | known | family | 200 | 0.005 | 0.011 | 0.950 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.015 | 0.011 | 1.153 |
| cv_generated_inference | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.020 | 0.014 | 1.161 |
| cv_generated_inference | sqrt_refit_df | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.010 | 0.012 | 1.148 |
| cv_generated_inference | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.005 | 0.011 | 1.154 |
| cv_generated_inference | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.000 | 0.010 | 1.154 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.015 | 0.011 | 1.162 |
| cv_generated_inference | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.010 | 0.012 | 1.162 |
| cv_generated_inference | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.015 | 0.013 | 1.156 |
| cv_generated_inference | sqrtlasso | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.030 | 0.016 | 1.120 |
| cv_generated_inference | sqrtlasso_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.020 | 0.014 | 1.128 |
| cv_generated_onese | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.835 |
| cv_generated_onese | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.838 |
| cv_generated_onese | sqrt_refit_df | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.830 |
| cv_generated_onese | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.833 |
| cv_generated_onese | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.015 | 0.018 | 0.833 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.836 |
| cv_generated_onese | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.836 |
| cv_generated_onese | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | 0.025 | 0.017 | 0.833 |
| cv_generated_onese | sqrtlasso | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.055 | 0.023 | 0.799 |
| cv_generated_onese | sqrtlasso_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.025 | 0.019 | 0.802 |
| cv_generated_prediction | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.080 | 0.022 | 1.052 |
| cv_generated_prediction | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.085 | 0.023 | 1.057 |
| cv_generated_prediction | sqrt_refit_df | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.045 | 0.018 | 1.031 |
| cv_generated_prediction | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.045 | 0.018 | 1.035 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.025 | 0.015 | 1.035 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.050 | 0.018 | 1.040 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.070 | 0.021 | 1.040 |
| cv_generated_prediction | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.070 | 0.021 | 1.036 |
| cv_generated_prediction | sqrtlasso_log | cv_generated_prediction/sqrtlasso | estimated | pointwise | 200 | -0.020 | 0.014 | 1.005 |
| cv_generated_inference | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | family | 200 | 0.000 | 0.017 | 1.182 |
| cv_generated_inference | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | 0.000 | 0.020 | 1.192 |
| cv_generated_inference | sqrt_refit_df | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.017 | 1.176 |
| cv_generated_inference | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.017 | 1.184 |
| cv_generated_inference | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.019 | 1.184 |
| cv_generated_inference | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | 0.005 | 0.019 | 1.193 |
| cv_generated_inference | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | 0.000 | 0.020 | 1.193 |
| cv_generated_inference | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | 0.000 | 0.020 | 1.186 |
| cv_generated_inference | sqrtlasso | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.075 | 0.022 | 1.151 |
| cv_generated_inference | sqrtlasso_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.020 | 0.020 | 1.160 |
| cv_generated_onese | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.019 | 0.812 |
| cv_generated_onese | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.015 | 0.019 | 0.815 |
| cv_generated_onese | sqrt_refit_df | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.020 | 0.808 |
| cv_generated_onese | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.005 | 0.019 | 0.810 |
| cv_generated_onese | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.015 | 0.019 | 0.810 |
| cv_generated_onese | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.005 | 0.021 | 0.813 |
| cv_generated_onese | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.020 | 0.813 |
| cv_generated_onese | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.015 | 0.021 | 0.811 |
| cv_generated_onese | sqrtlasso | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.205 | 0.032 | 0.775 |
| cv_generated_onese | sqrtlasso_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.170 | 0.030 | 0.778 |
| cv_generated_prediction | sqrt_refit_block | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.020 | 0.019 | 1.064 |
| cv_generated_prediction | sqrt_refit_block_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.085 | 0.025 | 1.070 |
| cv_generated_prediction | sqrt_refit_df | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.005 | 0.018 | 1.037 |
| cv_generated_prediction | sqrt_refit_df_hc3 | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.005 | 0.018 | 1.042 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc | cv_generated_prediction/sqrtlasso | estimated | family | 200 | 0.010 | 0.016 | 1.042 |
| cv_generated_prediction | sqrt_refit_df_hc3_bc_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.015 | 0.021 | 1.048 |
| cv_generated_prediction | sqrt_refit_df_hc3_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.025 | 0.022 | 1.048 |
| cv_generated_prediction | sqrt_refit_df_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.025 | 0.022 | 1.043 |
| cv_generated_prediction | sqrtlasso_log | cv_generated_prediction/sqrtlasso | estimated | family | 200 | -0.010 | 0.012 | 1.006 |
