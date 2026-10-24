# Diagonal and CV refinement

Coverage is unconditional over all requested fits. Failures count as noncoverage. Intervals are nominal 95%. Bands are joint over the three selected entries and the full grid. Wilson intervals quantify Monte Carlo uncertainty, not inferential uncertainty in a dataset.

Method suffixes: `refit` removes coefficient shrinkage; `df` refits the union support and normalizes residual-maker row variance; `hc3` adjusts final smoothing leverage; `bc` subtracts second-order inverse curvature. The components are defined in `docs/single_index/DIAGONAL_TUNING.md`. Actual tuning values are in `selected_tuning.csv`.

| Scenario | Method | Index | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| cv_chain_minpilot_r10 | oracle_hc3_log | known | 200 | 0 | 0.930 | [0.886, 0.958] | 0.796 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.788 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.793 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.803 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.803 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.807 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | 200 | 0 | 0.930 | [0.886, 0.958] | 0.796 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.788 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.792 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.803 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.803 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.807 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | 200 | 0 | 0.900 | [0.851, 0.934] | 0.623 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.623 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | 200 | 0 | 0.905 | [0.856, 0.938] | 0.619 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.616 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | 200 | 0 | 0.890 | [0.839, 0.926] | 0.620 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.617 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | 200 | 0 | 0.935 | [0.892, 0.962] | 0.625 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.622 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.625 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.622 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.628 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.625 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | 200 | 0 | 0.900 | [0.851, 0.934] | 0.623 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.623 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | 200 | 0 | 0.905 | [0.856, 0.938] | 0.619 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.616 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | 200 | 0 | 0.890 | [0.839, 0.926] | 0.620 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.617 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | 200 | 0 | 0.935 | [0.892, 0.962] | 0.625 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.622 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.625 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.621 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | 200 | 0 | 0.925 | [0.880, 0.954] | 0.628 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.625 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.917 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.902 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.909 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.921 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.920 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.923 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.917 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.901 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.908 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.920 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.920 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.922 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.986 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | 200 | 0 | 0.875 | [0.822, 0.914] | 0.966 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.973 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.992 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.991 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | 200 | 0 | 0.910 | [0.862, 0.942] | 1.000 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.986 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | 200 | 0 | 0.880 | [0.828, 0.918] | 0.966 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.905 | [0.856, 0.938] | 0.972 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.992 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.991 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.999 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.997 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.976 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.975 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.997 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.890 | [0.839, 0.926] | 0.995 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | 200 | 0 | 0.875 | [0.822, 0.914] | 1.006 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.997 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.976 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.950 | [0.910, 0.973] | 0.975 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.997 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.890 | [0.839, 0.926] | 0.995 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | 200 | 0 | 0.875 | [0.822, 0.914] | 1.006 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.623 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.615 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.616 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.621 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.621 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | 200 | 0 | 0.960 | [0.923, 0.980] | 0.623 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.623 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.614 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | 200 | 0 | 0.930 | [0.886, 0.958] | 0.616 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | 200 | 0 | 0.965 | [0.930, 0.983] | 0.621 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.621 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | 200 | 0 | 0.960 | [0.923, 0.980] | 0.623 |

## Pointwise intervals at zero

| Scenario | Method | Index | Entry | Coverage | Bias | Empirical SD | RMS SE | SD/SE | Mean length | Failures |
|---|---|---|---|---|---|---|---|---|---|---|
| cv_chain_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.950 | 0.010 | 0.137 | 0.142 | 0.969 | 0.553 | 0 |
| cv_chain_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.940 | -0.015 | 0.113 | 0.111 | 1.020 | 0.431 | 0 |
| cv_chain_minpilot_r10 | oracle_hc3_log | known | (1,38) | 0.935 | -0.011 | 0.108 | 0.107 | 1.015 | 0.415 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.960 | 0.024 | 0.140 | 0.142 | 0.990 | 0.548 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.940 | -0.015 | 0.115 | 0.111 | 1.033 | 0.431 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | (1,38) | 0.930 | -0.011 | 0.110 | 0.107 | 1.029 | 0.417 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.955 | 0.001 | 0.137 | 0.142 | 0.963 | 0.550 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.945 | -0.008 | 0.112 | 0.112 | 0.997 | 0.434 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.940 | -0.010 | 0.107 | 0.107 | 0.995 | 0.417 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | 0.001 | 0.137 | 0.142 | 0.963 | 0.557 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.945 | -0.008 | 0.112 | 0.112 | 0.997 | 0.434 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.940 | -0.010 | 0.107 | 0.107 | 0.995 | 0.417 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.945 | 0.019 | 0.139 | 0.142 | 0.979 | 0.556 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.940 | -0.015 | 0.115 | 0.112 | 1.025 | 0.434 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,38) | 0.935 | -0.010 | 0.110 | 0.107 | 1.023 | 0.417 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.945 | 0.024 | 0.140 | 0.143 | 0.980 | 0.559 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | -0.015 | 0.115 | 0.112 | 1.023 | 0.435 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | (1,38) | 0.930 | -0.011 | 0.110 | 0.108 | 1.019 | 0.421 | 0 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.950 | 0.010 | 0.137 | 0.142 | 0.969 | 0.553 | 0 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.940 | -0.015 | 0.113 | 0.111 | 1.020 | 0.431 | 0 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | (1,38) | 0.935 | -0.011 | 0.108 | 0.107 | 1.015 | 0.415 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.960 | 0.024 | 0.140 | 0.142 | 0.990 | 0.548 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.940 | -0.014 | 0.115 | 0.111 | 1.033 | 0.430 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | (1,38) | 0.930 | -0.011 | 0.110 | 0.107 | 1.029 | 0.417 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.955 | 0.001 | 0.137 | 0.142 | 0.963 | 0.550 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.945 | -0.008 | 0.112 | 0.112 | 0.997 | 0.434 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.940 | -0.010 | 0.107 | 0.107 | 0.994 | 0.417 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.950 | 0.001 | 0.137 | 0.142 | 0.963 | 0.557 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.945 | -0.008 | 0.112 | 0.112 | 0.997 | 0.434 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.940 | -0.010 | 0.107 | 0.107 | 0.994 | 0.417 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.945 | 0.019 | 0.139 | 0.142 | 0.979 | 0.556 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.940 | -0.015 | 0.115 | 0.112 | 1.025 | 0.434 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,38) | 0.935 | -0.010 | 0.110 | 0.107 | 1.023 | 0.417 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.945 | 0.024 | 0.140 | 0.143 | 0.980 | 0.559 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | -0.014 | 0.115 | 0.112 | 1.023 | 0.435 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | (1,38) | 0.930 | -0.011 | 0.110 | 0.108 | 1.019 | 0.421 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | (1,1) | 0.950 | 0.002 | 0.121 | 0.116 | 1.046 | 0.454 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | (1,2) | 0.950 | -0.001 | 0.083 | 0.083 | 0.996 | 0.325 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | (1,78) | 0.925 | -0.001 | 0.080 | 0.077 | 1.042 | 0.300 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.970 | -0.005 | 0.107 | 0.115 | 0.935 | 0.448 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.940 | 0.009 | 0.080 | 0.081 | 0.980 | 0.318 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | (1,78) | 0.965 | -0.000 | 0.073 | 0.076 | 0.967 | 0.296 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | (1,1) | 0.945 | 0.013 | 0.123 | 0.117 | 1.052 | 0.452 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | (1,2) | 0.955 | -0.000 | 0.083 | 0.084 | 0.998 | 0.326 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | (1,78) | 0.930 | -0.001 | 0.079 | 0.077 | 1.022 | 0.301 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.960 | 0.005 | 0.110 | 0.115 | 0.954 | 0.447 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.950 | 0.010 | 0.081 | 0.082 | 0.988 | 0.319 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | (1,78) | 0.965 | 0.000 | 0.073 | 0.076 | 0.964 | 0.296 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | (1,1) | 0.930 | -0.004 | 0.121 | 0.117 | 1.034 | 0.453 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | (1,2) | 0.955 | -0.000 | 0.082 | 0.084 | 0.979 | 0.326 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | (1,78) | 0.930 | -0.000 | 0.078 | 0.077 | 1.001 | 0.302 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.960 | -0.012 | 0.108 | 0.115 | 0.936 | 0.447 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.950 | 0.009 | 0.079 | 0.082 | 0.965 | 0.319 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.970 | 0.000 | 0.072 | 0.076 | 0.947 | 0.297 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.940 | -0.004 | 0.121 | 0.117 | 1.034 | 0.456 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.955 | -0.000 | 0.082 | 0.084 | 0.979 | 0.326 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.930 | -0.000 | 0.078 | 0.077 | 1.001 | 0.302 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.960 | -0.012 | 0.108 | 0.115 | 0.936 | 0.450 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | 0.009 | 0.079 | 0.082 | 0.965 | 0.319 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.970 | 0.000 | 0.072 | 0.076 | 0.947 | 0.297 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.935 | 0.008 | 0.122 | 0.117 | 1.046 | 0.456 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.955 | -0.000 | 0.083 | 0.084 | 0.994 | 0.326 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.930 | -0.000 | 0.079 | 0.077 | 1.017 | 0.302 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.970 | -0.000 | 0.109 | 0.115 | 0.947 | 0.450 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | 0.010 | 0.080 | 0.082 | 0.982 | 0.319 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,78) | 0.970 | 0.000 | 0.073 | 0.076 | 0.961 | 0.297 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | (1,1) | 0.935 | 0.013 | 0.123 | 0.117 | 1.046 | 0.458 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | (1,2) | 0.955 | -0.000 | 0.083 | 0.084 | 0.992 | 0.327 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | (1,78) | 0.930 | -0.001 | 0.079 | 0.078 | 1.017 | 0.303 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.980 | 0.005 | 0.110 | 0.116 | 0.949 | 0.452 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | 0.010 | 0.081 | 0.082 | 0.983 | 0.320 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | (1,78) | 0.965 | 0.000 | 0.073 | 0.076 | 0.958 | 0.298 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | (1,1) | 0.950 | 0.002 | 0.121 | 0.116 | 1.046 | 0.454 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | (1,2) | 0.950 | -0.001 | 0.083 | 0.083 | 0.996 | 0.325 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | (1,78) | 0.925 | -0.001 | 0.080 | 0.077 | 1.042 | 0.300 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.970 | -0.005 | 0.107 | 0.115 | 0.935 | 0.448 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.940 | 0.009 | 0.080 | 0.081 | 0.980 | 0.318 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | (1,78) | 0.965 | -0.000 | 0.073 | 0.076 | 0.967 | 0.296 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | (1,1) | 0.945 | 0.013 | 0.123 | 0.117 | 1.053 | 0.452 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | (1,2) | 0.955 | -0.000 | 0.083 | 0.084 | 0.997 | 0.326 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | (1,78) | 0.930 | -0.001 | 0.079 | 0.077 | 1.022 | 0.301 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.960 | 0.005 | 0.110 | 0.115 | 0.954 | 0.447 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.950 | 0.009 | 0.081 | 0.082 | 0.989 | 0.319 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | (1,78) | 0.965 | 0.000 | 0.073 | 0.076 | 0.962 | 0.296 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | (1,1) | 0.930 | -0.004 | 0.121 | 0.117 | 1.035 | 0.453 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | (1,2) | 0.955 | -0.000 | 0.082 | 0.084 | 0.978 | 0.326 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | (1,78) | 0.930 | -0.000 | 0.077 | 0.077 | 1.000 | 0.302 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.960 | -0.012 | 0.108 | 0.115 | 0.936 | 0.448 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.950 | 0.009 | 0.079 | 0.082 | 0.966 | 0.319 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.970 | 0.000 | 0.072 | 0.076 | 0.946 | 0.297 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.940 | -0.004 | 0.121 | 0.117 | 1.035 | 0.456 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.955 | -0.000 | 0.082 | 0.084 | 0.978 | 0.326 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.930 | -0.000 | 0.077 | 0.077 | 1.000 | 0.302 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.960 | -0.012 | 0.108 | 0.115 | 0.936 | 0.451 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | 0.009 | 0.079 | 0.082 | 0.966 | 0.319 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.970 | 0.000 | 0.072 | 0.076 | 0.946 | 0.297 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.935 | 0.008 | 0.122 | 0.117 | 1.047 | 0.456 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.955 | -0.000 | 0.083 | 0.084 | 0.993 | 0.326 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.930 | -0.000 | 0.079 | 0.077 | 1.016 | 0.302 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.970 | -0.000 | 0.109 | 0.115 | 0.947 | 0.450 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | 0.009 | 0.080 | 0.082 | 0.983 | 0.319 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,78) | 0.970 | 0.000 | 0.073 | 0.076 | 0.960 | 0.297 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | (1,1) | 0.935 | 0.013 | 0.123 | 0.117 | 1.047 | 0.458 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | (1,2) | 0.955 | -0.000 | 0.083 | 0.084 | 0.991 | 0.327 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | (1,78) | 0.930 | -0.001 | 0.079 | 0.078 | 1.016 | 0.303 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.980 | 0.005 | 0.110 | 0.116 | 0.949 | 0.453 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | 0.009 | 0.081 | 0.082 | 0.984 | 0.320 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | (1,78) | 0.965 | 0.000 | 0.073 | 0.076 | 0.957 | 0.298 | 0 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.960 | 0.016 | 0.157 | 0.164 | 0.957 | 0.641 | 0 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.945 | -0.008 | 0.127 | 0.124 | 1.027 | 0.481 | 0 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | (1,38) | 0.950 | -0.011 | 0.122 | 0.125 | 0.977 | 0.486 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.960 | 0.023 | 0.162 | 0.164 | 0.986 | 0.634 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.945 | -0.010 | 0.130 | 0.124 | 1.049 | 0.481 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | (1,38) | 0.965 | -0.012 | 0.119 | 0.124 | 0.961 | 0.481 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.950 | -0.002 | 0.157 | 0.165 | 0.951 | 0.638 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.960 | -0.006 | 0.127 | 0.125 | 1.013 | 0.484 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.970 | -0.011 | 0.116 | 0.125 | 0.930 | 0.484 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.960 | -0.002 | 0.157 | 0.165 | 0.951 | 0.645 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.960 | -0.006 | 0.127 | 0.125 | 1.013 | 0.484 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.970 | -0.011 | 0.116 | 0.125 | 0.930 | 0.484 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.955 | 0.019 | 0.160 | 0.165 | 0.970 | 0.645 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | -0.009 | 0.130 | 0.125 | 1.045 | 0.484 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,38) | 0.965 | -0.012 | 0.119 | 0.125 | 0.957 | 0.484 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.950 | 0.023 | 0.162 | 0.166 | 0.977 | 0.647 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | -0.010 | 0.130 | 0.125 | 1.039 | 0.486 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | (1,38) | 0.965 | -0.012 | 0.119 | 0.125 | 0.952 | 0.486 | 0 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.960 | 0.016 | 0.157 | 0.164 | 0.957 | 0.641 | 0 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.945 | -0.008 | 0.127 | 0.124 | 1.027 | 0.481 | 0 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | (1,38) | 0.950 | -0.011 | 0.122 | 0.125 | 0.977 | 0.486 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.960 | 0.022 | 0.162 | 0.164 | 0.990 | 0.634 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.945 | -0.010 | 0.130 | 0.124 | 1.045 | 0.481 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | (1,38) | 0.965 | -0.012 | 0.119 | 0.124 | 0.963 | 0.481 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.950 | -0.003 | 0.158 | 0.165 | 0.955 | 0.638 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | -0.006 | 0.126 | 0.125 | 1.011 | 0.485 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.970 | -0.012 | 0.116 | 0.125 | 0.933 | 0.484 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.960 | -0.003 | 0.158 | 0.165 | 0.955 | 0.645 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | -0.006 | 0.126 | 0.125 | 1.011 | 0.485 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.970 | -0.012 | 0.116 | 0.125 | 0.933 | 0.484 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.955 | 0.019 | 0.161 | 0.165 | 0.974 | 0.645 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.945 | -0.010 | 0.130 | 0.125 | 1.042 | 0.485 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,38) | 0.965 | -0.012 | 0.120 | 0.125 | 0.960 | 0.484 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.950 | 0.022 | 0.162 | 0.166 | 0.980 | 0.647 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.945 | -0.010 | 0.130 | 0.125 | 1.035 | 0.486 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | (1,38) | 0.965 | -0.012 | 0.119 | 0.125 | 0.954 | 0.485 | 0 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.920 | 0.037 | 0.210 | 0.189 | 1.111 | 0.731 | 0 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | (1,158) | 0.955 | -0.005 | 0.119 | 0.123 | 0.969 | 0.475 | 0 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.935 | -0.000 | 0.143 | 0.134 | 1.074 | 0.515 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.925 | 0.066 | 0.225 | 0.191 | 1.178 | 0.727 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | (1,158) | 0.940 | -0.007 | 0.126 | 0.125 | 1.014 | 0.480 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.955 | -0.006 | 0.150 | 0.135 | 1.111 | 0.520 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.915 | 0.023 | 0.212 | 0.191 | 1.109 | 0.727 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,158) | 0.970 | -0.007 | 0.120 | 0.125 | 0.960 | 0.483 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | -0.004 | 0.143 | 0.136 | 1.048 | 0.525 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.023 | 0.212 | 0.191 | 1.109 | 0.740 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.970 | -0.007 | 0.120 | 0.125 | 0.960 | 0.483 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | -0.004 | 0.143 | 0.136 | 1.048 | 0.525 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.910 | 0.052 | 0.219 | 0.191 | 1.145 | 0.739 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,158) | 0.950 | -0.006 | 0.125 | 0.125 | 1.001 | 0.483 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | -0.004 | 0.149 | 0.136 | 1.090 | 0.525 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.925 | 0.066 | 0.225 | 0.194 | 1.162 | 0.748 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | (1,158) | 0.945 | -0.007 | 0.126 | 0.126 | 1.000 | 0.487 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.960 | -0.006 | 0.150 | 0.137 | 1.096 | 0.527 | 0 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.920 | 0.037 | 0.210 | 0.189 | 1.111 | 0.731 | 0 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | (1,158) | 0.955 | -0.005 | 0.119 | 0.123 | 0.969 | 0.475 | 0 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.935 | -0.000 | 0.143 | 0.134 | 1.074 | 0.515 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.925 | 0.067 | 0.224 | 0.191 | 1.174 | 0.727 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | (1,158) | 0.945 | -0.006 | 0.127 | 0.125 | 1.020 | 0.480 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.955 | -0.005 | 0.150 | 0.135 | 1.114 | 0.520 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.915 | 0.024 | 0.211 | 0.191 | 1.106 | 0.727 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,158) | 0.970 | -0.006 | 0.121 | 0.125 | 0.967 | 0.482 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | -0.003 | 0.143 | 0.136 | 1.049 | 0.524 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.024 | 0.211 | 0.191 | 1.106 | 0.740 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.970 | -0.006 | 0.121 | 0.125 | 0.967 | 0.482 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | -0.003 | 0.143 | 0.136 | 1.049 | 0.524 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.910 | 0.053 | 0.218 | 0.191 | 1.142 | 0.739 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,158) | 0.950 | -0.006 | 0.126 | 0.125 | 1.008 | 0.482 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | -0.004 | 0.149 | 0.136 | 1.091 | 0.524 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.925 | 0.067 | 0.224 | 0.194 | 1.158 | 0.748 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | (1,158) | 0.950 | -0.006 | 0.127 | 0.126 | 1.006 | 0.486 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.960 | -0.005 | 0.150 | 0.137 | 1.099 | 0.527 | 0 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.935 | 0.051 | 0.209 | 0.196 | 1.065 | 0.761 | 0 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.950 | -0.006 | 0.133 | 0.129 | 1.035 | 0.495 | 0 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | (1,78) | 0.915 | -0.018 | 0.135 | 0.119 | 1.135 | 0.460 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.915 | 0.116 | 0.205 | 0.197 | 1.045 | 0.749 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.955 | -0.010 | 0.131 | 0.131 | 1.004 | 0.504 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | (1,78) | 0.925 | -0.015 | 0.140 | 0.121 | 1.158 | 0.467 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.950 | 0.065 | 0.200 | 0.197 | 1.015 | 0.750 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.965 | -0.012 | 0.128 | 0.130 | 0.985 | 0.502 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.930 | -0.013 | 0.136 | 0.120 | 1.125 | 0.466 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.910 | 0.065 | 0.200 | 0.197 | 1.015 | 0.764 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.965 | -0.012 | 0.128 | 0.130 | 0.985 | 0.502 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.930 | -0.013 | 0.136 | 0.120 | 1.125 | 0.466 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.895 | 0.096 | 0.204 | 0.197 | 1.038 | 0.763 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.960 | -0.011 | 0.131 | 0.130 | 1.007 | 0.502 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,78) | 0.920 | -0.014 | 0.139 | 0.120 | 1.151 | 0.466 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.890 | 0.116 | 0.205 | 0.199 | 1.035 | 0.769 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.955 | -0.010 | 0.131 | 0.132 | 0.994 | 0.509 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | (1,78) | 0.925 | -0.015 | 0.140 | 0.122 | 1.147 | 0.472 | 0 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.935 | 0.051 | 0.209 | 0.196 | 1.065 | 0.761 | 0 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.950 | -0.006 | 0.133 | 0.129 | 1.035 | 0.495 | 0 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | (1,78) | 0.915 | -0.018 | 0.135 | 0.119 | 1.135 | 0.460 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.915 | 0.116 | 0.206 | 0.197 | 1.046 | 0.749 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.955 | -0.010 | 0.131 | 0.131 | 1.002 | 0.504 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | (1,78) | 0.925 | -0.015 | 0.140 | 0.121 | 1.158 | 0.467 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.950 | 0.064 | 0.200 | 0.197 | 1.017 | 0.750 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.965 | -0.012 | 0.128 | 0.130 | 0.983 | 0.503 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.930 | -0.013 | 0.136 | 0.120 | 1.125 | 0.466 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.910 | 0.064 | 0.200 | 0.197 | 1.017 | 0.763 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.965 | -0.012 | 0.128 | 0.130 | 0.983 | 0.503 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.930 | -0.013 | 0.136 | 0.120 | 1.125 | 0.466 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.895 | 0.095 | 0.205 | 0.197 | 1.040 | 0.762 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.960 | -0.011 | 0.131 | 0.130 | 1.005 | 0.503 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,78) | 0.920 | -0.014 | 0.139 | 0.120 | 1.151 | 0.466 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.890 | 0.116 | 0.206 | 0.198 | 1.037 | 0.769 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.955 | -0.010 | 0.131 | 0.132 | 0.993 | 0.509 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | (1,78) | 0.925 | -0.015 | 0.140 | 0.122 | 1.147 | 0.472 | 0 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.930 | 0.012 | 0.118 | 0.117 | 1.015 | 0.456 | 0 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.975 | 0.000 | 0.081 | 0.082 | 0.985 | 0.319 | 0 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.056 | 0.300 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.915 | 0.022 | 0.121 | 0.117 | 1.036 | 0.454 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.975 | -0.000 | 0.082 | 0.082 | 0.995 | 0.319 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.052 | 0.299 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 0.005 | 0.119 | 0.117 | 1.016 | 0.455 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.975 | -0.000 | 0.080 | 0.082 | 0.974 | 0.320 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.950 | -0.004 | 0.079 | 0.077 | 1.032 | 0.300 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.005 | 0.119 | 0.117 | 1.016 | 0.458 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.975 | -0.000 | 0.080 | 0.082 | 0.974 | 0.320 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.950 | -0.004 | 0.079 | 0.077 | 1.032 | 0.300 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.925 | 0.017 | 0.120 | 0.117 | 1.027 | 0.458 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.975 | -0.000 | 0.081 | 0.082 | 0.992 | 0.320 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.047 | 0.300 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.930 | 0.022 | 0.121 | 0.118 | 1.031 | 0.460 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.975 | -0.000 | 0.082 | 0.082 | 0.989 | 0.321 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.046 | 0.301 | 0 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.930 | 0.012 | 0.118 | 0.117 | 1.015 | 0.456 | 0 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.975 | 0.000 | 0.081 | 0.082 | 0.985 | 0.319 | 0 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.056 | 0.300 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.915 | 0.022 | 0.121 | 0.117 | 1.038 | 0.454 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.975 | -0.000 | 0.082 | 0.082 | 0.996 | 0.319 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.051 | 0.299 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 0.005 | 0.119 | 0.117 | 1.018 | 0.455 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.975 | -0.000 | 0.080 | 0.082 | 0.975 | 0.320 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.950 | -0.004 | 0.079 | 0.077 | 1.031 | 0.300 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.920 | 0.005 | 0.119 | 0.117 | 1.018 | 0.458 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.975 | -0.000 | 0.080 | 0.082 | 0.975 | 0.320 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.950 | -0.004 | 0.079 | 0.077 | 1.031 | 0.300 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.925 | 0.017 | 0.121 | 0.117 | 1.029 | 0.458 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.975 | -0.000 | 0.081 | 0.082 | 0.993 | 0.320 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.046 | 0.300 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.930 | 0.022 | 0.121 | 0.118 | 1.032 | 0.460 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.975 | -0.000 | 0.082 | 0.082 | 0.991 | 0.321 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | (1,78) | 0.940 | -0.004 | 0.081 | 0.077 | 1.045 | 0.301 | 0 |

## Curvewise bands

| Scenario | Method | Index | Entry | Coverage | Mean width | Failures |
|---|---|---|---|---|---|---|
| cv_chain_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.905 | 0.852 | 0 |
| cv_chain_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.935 | 0.636 | 0 |
| cv_chain_minpilot_r10 | oracle_hc3_log | known | (1,38) | 0.955 | 0.617 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.930 | 0.833 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.925 | 0.636 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_block | known | (1,38) | 0.960 | 0.620 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 0.837 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.935 | 0.642 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.980 | 0.622 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.910 | 0.860 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.935 | 0.642 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.980 | 0.622 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.900 | 0.859 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.925 | 0.642 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,38) | 0.970 | 0.622 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.905 | 0.863 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.925 | 0.643 | 0 |
| cv_chain_minpilot_r10 | sqrt_refit_hc3_log | known | (1,38) | 0.965 | 0.627 | 0 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.905 | 0.852 | 0 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.935 | 0.636 | 0 |
| cv_chain_minpilot_r15 | oracle_hc3_log | known | (1,38) | 0.955 | 0.617 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.930 | 0.833 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.925 | 0.636 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_block | known | (1,38) | 0.960 | 0.620 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 0.837 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.935 | 0.642 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.980 | 0.621 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.910 | 0.860 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.935 | 0.642 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.980 | 0.621 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.900 | 0.859 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.925 | 0.642 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,38) | 0.970 | 0.621 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.905 | 0.863 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.925 | 0.643 | 0 |
| cv_chain_minpilot_r15 | sqrt_refit_hc3_log | known | (1,38) | 0.965 | 0.627 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | (1,1) | 0.915 | 0.700 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | (1,2) | 0.905 | 0.500 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | estimated | (1,78) | 0.955 | 0.453 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.905 | 0.700 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.955 | 0.499 | 0 |
| cv_generated_minpilot_r10 | oracle_hc3_log | known | (1,78) | 0.940 | 0.452 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | (1,1) | 0.905 | 0.690 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | (1,2) | 0.915 | 0.502 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | estimated | (1,78) | 0.955 | 0.453 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.895 | 0.685 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.950 | 0.500 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_block | known | (1,78) | 0.950 | 0.450 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | (1,1) | 0.890 | 0.691 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | (1,2) | 0.920 | 0.503 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | estimated | (1,78) | 0.960 | 0.454 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.895 | 0.685 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.965 | 0.501 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.950 | 0.451 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.915 | 0.701 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.920 | 0.503 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.960 | 0.454 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.910 | 0.695 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.965 | 0.501 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.950 | 0.451 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.910 | 0.701 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.915 | 0.503 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.960 | 0.454 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.910 | 0.695 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | 0.501 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,78) | 0.950 | 0.451 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | (1,1) | 0.920 | 0.705 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | (1,2) | 0.915 | 0.505 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | estimated | (1,78) | 0.955 | 0.455 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.905 | 0.699 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | 0.504 | 0 |
| cv_generated_minpilot_r10 | sqrt_refit_hc3_log | known | (1,78) | 0.955 | 0.453 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | (1,1) | 0.915 | 0.700 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | (1,2) | 0.905 | 0.500 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | estimated | (1,78) | 0.955 | 0.453 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.905 | 0.700 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.955 | 0.499 | 0 |
| cv_generated_minpilot_r15 | oracle_hc3_log | known | (1,78) | 0.940 | 0.452 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | (1,1) | 0.905 | 0.690 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | (1,2) | 0.915 | 0.502 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | estimated | (1,78) | 0.955 | 0.452 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.890 | 0.684 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.950 | 0.500 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_block | known | (1,78) | 0.950 | 0.450 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | (1,1) | 0.890 | 0.691 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | (1,2) | 0.920 | 0.502 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | estimated | (1,78) | 0.960 | 0.454 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.895 | 0.684 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.965 | 0.501 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.950 | 0.451 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | (1,1) | 0.915 | 0.701 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | (1,2) | 0.920 | 0.502 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | estimated | (1,78) | 0.960 | 0.454 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.915 | 0.695 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.965 | 0.501 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.950 | 0.451 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | (1,1) | 0.910 | 0.701 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | (1,2) | 0.915 | 0.502 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | estimated | (1,78) | 0.960 | 0.454 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.910 | 0.695 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | 0.501 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,78) | 0.950 | 0.451 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | (1,1) | 0.920 | 0.705 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | (1,2) | 0.915 | 0.505 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | estimated | (1,78) | 0.955 | 0.455 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.905 | 0.699 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | 0.503 | 0 |
| cv_generated_minpilot_r15 | sqrt_refit_hc3_log | known | (1,78) | 0.955 | 0.452 | 0 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.925 | 0.975 | 0 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.965 | 0.721 | 0 |
| cv_grid_minpilot_r10 | oracle_hc3_log | known | (1,38) | 0.975 | 0.720 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.935 | 0.947 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.950 | 0.721 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_block | known | (1,38) | 0.970 | 0.718 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 0.954 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | 0.728 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.985 | 0.723 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.940 | 0.979 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.728 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.985 | 0.723 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.935 | 0.978 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | 0.728 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,38) | 0.970 | 0.723 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.930 | 0.981 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.955 | 0.729 | 0 |
| cv_grid_minpilot_r10 | sqrt_refit_hc3_log | known | (1,38) | 0.975 | 0.726 | 0 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.925 | 0.975 | 0 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.965 | 0.721 | 0 |
| cv_grid_minpilot_r15 | oracle_hc3_log | known | (1,38) | 0.975 | 0.720 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.935 | 0.947 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.950 | 0.720 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_block | known | (1,38) | 0.970 | 0.717 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 0.954 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | 0.727 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,38) | 0.985 | 0.723 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.940 | 0.979 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.727 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,38) | 0.985 | 0.723 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.935 | 0.978 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.950 | 0.727 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,38) | 0.970 | 0.723 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.930 | 0.981 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.955 | 0.728 | 0 |
| cv_grid_minpilot_r15 | sqrt_refit_hc3_log | known | (1,38) | 0.975 | 0.725 | 0 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.905 | 1.099 | 0 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | (1,158) | 0.950 | 0.712 | 0 |
| cv_highdim_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.945 | 0.780 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.875 | 1.060 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | (1,158) | 0.960 | 0.714 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.935 | 0.787 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.870 | 1.062 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,158) | 0.955 | 0.720 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | 0.795 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.875 | 1.102 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.955 | 0.720 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.795 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.870 | 1.099 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,158) | 0.955 | 0.720 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.935 | 0.795 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.870 | 1.114 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | (1,158) | 0.960 | 0.725 | 0 |
| cv_highdim_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | 0.799 | 0 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.905 | 1.099 | 0 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | (1,158) | 0.950 | 0.712 | 0 |
| cv_highdim_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.945 | 0.780 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.875 | 1.059 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | (1,158) | 0.960 | 0.714 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.940 | 0.787 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.865 | 1.061 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,158) | 0.955 | 0.719 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.955 | 0.795 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.875 | 1.101 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,158) | 0.955 | 0.719 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.955 | 0.795 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.870 | 1.099 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,158) | 0.955 | 0.719 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.935 | 0.795 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.870 | 1.113 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | (1,158) | 0.960 | 0.724 | 0 |
| cv_highdim_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.950 | 0.799 | 0 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.900 | 1.169 | 0 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.940 | 0.765 | 0 |
| cv_t8_minpilot_r10 | oracle_hc3_log | known | (1,78) | 0.915 | 0.695 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.930 | 1.108 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.940 | 0.778 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_block | known | (1,78) | 0.945 | 0.701 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 1.109 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.950 | 0.776 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.955 | 0.700 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.895 | 1.153 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | 0.776 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.955 | 0.700 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.855 | 1.150 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.940 | 0.776 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,78) | 0.950 | 0.700 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.835 | 1.161 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.940 | 0.786 | 0 |
| cv_t8_minpilot_r10 | sqrt_refit_hc3_log | known | (1,78) | 0.945 | 0.709 | 0 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.900 | 1.169 | 0 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.940 | 0.765 | 0 |
| cv_t8_minpilot_r15 | oracle_hc3_log | known | (1,78) | 0.915 | 0.695 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.930 | 1.108 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.940 | 0.778 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_block | known | (1,78) | 0.945 | 0.701 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.940 | 1.108 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.950 | 0.776 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.955 | 0.700 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.895 | 1.153 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.950 | 0.776 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.955 | 0.700 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.855 | 1.150 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.940 | 0.776 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,78) | 0.950 | 0.700 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.835 | 1.161 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.940 | 0.786 | 0 |
| cv_t8_minpilot_r15 | sqrt_refit_hc3_log | known | (1,78) | 0.945 | 0.709 | 0 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | (1,1) | 0.935 | 0.700 | 0 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | (1,2) | 0.955 | 0.499 | 0 |
| cv_varying_minpilot_r10 | oracle_hc3_log | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | (1,1) | 0.920 | 0.685 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | (1,2) | 0.975 | 0.499 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_block | known | (1,78) | 0.965 | 0.453 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.895 | 0.685 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.975 | 0.500 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.915 | 0.695 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.975 | 0.500 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,1) | 0.915 | 0.695 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,2) | 0.965 | 0.500 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_df_hc3_log | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | (1,1) | 0.915 | 0.699 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | (1,2) | 0.975 | 0.502 | 0 |
| cv_varying_minpilot_r10 | sqrt_refit_hc3_log | known | (1,78) | 0.965 | 0.455 | 0 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | (1,1) | 0.935 | 0.700 | 0 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | (1,2) | 0.955 | 0.499 | 0 |
| cv_varying_minpilot_r15 | oracle_hc3_log | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | (1,1) | 0.920 | 0.684 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | (1,2) | 0.975 | 0.499 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_block | known | (1,78) | 0.965 | 0.452 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,1) | 0.895 | 0.685 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,2) | 0.975 | 0.500 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,1) | 0.910 | 0.695 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,2) | 0.975 | 0.500 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_bc_log | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,1) | 0.915 | 0.695 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,2) | 0.965 | 0.500 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_df_hc3_log | known | (1,78) | 0.965 | 0.454 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | (1,1) | 0.910 | 0.699 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | (1,2) | 0.975 | 0.502 | 0 |
| cv_varying_minpilot_r15 | sqrt_refit_hc3_log | known | (1,78) | 0.965 | 0.455 | 0 |

## Paired rule comparisons

The seed identifies the same generated dataset. The comparison includes support selection, bandwidth selection, and index estimation. Width ratios use the same replications.

| Scenario | Method | Reference | Index | Kind | NMC | Coverage difference | MCSE | Width ratio |
|---|---|---|---|---|---|---|---|---|
