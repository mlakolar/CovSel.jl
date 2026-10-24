# Simulation summary

Failures count as noncoverage. Widths and bias use valid fits only.

| Scenario | Method | Index | h factor | pilot factor | penalty | target | nMC | failures | coverage | MCSE | width | bias | SD / RMS SE |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_window_cv | sqrtlasso | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.940 | 0.017 | 0.746 | nan | nan |
| fixed_window_cv | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.945 | 0.016 | 0.556 | 0.054 | 1.012 |
| fixed_window_cv | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.930 | 0.018 | 0.400 | -0.017 | 1.108 |
| fixed_window_cv | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.920 | 0.019 | 0.376 | -0.006 | 1.072 |
| grid_21 | oracle | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.920 | 0.019 | 0.661 | nan | nan |
| grid_21 | oracle | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.950 | 0.015 | 0.492 | 0.022 | 1.003 |
| grid_21 | oracle | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.935 | 0.017 | 0.343 | -0.007 | 1.040 |
| grid_21 | oracle | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.930 | 0.018 | 0.323 | 0.008 | 1.106 |
| grid_21 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.945 | 0.016 | 0.700 | nan | nan |
| grid_21 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.935 | 0.017 | 0.515 | 0.076 | 1.025 |
| grid_21 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.915 | 0.020 | 0.360 | -0.009 | 1.067 |
| grid_21 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.925 | 0.019 | 0.341 | 0.009 | 1.094 |
| grid_21 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.915 | 0.020 | 0.661 | nan | nan |
| grid_21 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.965 | 0.013 | 0.492 | 0.024 | 1.014 |
| grid_21 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.925 | 0.019 | 0.345 | -0.032 | 1.034 |
| grid_21 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.930 | 0.018 | 0.326 | 0.008 | 1.085 |
| grid_41 | oracle | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.920 | 0.019 | 0.661 | nan | nan |
| grid_41 | oracle | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.950 | 0.015 | 0.492 | 0.022 | 1.003 |
| grid_41 | oracle | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.935 | 0.017 | 0.343 | -0.007 | 1.040 |
| grid_41 | oracle | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.930 | 0.018 | 0.323 | 0.008 | 1.106 |
| grid_41 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.940 | 0.017 | 0.703 | nan | nan |
| grid_41 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.935 | 0.017 | 0.515 | 0.076 | 1.025 |
| grid_41 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.915 | 0.020 | 0.360 | -0.009 | 1.067 |
| grid_41 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.925 | 0.019 | 0.341 | 0.009 | 1.094 |
| grid_41 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.910 | 0.020 | 0.662 | nan | nan |
| grid_41 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.965 | 0.013 | 0.492 | 0.024 | 1.014 |
| grid_41 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.925 | 0.019 | 0.345 | -0.032 | 1.034 |
| grid_41 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.930 | 0.018 | 0.326 | 0.008 | 1.085 |
| grid_81 | oracle | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.920 | 0.019 | 0.661 | nan | nan |
| grid_81 | oracle | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.950 | 0.015 | 0.492 | 0.022 | 1.003 |
| grid_81 | oracle | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.935 | 0.017 | 0.343 | -0.007 | 1.040 |
| grid_81 | oracle | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.930 | 0.018 | 0.323 | 0.008 | 1.106 |
| grid_81 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.935 | 0.017 | 0.704 | nan | nan |
| grid_81 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.935 | 0.017 | 0.515 | 0.076 | 1.025 |
| grid_81 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.915 | 0.020 | 0.360 | -0.009 | 1.067 |
| grid_81 | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.925 | 0.019 | 0.341 | 0.009 | 1.094 |
| grid_81 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | family | 200 | 0 | 0.905 | 0.021 | 0.662 | nan | nan |
| grid_81 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 200 | 0 | 0.965 | 0.013 | 0.492 | 0.024 | 1.014 |
| grid_81 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 200 | 0 | 0.925 | 0.019 | 0.345 | -0.032 | 1.034 |
| grid_81 | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 200 | 0 | 0.930 | 0.018 | 0.326 | 0.008 | 1.085 |
| refit_baseline | aclime_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.777 | 0.024 | 0.750 | nan | nan |
| refit_baseline | aclime_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.923 | 0.015 | 0.551 | -0.053 | 0.909 |
| refit_baseline | aclime_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.890 | 0.018 | 0.404 | -0.075 | 0.974 |
| refit_baseline | aclime_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.950 | 0.013 | 0.383 | -0.001 | 0.975 |
| refit_baseline | aclime_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.323 | 0.027 | 0.474 | nan | nan |
| refit_baseline | aclime_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.750 | 0.025 | 0.363 | -0.095 | 1.172 |
| refit_baseline | aclime_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.883 | 0.019 | 0.263 | -0.049 | 0.968 |
| refit_baseline | aclime_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.947 | 0.013 | 0.255 | -0.001 | 0.962 |
| refit_baseline | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.943 | 0.013 | 0.808 | nan | nan |
| refit_baseline | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.950 | 0.013 | 0.597 | 0.031 | 0.935 |
| refit_baseline | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.957 | 0.012 | 0.434 | 0.002 | 0.988 |
| refit_baseline | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.957 | 0.012 | 0.400 | 0.002 | 0.994 |
| refit_baseline | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.897 | 0.018 | 0.767 | nan | nan |
| refit_baseline | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.950 | 0.013 | 0.582 | 0.027 | 0.949 |
| refit_baseline | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.957 | 0.012 | 0.421 | 0.001 | 0.983 |
| refit_baseline | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.950 | 0.013 | 0.387 | 0.001 | 0.978 |
| refit_baseline | oracle | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.963 | 0.011 | 0.803 | nan | nan |
| refit_baseline | oracle | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.960 | 0.011 | 0.590 | 0.022 | 0.906 |
| refit_baseline | oracle | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.957 | 0.012 | 0.429 | 0.009 | 0.997 |
| refit_baseline | oracle | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.947 | 0.013 | 0.396 | 0.001 | 1.016 |
| refit_baseline | scaled_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.933 | 0.014 | 0.794 | nan | nan |
| refit_baseline | scaled_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.963 | 0.011 | 0.583 | 0.008 | 0.919 |
| refit_baseline | scaled_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.943 | 0.013 | 0.425 | -0.037 | 1.005 |
| refit_baseline | scaled_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.957 | 0.012 | 0.397 | -0.000 | 0.980 |
| refit_baseline | scaled_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.893 | 0.018 | 0.770 | nan | nan |
| refit_baseline | scaled_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.957 | 0.012 | 0.582 | 0.003 | 0.917 |
| refit_baseline | scaled_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.940 | 0.014 | 0.422 | -0.038 | 0.997 |
| refit_baseline | scaled_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.953 | 0.012 | 0.390 | -0.000 | 0.967 |
| refit_baseline | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.950 | 0.013 | 0.875 | nan | nan |
| refit_baseline | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.940 | 0.014 | 0.637 | 0.105 | 0.974 |
| refit_baseline | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.960 | 0.011 | 0.462 | 0.008 | 1.029 |
| refit_baseline | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.963 | 0.011 | 0.428 | 0.003 | 0.983 |
| refit_baseline | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.830 | 0.022 | 0.821 | nan | nan |
| refit_baseline | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.900 | 0.017 | 0.616 | 0.100 | 0.988 |
| refit_baseline | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.953 | 0.012 | 0.443 | 0.006 | 1.019 |
| refit_baseline | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.960 | 0.011 | 0.406 | 0.003 | 0.976 |
| refit_baseline | sqrt_refit | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.950 | 0.013 | 0.898 | nan | nan |
| refit_baseline | sqrt_refit | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.920 | 0.016 | 0.652 | 0.130 | 1.013 |
| refit_baseline | sqrt_refit | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.953 | 0.012 | 0.474 | 0.012 | 1.022 |
| refit_baseline | sqrt_refit | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.963 | 0.011 | 0.435 | 0.004 | 0.987 |
| refit_baseline | sqrtlasso | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.940 | 0.014 | 0.804 | nan | nan |
| refit_baseline | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.960 | 0.011 | 0.591 | 0.024 | 0.922 |
| refit_baseline | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.950 | 0.013 | 0.431 | -0.028 | 0.978 |
| refit_baseline | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.953 | 0.012 | 0.401 | 0.000 | 0.981 |
| refit_varying | aclime_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.820 | 0.022 | 0.759 | nan | nan |
| refit_varying | aclime_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.907 | 0.017 | 0.560 | -0.051 | 0.964 |
| refit_varying | aclime_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.847 | 0.021 | 0.404 | -0.091 | 1.037 |
| refit_varying | aclime_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.970 | 0.010 | 0.385 | -0.001 | 1.032 |
| refit_varying | aclime_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.337 | 0.027 | 0.477 | nan | nan |
| refit_varying | aclime_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.737 | 0.025 | 0.365 | -0.095 | 1.243 |
| refit_varying | aclime_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.843 | 0.021 | 0.259 | -0.058 | 1.025 |
| refit_varying | aclime_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.960 | 0.011 | 0.254 | -0.001 | 1.005 |
| refit_varying | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.907 | 0.017 | 0.813 | nan | nan |
| refit_varying | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.947 | 0.013 | 0.604 | 0.036 | 1.004 |
| refit_varying | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.940 | 0.014 | 0.433 | -0.015 | 1.071 |
| refit_varying | aclime_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.977 | 0.009 | 0.403 | -0.003 | 1.040 |
| refit_varying | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.910 | 0.017 | 0.772 | nan | nan |
| refit_varying | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.930 | 0.015 | 0.585 | 0.031 | 1.014 |
| refit_varying | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.940 | 0.014 | 0.414 | -0.015 | 1.056 |
| refit_varying | aclime_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.970 | 0.010 | 0.386 | -0.003 | 1.013 |
| refit_varying | oracle | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.927 | 0.015 | 0.808 | nan | nan |
| refit_varying | oracle | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.967 | 0.010 | 0.598 | 0.024 | 0.951 |
| refit_varying | oracle | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.950 | 0.013 | 0.427 | -0.006 | 1.071 |
| refit_varying | oracle | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.970 | 0.010 | 0.398 | -0.002 | 1.053 |
| refit_varying | scaled_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.907 | 0.017 | 0.800 | nan | nan |
| refit_varying | scaled_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.940 | 0.014 | 0.592 | 0.011 | 0.974 |
| refit_varying | scaled_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.917 | 0.016 | 0.424 | -0.054 | 1.068 |
| refit_varying | scaled_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.973 | 0.009 | 0.401 | -0.001 | 1.029 |
| refit_varying | scaled_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.900 | 0.017 | 0.776 | nan | nan |
| refit_varying | scaled_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.937 | 0.014 | 0.585 | 0.006 | 0.974 |
| refit_varying | scaled_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.913 | 0.016 | 0.414 | -0.054 | 1.052 |
| refit_varying | scaled_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.970 | 0.010 | 0.388 | -0.001 | 1.005 |
| refit_varying | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.940 | 0.014 | 0.880 | nan | nan |
| refit_varying | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.920 | 0.016 | 0.645 | 0.113 | 1.036 |
| refit_varying | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.947 | 0.013 | 0.462 | -0.010 | 1.069 |
| refit_varying | scaled_refit_block | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.967 | 0.010 | 0.432 | 0.000 | 1.018 |
| refit_varying | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.820 | 0.022 | 0.822 | nan | nan |
| refit_varying | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.877 | 0.019 | 0.613 | 0.106 | 1.060 |
| refit_varying | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.937 | 0.014 | 0.435 | -0.011 | 1.055 |
| refit_varying | scaled_refit_onestep | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.963 | 0.011 | 0.403 | 0.000 | 1.015 |
| refit_varying | sqrt_refit | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.930 | 0.015 | 0.901 | nan | nan |
| refit_varying | sqrt_refit | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.917 | 0.016 | 0.660 | 0.137 | 1.047 |
| refit_varying | sqrt_refit | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.950 | 0.013 | 0.472 | -0.007 | 1.057 |
| refit_varying | sqrt_refit | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.963 | 0.011 | 0.439 | 0.002 | 1.028 |
| refit_varying | sqrtlasso | known | 1.0 | 1.0 | 0.7 | family | 300 | 0 | 0.913 | 0.016 | 0.811 | nan | nan |
| refit_varying | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,1) at 0 | 300 | 0 | 0.937 | 0.014 | 0.600 | 0.027 | 0.980 |
| refit_varying | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,2) at 0 | 300 | 0 | 0.933 | 0.014 | 0.430 | -0.045 | 1.041 |
| refit_varying | sqrtlasso | known | 1.0 | 1.0 | 0.7 | (1,38) at 0 | 300 | 0 | 0.973 | 0.009 | 0.404 | -0.001 | 1.033 |
