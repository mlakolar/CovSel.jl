# Independent paper-focused confirmation

Each configuration uses 1000 replications, 1999 Gaussian observation-multiplier draws and a 21-point index grid. Failures count as noncoverage. The calibration data and this study use different seeds. The rules and comparison constants were fixed before these outcomes.

The two fixed-rule scenarios use pilot bandwidth 1.6 times the final bandwidth and prespecified penalty multipliers 0.35 and 0.7. The CV scenarios use the final fixed-window selector; their nominal-factor column labels the requested configuration, while the actual selected multipliers are reported in the selection-frequency table. They assess procedures on the supplied index; the estimated-index mode also estimates that index from the same data.

| Scenario | Method | Index | Nominal factor | NMC | Failures | Joint coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|---|
| block_wide_pilot | aclime_refit_block | known | 0.35 | 1000 | 0 | 0.953 | [0.938, 0.964] | 0.906 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | 1000 | 0 | 0.934 | [0.917, 0.948] | 0.811 |
| block_wide_pilot | oracle | known | 0.35 | 1000 | 0 | 0.934 | [0.917, 0.948] | 0.805 |
| block_wide_pilot | oracle | known | 0.7 | 1000 | 0 | 0.934 | [0.917, 0.948] | 0.805 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | 1000 | 0 | 0.943 | [0.927, 0.956] | 0.944 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | 1000 | 0 | 0.957 | [0.943, 0.968] | 0.851 |
| block_wide_pilot | sqrtlasso | known | 0.35 | 1000 | 0 | 0.967 | [0.954, 0.976] | 0.883 |
| block_wide_pilot | sqrtlasso | known | 0.7 | 1000 | 0 | 0.918 | [0.899, 0.933] | 0.803 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | 1000 | 0 | 0.907 | [0.887, 0.923] | 0.525 |
| cv_generated_index | sqrtlasso | known | 0.7 | 1000 | 0 | 0.928 | [0.910, 0.942] | 0.524 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | 1000 | 0 | 0.893 | [0.872, 0.911] | 0.899 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | 1000 | 0 | 0.915 | [0.896, 0.931] | 0.686 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | 1000 | 0 | 0.936 | [0.919, 0.950] | 0.614 |
| varying_wide_pilot | oracle | known | 0.35 | 1000 | 0 | 0.942 | [0.926, 0.955] | 0.612 |
| varying_wide_pilot | oracle | known | 0.7 | 1000 | 0 | 0.942 | [0.926, 0.955] | 0.612 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | 1000 | 0 | 0.838 | [0.814, 0.860] | 0.722 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | 1000 | 0 | 0.950 | [0.935, 0.962] | 0.641 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | 1000 | 0 | 0.932 | [0.915, 0.946] | 0.673 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | 1000 | 0 | 0.934 | [0.917, 0.948] | 0.612 |

## Pointwise coverage at zero

| Scenario | Method | Index | Nominal factor | Entry | Coverage | Wilson 95% CI | Bias | Empirical SD | RMS SE | Mean length | Failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block_wide_pilot | aclime_refit_block | known | 0.35 | (1,1) | 0.907 | [0.887, 0.923] | 0.133 | 0.181 | 0.169 | 0.651 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.35 | (1,2) | 0.949 | [0.934, 0.961] | -0.004 | 0.127 | 0.121 | 0.471 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.35 | (1,38) | 0.958 | [0.944, 0.969] | -0.001 | 0.118 | 0.114 | 0.443 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | (1,1) | 0.936 | [0.919, 0.950] | 0.026 | 0.160 | 0.154 | 0.592 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | (1,2) | 0.948 | [0.932, 0.960] | -0.004 | 0.116 | 0.111 | 0.431 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | (1,38) | 0.955 | [0.940, 0.966] | 0.001 | 0.106 | 0.104 | 0.402 | 0 |
| block_wide_pilot | oracle | known | 0.35 | (1,1) | 0.931 | [0.914, 0.945] | 0.011 | 0.158 | 0.151 | 0.583 | 0 |
| block_wide_pilot | oracle | known | 0.35 | (1,2) | 0.945 | [0.929, 0.958] | -0.001 | 0.113 | 0.110 | 0.425 | 0 |
| block_wide_pilot | oracle | known | 0.35 | (1,38) | 0.960 | [0.946, 0.970] | 0.000 | 0.105 | 0.103 | 0.398 | 0 |
| block_wide_pilot | oracle | known | 0.7 | (1,1) | 0.931 | [0.914, 0.945] | 0.011 | 0.158 | 0.151 | 0.583 | 0 |
| block_wide_pilot | oracle | known | 0.7 | (1,2) | 0.945 | [0.929, 0.958] | -0.001 | 0.113 | 0.110 | 0.425 | 0 |
| block_wide_pilot | oracle | known | 0.7 | (1,38) | 0.960 | [0.946, 0.970] | 0.000 | 0.105 | 0.103 | 0.398 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | (1,1) | 0.858 | [0.835, 0.878] | 0.181 | 0.190 | 0.176 | 0.677 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | (1,2) | 0.951 | [0.936, 0.963] | -0.004 | 0.134 | 0.126 | 0.489 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | (1,38) | 0.955 | [0.940, 0.966] | 0.000 | 0.123 | 0.119 | 0.461 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | (1,1) | 0.944 | [0.928, 0.957] | 0.072 | 0.170 | 0.160 | 0.617 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | (1,2) | 0.949 | [0.934, 0.961] | -0.004 | 0.120 | 0.115 | 0.448 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | (1,38) | 0.955 | [0.940, 0.966] | -0.001 | 0.111 | 0.108 | 0.420 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.35 | (1,1) | 0.927 | [0.909, 0.942] | 0.114 | 0.176 | 0.166 | 0.639 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.35 | (1,2) | 0.952 | [0.937, 0.964] | -0.009 | 0.124 | 0.120 | 0.464 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.35 | (1,38) | 0.955 | [0.940, 0.966] | -0.000 | 0.115 | 0.112 | 0.434 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.7 | (1,1) | 0.931 | [0.914, 0.945] | 0.015 | 0.158 | 0.152 | 0.586 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.7 | (1,2) | 0.945 | [0.929, 0.958] | -0.027 | 0.112 | 0.110 | 0.427 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.7 | (1,38) | 0.956 | [0.941, 0.967] | 0.001 | 0.106 | 0.104 | 0.402 | 0 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | (1,1) | 0.933 | [0.916, 0.947] | 0.030 | 0.112 | 0.103 | 0.400 | 0 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | (1,2) | 0.951 | [0.936, 0.963] | -0.007 | 0.075 | 0.074 | 0.287 | 0 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | (1,78) | 0.945 | [0.929, 0.958] | 0.000 | 0.072 | 0.069 | 0.267 | 0 |
| cv_generated_index | sqrtlasso | known | 0.7 | (1,1) | 0.911 | [0.892, 0.927] | 0.030 | 0.117 | 0.102 | 0.396 | 0 |
| cv_generated_index | sqrtlasso | known | 0.7 | (1,2) | 0.946 | [0.930, 0.958] | -0.012 | 0.072 | 0.072 | 0.281 | 0 |
| cv_generated_index | sqrtlasso | known | 0.7 | (1,78) | 0.950 | [0.935, 0.962] | -0.002 | 0.070 | 0.068 | 0.264 | 0 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | (1,1) | 0.922 | [0.904, 0.937] | 0.054 | 0.213 | 0.177 | 0.673 | 0 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | (1,158) | 0.950 | [0.935, 0.962] | 0.006 | 0.123 | 0.120 | 0.460 | 0 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | (1,2) | 0.930 | [0.912, 0.944] | -0.053 | 0.128 | 0.128 | 0.490 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | (1,1) | 0.848 | [0.824, 0.869] | 0.136 | 0.136 | 0.128 | 0.497 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | (1,2) | 0.954 | [0.939, 0.965] | -0.005 | 0.090 | 0.091 | 0.353 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | (1,78) | 0.945 | [0.929, 0.958] | 0.002 | 0.088 | 0.086 | 0.333 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | (1,1) | 0.940 | [0.924, 0.953] | 0.027 | 0.122 | 0.116 | 0.452 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | (1,2) | 0.956 | [0.941, 0.967] | -0.007 | 0.083 | 0.083 | 0.322 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | (1,78) | 0.947 | [0.931, 0.959] | 0.002 | 0.079 | 0.077 | 0.302 | 0 |
| varying_wide_pilot | oracle | known | 0.35 | (1,1) | 0.944 | [0.928, 0.957] | 0.019 | 0.121 | 0.116 | 0.448 | 0 |
| varying_wide_pilot | oracle | known | 0.35 | (1,2) | 0.955 | [0.940, 0.966] | -0.003 | 0.081 | 0.082 | 0.320 | 0 |
| varying_wide_pilot | oracle | known | 0.35 | (1,78) | 0.948 | [0.932, 0.960] | 0.002 | 0.078 | 0.077 | 0.299 | 0 |
| varying_wide_pilot | oracle | known | 0.7 | (1,1) | 0.944 | [0.928, 0.957] | 0.019 | 0.121 | 0.116 | 0.448 | 0 |
| varying_wide_pilot | oracle | known | 0.7 | (1,2) | 0.955 | [0.940, 0.966] | -0.003 | 0.081 | 0.082 | 0.320 | 0 |
| varying_wide_pilot | oracle | known | 0.7 | (1,78) | 0.948 | [0.932, 0.960] | 0.002 | 0.078 | 0.077 | 0.299 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | (1,1) | 0.741 | [0.713, 0.767] | 0.194 | 0.144 | 0.134 | 0.520 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | (1,2) | 0.951 | [0.936, 0.963] | -0.006 | 0.095 | 0.095 | 0.371 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | (1,78) | 0.937 | [0.920, 0.950] | 0.002 | 0.094 | 0.090 | 0.349 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | (1,1) | 0.931 | [0.914, 0.945] | 0.066 | 0.126 | 0.121 | 0.468 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | (1,2) | 0.956 | [0.941, 0.967] | -0.004 | 0.085 | 0.085 | 0.333 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | (1,78) | 0.951 | [0.936, 0.963] | 0.002 | 0.082 | 0.080 | 0.313 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | (1,1) | 0.875 | [0.853, 0.894] | 0.121 | 0.133 | 0.126 | 0.490 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | (1,2) | 0.952 | [0.937, 0.964] | -0.008 | 0.089 | 0.090 | 0.349 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | (1,78) | 0.943 | [0.927, 0.956] | 0.001 | 0.087 | 0.084 | 0.327 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | (1,1) | 0.943 | [0.927, 0.956] | 0.021 | 0.121 | 0.116 | 0.450 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | (1,2) | 0.950 | [0.935, 0.962] | -0.021 | 0.082 | 0.082 | 0.321 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | (1,78) | 0.946 | [0.930, 0.958] | 0.002 | 0.080 | 0.077 | 0.302 | 0 |

## Curvewise coverage by entry

| Scenario | Method | Index | Nominal factor | Entry | Curve coverage | Mean width | Failures |
|---|---|---|---|---|---|---|---|
| block_wide_pilot | aclime_refit_block | known | 0.35 | (1,1) | 0.934 | 1.000 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.35 | (1,2) | 0.942 | 0.730 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.35 | (1,38) | 0.952 | 0.673 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | (1,1) | 0.928 | 0.892 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | (1,2) | 0.951 | 0.659 | 0 |
| block_wide_pilot | aclime_refit_block | known | 0.7 | (1,38) | 0.957 | 0.598 | 0 |
| block_wide_pilot | oracle | known | 0.35 | (1,1) | 0.924 | 0.890 | 0 |
| block_wide_pilot | oracle | known | 0.35 | (1,2) | 0.947 | 0.649 | 0 |
| block_wide_pilot | oracle | known | 0.35 | (1,38) | 0.957 | 0.595 | 0 |
| block_wide_pilot | oracle | known | 0.7 | (1,1) | 0.924 | 0.890 | 0 |
| block_wide_pilot | oracle | known | 0.7 | (1,2) | 0.947 | 0.649 | 0 |
| block_wide_pilot | oracle | known | 0.7 | (1,38) | 0.957 | 0.595 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | (1,1) | 0.869 | 1.043 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | (1,2) | 0.942 | 0.760 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.35 | (1,38) | 0.946 | 0.701 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | (1,1) | 0.949 | 0.938 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | (1,2) | 0.949 | 0.688 | 0 |
| block_wide_pilot | scaled_refit_block | known | 0.7 | (1,38) | 0.951 | 0.631 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.35 | (1,1) | 0.953 | 0.974 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.35 | (1,2) | 0.946 | 0.715 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.35 | (1,38) | 0.949 | 0.652 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.7 | (1,1) | 0.924 | 0.881 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.7 | (1,2) | 0.939 | 0.651 | 0 |
| block_wide_pilot | sqrtlasso | known | 0.7 | (1,38) | 0.956 | 0.597 | 0 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | (1,1) | 0.897 | 0.578 | 0 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | (1,2) | 0.902 | 0.421 | 0 |
| cv_generated_index | sqrtlasso | estimated | 0.7 | (1,78) | 0.950 | 0.385 | 0 |
| cv_generated_index | sqrtlasso | known | 0.7 | (1,1) | 0.908 | 0.577 | 0 |
| cv_generated_index | sqrtlasso | known | 0.7 | (1,2) | 0.950 | 0.421 | 0 |
| cv_generated_index | sqrtlasso | known | 0.7 | (1,78) | 0.950 | 0.384 | 0 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | (1,1) | 0.878 | 0.972 | 0 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | (1,158) | 0.967 | 0.670 | 0 |
| cv_high_dimensional | sqrtlasso | known | 0.7 | (1,2) | 0.917 | 0.730 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | (1,1) | 0.866 | 0.764 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | (1,2) | 0.966 | 0.553 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.35 | (1,78) | 0.956 | 0.509 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | (1,1) | 0.935 | 0.681 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | (1,2) | 0.959 | 0.498 | 0 |
| varying_wide_pilot | aclime_refit_block | known | 0.7 | (1,78) | 0.949 | 0.452 | 0 |
| varying_wide_pilot | oracle | known | 0.35 | (1,1) | 0.931 | 0.682 | 0 |
| varying_wide_pilot | oracle | known | 0.35 | (1,2) | 0.956 | 0.494 | 0 |
| varying_wide_pilot | oracle | known | 0.35 | (1,78) | 0.945 | 0.451 | 0 |
| varying_wide_pilot | oracle | known | 0.7 | (1,1) | 0.931 | 0.682 | 0 |
| varying_wide_pilot | oracle | known | 0.7 | (1,2) | 0.956 | 0.494 | 0 |
| varying_wide_pilot | oracle | known | 0.7 | (1,78) | 0.945 | 0.451 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | (1,1) | 0.678 | 0.804 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | (1,2) | 0.960 | 0.581 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.35 | (1,78) | 0.941 | 0.537 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | (1,1) | 0.940 | 0.712 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | (1,2) | 0.972 | 0.517 | 0 |
| varying_wide_pilot | scaled_refit_block | known | 0.7 | (1,78) | 0.948 | 0.474 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | (1,1) | 0.884 | 0.748 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | (1,2) | 0.962 | 0.544 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.35 | (1,78) | 0.943 | 0.496 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | (1,1) | 0.928 | 0.677 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | (1,2) | 0.950 | 0.496 | 0 |
| varying_wide_pilot | sqrtlasso | known | 0.7 | (1,78) | 0.951 | 0.452 | 0 |

## Data-driven selections

These are observed selection frequencies, not coverage-optimality claims.

| Scenario | Index | Final h | Pilot h | Penalty multiplier | Count | Fraction |
|---|---|---|---|---|---|---|
| cv_generated_index | estimated | 0.842 | 1.200 | 0.700 | 754 | 0.754 |
| cv_generated_index | estimated | 0.842 | 1.200 | 0.400 | 169 | 0.169 |
| cv_generated_index | estimated | 0.631 | 1.200 | 0.700 | 57 | 0.057 |
| cv_generated_index | estimated | 0.631 | 1.200 | 0.400 | 19 | 0.019 |
| cv_generated_index | estimated | 0.421 | 1.200 | 0.400 | 1 | 0.001 |
| cv_generated_index | known | 0.842 | 1.200 | 0.700 | 707 | 0.707 |
| cv_generated_index | known | 0.842 | 1.200 | 0.400 | 198 | 0.198 |
| cv_generated_index | known | 0.631 | 1.200 | 0.700 | 62 | 0.062 |
| cv_generated_index | known | 0.631 | 1.200 | 0.400 | 28 | 0.028 |
| cv_generated_index | known | 0.842 | 0.900 | 0.400 | 2 | 0.002 |
| cv_generated_index | known | 0.631 | 0.900 | 0.700 | 1 | 0.001 |
| cv_generated_index | known | 0.421 | 1.200 | 0.700 | 1 | 0.001 |
| cv_generated_index | known | 0.421 | 1.200 | 0.400 | 1 | 0.001 |
| cv_high_dimensional | known | 0.889 | 1.200 | 0.700 | 818 | 0.818 |
| cv_high_dimensional | known | 0.667 | 1.200 | 0.700 | 99 | 0.099 |
| cv_high_dimensional | known | 0.889 | 1.200 | 0.400 | 18 | 0.018 |
| cv_high_dimensional | known | 0.889 | 1.200 | 1.000 | 13 | 0.013 |
| cv_high_dimensional | known | 0.889 | 0.600 | 1.000 | 12 | 0.012 |
| cv_high_dimensional | known | 0.889 | 0.900 | 0.700 | 10 | 0.010 |
| cv_high_dimensional | known | 0.889 | 0.600 | 0.700 | 9 | 0.009 |
| cv_high_dimensional | known | 0.667 | 1.200 | 0.400 | 7 | 0.007 |

## Matrix errors at zero

These are full-matrix errors, not errors of an assembled set of pairwise intervals. Failed matrices are counted separately.

| Scenario | Method | Nominal factor | Norm | NMC | Mean error | Failures |
|---|---|---|---|---|---|---|
| block_wide_pilot | aclime_onestep | 0.35 | frobenius | 1000 | 3.419 | 0 |
| block_wide_pilot | aclime_onestep | 0.35 | induced_l1 | 1000 | 3.511 | 0 |
| block_wide_pilot | aclime_onestep | 0.35 | max_entry | 1000 | 0.355 | 0 |
| block_wide_pilot | aclime_onestep | 0.35 | spectral | 1000 | 1.066 | 0 |
| block_wide_pilot | aclime_onestep | 0.7 | frobenius | 1000 | 2.878 | 0 |
| block_wide_pilot | aclime_onestep | 0.7 | induced_l1 | 1000 | 2.956 | 0 |
| block_wide_pilot | aclime_onestep | 0.7 | max_entry | 1000 | 0.291 | 0 |
| block_wide_pilot | aclime_onestep | 0.7 | spectral | 1000 | 1.038 | 0 |
| block_wide_pilot | aclime_pilot | 0.35 | frobenius | 1000 | 0.870 | 0 |
| block_wide_pilot | aclime_pilot | 0.35 | induced_l1 | 1000 | 0.665 | 0 |
| block_wide_pilot | aclime_pilot | 0.35 | max_entry | 1000 | 0.211 | 0 |
| block_wide_pilot | aclime_pilot | 0.35 | spectral | 1000 | 0.417 | 0 |
| block_wide_pilot | aclime_pilot | 0.7 | frobenius | 1000 | 1.176 | 0 |
| block_wide_pilot | aclime_pilot | 0.7 | induced_l1 | 1000 | 0.829 | 0 |
| block_wide_pilot | aclime_pilot | 0.7 | max_entry | 1000 | 0.303 | 0 |
| block_wide_pilot | aclime_pilot | 0.7 | spectral | 1000 | 0.716 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.35 | frobenius | 1000 | 4.246 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.35 | induced_l1 | 1000 | 4.477 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.35 | max_entry | 1000 | 0.478 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.35 | spectral | 1000 | 1.491 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.7 | frobenius | 1000 | 3.870 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.7 | induced_l1 | 1000 | 4.006 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.7 | max_entry | 1000 | 0.368 | 0 |
| block_wide_pilot | aclime_refit_onestep | 0.7 | spectral | 1000 | 1.353 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.35 | frobenius | 1000 | 1.882 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.35 | induced_l1 | 1000 | 1.471 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.35 | max_entry | 1000 | 0.294 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.35 | spectral | 1000 | 0.690 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.7 | frobenius | 1000 | 0.752 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.7 | induced_l1 | 1000 | 0.471 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.7 | max_entry | 1000 | 0.229 | 0 |
| block_wide_pilot | aclime_refit_pilot | 0.7 | spectral | 1000 | 0.341 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.35 | frobenius | 1000 | 4.164 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.35 | induced_l1 | 1000 | 4.338 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.35 | max_entry | 1000 | 0.439 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.35 | spectral | 1000 | 1.291 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.7 | frobenius | 1000 | 3.894 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.7 | induced_l1 | 1000 | 4.021 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.7 | max_entry | 1000 | 0.370 | 0 |
| block_wide_pilot | scaledlasso_onestep | 0.7 | spectral | 1000 | 1.335 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.35 | frobenius | 1000 | 1.187 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.35 | induced_l1 | 1000 | 0.924 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.35 | max_entry | 1000 | 0.277 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.35 | spectral | 1000 | 0.452 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.7 | frobenius | 1000 | 0.738 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.7 | induced_l1 | 1000 | 0.572 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.7 | max_entry | 1000 | 0.225 | 0 |
| block_wide_pilot | scaledlasso_pilot | 0.7 | spectral | 1000 | 0.389 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.35 | frobenius | 1000 | 4.484 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.35 | induced_l1 | 1000 | 4.774 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.35 | max_entry | 1000 | 0.530 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.35 | spectral | 1000 | 1.697 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.7 | frobenius | 1000 | 4.014 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.7 | induced_l1 | 1000 | 4.187 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.7 | max_entry | 1000 | 0.412 | 0 |
| block_wide_pilot | scaledlasso_refit_onestep | 0.7 | spectral | 1000 | 1.278 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.35 | frobenius | 1000 | 2.262 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.35 | induced_l1 | 1000 | 1.977 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.35 | max_entry | 1000 | 0.332 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.35 | spectral | 1000 | 0.849 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.7 | frobenius | 1000 | 1.336 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.7 | induced_l1 | 1000 | 0.882 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.7 | max_entry | 1000 | 0.251 | 0 |
| block_wide_pilot | scaledlasso_refit_pilot | 0.7 | spectral | 1000 | 0.504 | 0 |
| varying_wide_pilot | aclime_onestep | 0.35 | frobenius | 1000 | 5.168 | 0 |
| varying_wide_pilot | aclime_onestep | 0.35 | induced_l1 | 1000 | 5.071 | 0 |
| varying_wide_pilot | aclime_onestep | 0.35 | max_entry | 1000 | 0.300 | 0 |
| varying_wide_pilot | aclime_onestep | 0.35 | spectral | 1000 | 1.233 | 0 |
| varying_wide_pilot | aclime_onestep | 0.7 | frobenius | 1000 | 4.500 | 0 |
| varying_wide_pilot | aclime_onestep | 0.7 | induced_l1 | 1000 | 4.433 | 0 |
| varying_wide_pilot | aclime_onestep | 0.7 | max_entry | 1000 | 0.245 | 0 |
| varying_wide_pilot | aclime_onestep | 0.7 | spectral | 1000 | 1.229 | 0 |
| varying_wide_pilot | aclime_pilot | 0.35 | frobenius | 1000 | 0.980 | 0 |
| varying_wide_pilot | aclime_pilot | 0.35 | induced_l1 | 1000 | 0.613 | 0 |
| varying_wide_pilot | aclime_pilot | 0.35 | max_entry | 1000 | 0.174 | 0 |
| varying_wide_pilot | aclime_pilot | 0.35 | spectral | 1000 | 0.355 | 0 |
| varying_wide_pilot | aclime_pilot | 0.7 | frobenius | 1000 | 1.249 | 0 |
| varying_wide_pilot | aclime_pilot | 0.7 | induced_l1 | 1000 | 0.731 | 0 |
| varying_wide_pilot | aclime_pilot | 0.7 | max_entry | 1000 | 0.256 | 0 |
| varying_wide_pilot | aclime_pilot | 0.7 | spectral | 1000 | 0.626 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.35 | frobenius | 1000 | 6.232 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.35 | induced_l1 | 1000 | 6.201 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.35 | max_entry | 1000 | 0.417 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.35 | spectral | 1000 | 1.525 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.7 | frobenius | 1000 | 5.730 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.7 | induced_l1 | 1000 | 5.674 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.7 | max_entry | 1000 | 0.298 | 0 |
| varying_wide_pilot | aclime_refit_onestep | 0.7 | spectral | 1000 | 1.533 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.35 | frobenius | 1000 | 2.556 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.35 | induced_l1 | 1000 | 1.679 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.35 | max_entry | 1000 | 0.254 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.35 | spectral | 1000 | 0.660 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.7 | frobenius | 1000 | 0.782 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.7 | induced_l1 | 1000 | 0.383 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.7 | max_entry | 1000 | 0.176 | 0 |
| varying_wide_pilot | aclime_refit_pilot | 0.7 | spectral | 1000 | 0.272 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.35 | frobenius | 1000 | 6.148 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.35 | induced_l1 | 1000 | 6.092 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.35 | max_entry | 1000 | 0.376 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.35 | spectral | 1000 | 1.334 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.7 | frobenius | 1000 | 5.762 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.7 | induced_l1 | 1000 | 5.701 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.7 | max_entry | 1000 | 0.301 | 0 |
| varying_wide_pilot | scaledlasso_onestep | 0.7 | spectral | 1000 | 1.514 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.35 | frobenius | 1000 | 1.510 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.35 | induced_l1 | 1000 | 0.997 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.35 | max_entry | 1000 | 0.240 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.35 | spectral | 1000 | 0.410 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.7 | frobenius | 1000 | 0.734 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.7 | induced_l1 | 1000 | 0.488 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.7 | max_entry | 1000 | 0.183 | 0 |
| varying_wide_pilot | scaledlasso_pilot | 0.7 | spectral | 1000 | 0.329 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.35 | frobenius | 1000 | 6.652 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.35 | induced_l1 | 1000 | 6.656 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.35 | max_entry | 1000 | 0.481 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.35 | spectral | 1000 | 1.787 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.7 | frobenius | 1000 | 5.887 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.7 | induced_l1 | 1000 | 5.844 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.7 | max_entry | 1000 | 0.338 | 0 |
| varying_wide_pilot | scaledlasso_refit_onestep | 0.7 | spectral | 1000 | 1.388 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.35 | frobenius | 1000 | 3.215 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.35 | induced_l1 | 1000 | 2.448 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.35 | max_entry | 1000 | 0.301 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.35 | spectral | 1000 | 0.854 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.7 | frobenius | 1000 | 1.661 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.7 | induced_l1 | 1000 | 0.871 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.7 | max_entry | 1000 | 0.204 | 0 |
| varying_wide_pilot | scaledlasso_refit_pilot | 0.7 | spectral | 1000 | 0.447 | 0 |
