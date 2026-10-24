# Coverage and pilot assessment

Nominal level: 95%. Failures count as noncoverage. The studies use independent seed namespaces. Widths and bias are conditional on a valid fit. Wilson intervals describe Monte Carlo uncertainty; they do not cover untested designs.

## Independent fixed-baseline confirmation

Each cell uses 200 replications. The penalty multiplier is 0.7; final and pilot bandwidths are both 2.5*n^(-1/5). The index is known except where explicitly labeled.

| Scenario | Method | Index | NMC | Failures | Family coverage | Wilson 95% CI | Mean width |
|---|---|---|---|---|---|---|---|
| block_n1200 | aclime_block | known | 200 | 0 | 0.785 | [0.723, 0.836] | 0.578 |
| block_n1200 | aclime_onestep | known | 200 | 0 | 0.385 | [0.320, 0.454] | 0.401 |
| block_n1200 | oracle | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.604 |
| block_n1200 | scaled_block | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.598 |
| block_n1200 | scaled_onestep | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.588 |
| block_n1200 | sqrtlasso | known | 200 | 0 | 0.915 | [0.868, 0.946] | 0.603 |
| block_n600 | aclime_block | known | 200 | 0 | 0.775 | [0.712, 0.827] | 0.754 |
| block_n600 | aclime_onestep | known | 200 | 0 | 0.315 | [0.255, 0.382] | 0.473 |
| block_n600 | oracle | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.807 |
| block_n600 | scaled_block | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.798 |
| block_n600 | scaled_onestep | known | 200 | 0 | 0.885 | [0.833, 0.922] | 0.772 |
| block_n600 | sqrtlasso | known | 200 | 0 | 0.930 | [0.886, 0.958] | 0.808 |
| bump_index | oracle | estimated | 200 | 0 | 0.920 | [0.874, 0.950] | 0.822 |
| bump_index | oracle | known | 200 | 0 | 0.945 | [0.904, 0.969] | 0.822 |
| bump_index | sqrtlasso | estimated | 200 | 0 | 0.920 | [0.874, 0.950] | 0.828 |
| bump_index | sqrtlasso | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.824 |
| chain | aclime_block | known | 200 | 0 | 0.885 | [0.833, 0.922] | 0.748 |
| chain | aclime_onestep | known | 200 | 0 | 0.540 | [0.471, 0.608] | 0.473 |
| chain | oracle | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.768 |
| chain | scaled_block | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.778 |
| chain | scaled_onestep | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.749 |
| chain | sqrtlasso | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.784 |
| curvature | aclime_block | known | 200 | 0 | 0.750 | [0.686, 0.805] | 0.755 |
| curvature | aclime_onestep | known | 200 | 0 | 0.225 | [0.173, 0.288] | 0.474 |
| curvature | oracle | known | 200 | 0 | 0.955 | [0.917, 0.976] | 0.815 |
| curvature | scaled_block | known | 200 | 0 | 0.890 | [0.839, 0.926] | 0.802 |
| curvature | scaled_onestep | known | 200 | 0 | 0.880 | [0.828, 0.918] | 0.773 |
| curvature | sqrtlasso | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.812 |
| grid | aclime_block | known | 200 | 0 | 0.935 | [0.892, 0.962] | 0.880 |
| grid | aclime_onestep | known | 200 | 0 | 0.760 | [0.696, 0.814] | 0.597 |
| grid | oracle | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.893 |
| grid | scaled_block | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.903 |
| grid | scaled_onestep | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.874 |
| grid | sqrtlasso | known | 200 | 0 | 0.940 | [0.898, 0.965] | 0.906 |
| high_dimensional | aclime_block | known | 200 | 0 | 0.640 | [0.571, 0.703] | 0.853 |
| high_dimensional | aclime_onestep | known | 200 | 0 | 0.110 | [0.074, 0.161] | 0.474 |
| high_dimensional | oracle | known | 200 | 0 | 0.910 | [0.862, 0.942] | 0.944 |
| high_dimensional | scaled_block | known | 200 | 0 | 0.880 | [0.828, 0.918] | 0.947 |
| high_dimensional | scaled_onestep | known | 200 | 0 | 0.820 | [0.761, 0.867] | 0.906 |
| high_dimensional | sqrtlasso | known | 200 | 0 | 0.885 | [0.833, 0.922] | 0.966 |
| mixture | aclime_block | known | 200 | 0 | 0.800 | [0.739, 0.850] | 1.098 |
| mixture | aclime_onestep | known | 200 | 0 | 0.500 | [0.431, 0.569] | 0.661 |
| mixture | oracle | known | 200 | 0 | 0.790 | [0.728, 0.841] | 1.155 |
| mixture | scaled_block | known | 200 | 0 | 0.890 | [0.839, 0.926] | 1.153 |
| mixture | scaled_onestep | known | 200 | 0 | 0.670 | [0.602, 0.731] | 1.097 |
| mixture | sqrtlasso | known | 200 | 0 | 0.875 | [0.822, 0.914] | 1.170 |
| selected | sqrtlasso | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.883 |
| sparse_index_d100 | oracle | estimated | 200 | 0 | 0.920 | [0.874, 0.950] | 0.807 |
| sparse_index_d100 | oracle | known | 200 | 0 | 0.930 | [0.886, 0.958] | 0.804 |
| sparse_index_d100 | sqrtlasso | estimated | 200 | 0 | 0.900 | [0.851, 0.934] | 0.817 |
| sparse_index_d100 | sqrtlasso | known | 200 | 0 | 0.925 | [0.880, 0.954] | 0.806 |
| t8 | aclime_block | known | 200 | 0 | 0.815 | [0.755, 0.863] | 0.905 |
| t8 | aclime_onestep | known | 200 | 0 | 0.450 | [0.383, 0.519] | 0.563 |
| t8 | oracle | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.957 |
| t8 | scaled_block | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.964 |
| t8 | scaled_onestep | known | 200 | 0 | 0.810 | [0.750, 0.858] | 0.929 |
| t8 | sqrtlasso | known | 200 | 0 | 0.900 | [0.851, 0.934] | 0.979 |
| varying | aclime_block | known | 200 | 0 | 0.770 | [0.707, 0.823] | 0.758 |
| varying | aclime_onestep | known | 200 | 0 | 0.350 | [0.287, 0.418] | 0.475 |
| varying | oracle | known | 200 | 0 | 0.920 | [0.874, 0.950] | 0.804 |
| varying | scaled_block | known | 200 | 0 | 0.880 | [0.828, 0.918] | 0.801 |
| varying | scaled_onestep | known | 200 | 0 | 0.870 | [0.816, 0.910] | 0.772 |
| varying | sqrtlasso | known | 200 | 0 | 0.890 | [0.839, 0.926] | 0.811 |

## Bandwidth and penalty sensitivity

The development results compare settings on the same simulated datasets. They are calibration evidence, not independent validation of a selected constant.

| Scenario | Method | h factor | Pilot factor | Penalty | NMC | Family coverage | Mean width | Failures |
|---|---|---|---|---|---|---|---|---|
| final_bandwidth | aclime_block | 0.65 | 1.0 | 0.7 | 200 | 0.795 | 0.954 | 0 |
| final_bandwidth | aclime_block | 1.0 | 1.0 | 0.7 | 200 | 0.775 | 0.752 | 0 |
| final_bandwidth | aclime_block | 1.4 | 1.0 | 0.7 | 200 | 0.770 | 0.625 | 0 |
| final_bandwidth | aclime_onestep | 0.65 | 1.0 | 0.7 | 200 | 0.415 | 0.593 | 0 |
| final_bandwidth | aclime_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.290 | 0.476 | 0 |
| final_bandwidth | aclime_onestep | 1.4 | 1.0 | 0.7 | 200 | 0.200 | 0.400 | 0 |
| final_bandwidth | oracle | 0.65 | 1.0 | 0.7 | 200 | 0.955 | 1.020 | 0 |
| final_bandwidth | oracle | 1.0 | 1.0 | 0.7 | 200 | 0.950 | 0.805 | 0 |
| final_bandwidth | oracle | 1.4 | 1.0 | 0.7 | 200 | 0.955 | 0.670 | 0 |
| final_bandwidth | scaled_block | 0.65 | 1.0 | 0.7 | 200 | 0.905 | 1.010 | 0 |
| final_bandwidth | scaled_block | 1.0 | 1.0 | 0.7 | 200 | 0.905 | 0.796 | 0 |
| final_bandwidth | scaled_block | 1.4 | 1.0 | 0.7 | 200 | 0.915 | 0.661 | 0 |
| final_bandwidth | scaled_onestep | 0.65 | 1.0 | 0.7 | 200 | 0.870 | 0.963 | 0 |
| final_bandwidth | scaled_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.895 | 0.775 | 0 |
| final_bandwidth | scaled_onestep | 1.4 | 1.0 | 0.7 | 200 | 0.905 | 0.652 | 0 |
| final_bandwidth | sqrtlasso | 0.65 | 1.0 | 0.7 | 200 | 0.910 | 1.022 | 0 |
| final_bandwidth | sqrtlasso | 1.0 | 1.0 | 0.7 | 200 | 0.915 | 0.805 | 0 |
| final_bandwidth | sqrtlasso | 1.4 | 1.0 | 0.7 | 200 | 0.930 | 0.669 | 0 |
| penalty | aclime_block | 1.0 | 1.0 | 0.35 | 200 | 0.955 | 0.848 | 0 |
| penalty | aclime_block | 1.0 | 1.0 | 0.7 | 200 | 0.765 | 0.746 | 0 |
| penalty | aclime_block | 1.0 | 1.0 | 1.2 | 200 | 0.410 | 0.705 | 0 |
| penalty | aclime_onestep | 1.0 | 1.0 | 0.35 | 200 | 0.780 | 0.609 | 0 |
| penalty | aclime_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.250 | 0.471 | 0 |
| penalty | aclime_onestep | 1.0 | 1.0 | 1.2 | 200 | 0.005 | 0.340 | 0 |
| penalty | oracle | 1.0 | 1.0 | 0.35 | 200 | 0.930 | 0.796 | 0 |
| penalty | oracle | 1.0 | 1.0 | 0.7 | 200 | 0.930 | 0.796 | 0 |
| penalty | oracle | 1.0 | 1.0 | 1.2 | 200 | 0.930 | 0.796 | 0 |
| penalty | scaled_block | 1.0 | 1.0 | 0.35 | 200 | 0.950 | 0.900 | 0 |
| penalty | scaled_block | 1.0 | 1.0 | 0.7 | 200 | 0.910 | 0.789 | 0 |
| penalty | scaled_block | 1.0 | 1.0 | 1.2 | 200 | 0.675 | 0.734 | 0 |
| penalty | scaled_onestep | 1.0 | 1.0 | 0.35 | 200 | 0.815 | 0.851 | 0 |
| penalty | scaled_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.905 | 0.766 | 0 |
| penalty | scaled_onestep | 1.0 | 1.0 | 1.2 | 200 | 0.740 | 0.713 | 0 |
| penalty | sqrtlasso | 1.0 | 1.0 | 0.35 | 200 | 0.955 | 0.929 | 0 |
| penalty | sqrtlasso | 1.0 | 1.0 | 0.7 | 200 | 0.925 | 0.799 | 0 |
| penalty | sqrtlasso | 1.0 | 1.0 | 1.2 | 200 | 0.705 | 0.737 | 0 |
| pilot_bandwidth | aclime_block | 1.0 | 0.65 | 0.7 | 200 | 0.655 | 0.733 | 0 |
| pilot_bandwidth | aclime_block | 1.0 | 1.0 | 0.7 | 200 | 0.765 | 0.745 | 0 |
| pilot_bandwidth | aclime_block | 1.0 | 1.6 | 0.7 | 200 | 0.815 | 0.757 | 0 |
| pilot_bandwidth | aclime_onestep | 1.0 | 0.65 | 0.7 | 200 | 0.075 | 0.433 | 0 |
| pilot_bandwidth | aclime_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.265 | 0.472 | 0 |
| pilot_bandwidth | aclime_onestep | 1.0 | 1.6 | 0.7 | 200 | 0.550 | 0.509 | 0 |
| pilot_bandwidth | oracle | 1.0 | 0.65 | 0.7 | 200 | 0.935 | 0.797 | 0 |
| pilot_bandwidth | oracle | 1.0 | 1.0 | 0.7 | 200 | 0.935 | 0.797 | 0 |
| pilot_bandwidth | oracle | 1.0 | 1.6 | 0.7 | 200 | 0.935 | 0.797 | 0 |
| pilot_bandwidth | scaled_block | 1.0 | 0.65 | 0.7 | 200 | 0.870 | 0.786 | 0 |
| pilot_bandwidth | scaled_block | 1.0 | 1.0 | 0.7 | 200 | 0.880 | 0.790 | 0 |
| pilot_bandwidth | scaled_block | 1.0 | 1.6 | 0.7 | 200 | 0.910 | 0.786 | 0 |
| pilot_bandwidth | scaled_onestep | 1.0 | 0.65 | 0.7 | 200 | 0.855 | 0.774 | 0 |
| pilot_bandwidth | scaled_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.875 | 0.765 | 0 |
| pilot_bandwidth | scaled_onestep | 1.0 | 1.6 | 0.7 | 200 | 0.865 | 0.761 | 0 |
| pilot_bandwidth | sqrtlasso | 1.0 | 0.65 | 0.7 | 200 | 0.885 | 0.796 | 0 |
| pilot_bandwidth | sqrtlasso | 1.0 | 1.0 | 0.7 | 200 | 0.915 | 0.798 | 0 |
| pilot_bandwidth | sqrtlasso | 1.0 | 1.6 | 0.7 | 200 | 0.920 | 0.793 | 0 |
| varying_nuisance | aclime_block | 1.0 | 0.65 | 0.7 | 200 | 0.675 | 0.738 | 0 |
| varying_nuisance | aclime_block | 1.0 | 1.0 | 0.7 | 200 | 0.765 | 0.751 | 0 |
| varying_nuisance | aclime_block | 1.0 | 1.6 | 0.7 | 200 | 0.830 | 0.760 | 0 |
| varying_nuisance | aclime_onestep | 1.0 | 0.65 | 0.7 | 200 | 0.155 | 0.433 | 0 |
| varying_nuisance | aclime_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.345 | 0.471 | 0 |
| varying_nuisance | aclime_onestep | 1.0 | 1.6 | 0.7 | 200 | 0.590 | 0.507 | 0 |
| varying_nuisance | oracle | 1.0 | 0.65 | 0.7 | 200 | 0.930 | 0.800 | 0 |
| varying_nuisance | oracle | 1.0 | 1.0 | 0.7 | 200 | 0.930 | 0.800 | 0 |
| varying_nuisance | oracle | 1.0 | 1.6 | 0.7 | 200 | 0.930 | 0.800 | 0 |
| varying_nuisance | scaled_block | 1.0 | 0.65 | 0.7 | 200 | 0.875 | 0.791 | 0 |
| varying_nuisance | scaled_block | 1.0 | 1.0 | 0.7 | 200 | 0.900 | 0.792 | 0 |
| varying_nuisance | scaled_block | 1.0 | 1.6 | 0.7 | 200 | 0.895 | 0.790 | 0 |
| varying_nuisance | scaled_onestep | 1.0 | 0.65 | 0.7 | 200 | 0.870 | 0.780 | 0 |
| varying_nuisance | scaled_onestep | 1.0 | 1.0 | 0.7 | 200 | 0.865 | 0.767 | 0 |
| varying_nuisance | scaled_onestep | 1.0 | 1.6 | 0.7 | 200 | 0.865 | 0.759 | 0 |
| varying_nuisance | sqrtlasso | 1.0 | 0.65 | 0.7 | 200 | 0.890 | 0.803 | 0 |
| varying_nuisance | sqrtlasso | 1.0 | 1.0 | 0.7 | 200 | 0.910 | 0.804 | 0 |
| varying_nuisance | sqrtlasso | 1.0 | 1.6 | 0.7 | 200 | 0.910 | 0.798 | 0 |

## Support-refit refinement

These fresh data evaluate changes motivated by the earlier scout. The fixed bandwidths and penalty were not selected using these outcomes.

| Scenario | Method | Target | NMC | Coverage | Wilson 95% CI | Mean width | Bias | Failures |
|---|---|---|---|---|---|---|---|---|
| refit_baseline | aclime_block | joint family | 300 | 0.777 | [0.726, 0.820] | 0.750 | nan | 0 |
| refit_baseline | aclime_block | (1,1) at 0 | 300 | 0.923 | [0.888, 0.948] | 0.551 | -0.053 | 0 |
| refit_baseline | aclime_block | (1,2) at 0 | 300 | 0.890 | [0.850, 0.921] | 0.404 | -0.075 | 0 |
| refit_baseline | aclime_block | (1,38) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.383 | -0.001 | 0 |
| refit_baseline | aclime_onestep | joint family | 300 | 0.323 | [0.273, 0.378] | 0.474 | nan | 0 |
| refit_baseline | aclime_onestep | (1,1) at 0 | 300 | 0.750 | [0.698, 0.796] | 0.363 | -0.095 | 0 |
| refit_baseline | aclime_onestep | (1,2) at 0 | 300 | 0.883 | [0.842, 0.915] | 0.263 | -0.049 | 0 |
| refit_baseline | aclime_onestep | (1,38) at 0 | 300 | 0.947 | [0.915, 0.967] | 0.255 | -0.001 | 0 |
| refit_baseline | aclime_refit_block | joint family | 300 | 0.943 | [0.911, 0.964] | 0.808 | nan | 0 |
| refit_baseline | aclime_refit_block | (1,1) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.597 | 0.031 | 0 |
| refit_baseline | aclime_refit_block | (1,2) at 0 | 300 | 0.957 | [0.927, 0.975] | 0.434 | 0.002 | 0 |
| refit_baseline | aclime_refit_block | (1,38) at 0 | 300 | 0.957 | [0.927, 0.975] | 0.400 | 0.002 | 0 |
| refit_baseline | aclime_refit_onestep | joint family | 300 | 0.897 | [0.857, 0.926] | 0.767 | nan | 0 |
| refit_baseline | aclime_refit_onestep | (1,1) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.582 | 0.027 | 0 |
| refit_baseline | aclime_refit_onestep | (1,2) at 0 | 300 | 0.957 | [0.927, 0.975] | 0.421 | 0.001 | 0 |
| refit_baseline | aclime_refit_onestep | (1,38) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.387 | 0.001 | 0 |
| refit_baseline | oracle | joint family | 300 | 0.963 | [0.936, 0.979] | 0.803 | nan | 0 |
| refit_baseline | oracle | (1,1) at 0 | 300 | 0.960 | [0.931, 0.977] | 0.590 | 0.022 | 0 |
| refit_baseline | oracle | (1,2) at 0 | 300 | 0.957 | [0.927, 0.975] | 0.429 | 0.009 | 0 |
| refit_baseline | oracle | (1,38) at 0 | 300 | 0.947 | [0.915, 0.967] | 0.396 | 0.001 | 0 |
| refit_baseline | scaled_block | joint family | 300 | 0.933 | [0.899, 0.956] | 0.794 | nan | 0 |
| refit_baseline | scaled_block | (1,1) at 0 | 300 | 0.963 | [0.936, 0.979] | 0.583 | 0.008 | 0 |
| refit_baseline | scaled_block | (1,2) at 0 | 300 | 0.943 | [0.911, 0.964] | 0.425 | -0.037 | 0 |
| refit_baseline | scaled_block | (1,38) at 0 | 300 | 0.957 | [0.927, 0.975] | 0.397 | -0.000 | 0 |
| refit_baseline | scaled_onestep | joint family | 300 | 0.893 | [0.853, 0.923] | 0.770 | nan | 0 |
| refit_baseline | scaled_onestep | (1,1) at 0 | 300 | 0.957 | [0.927, 0.975] | 0.582 | 0.003 | 0 |
| refit_baseline | scaled_onestep | (1,2) at 0 | 300 | 0.940 | [0.907, 0.962] | 0.422 | -0.038 | 0 |
| refit_baseline | scaled_onestep | (1,38) at 0 | 300 | 0.953 | [0.923, 0.972] | 0.390 | -0.000 | 0 |
| refit_baseline | scaled_refit_block | joint family | 300 | 0.950 | [0.919, 0.969] | 0.875 | nan | 0 |
| refit_baseline | scaled_refit_block | (1,1) at 0 | 300 | 0.940 | [0.907, 0.962] | 0.637 | 0.105 | 0 |
| refit_baseline | scaled_refit_block | (1,2) at 0 | 300 | 0.960 | [0.931, 0.977] | 0.462 | 0.008 | 0 |
| refit_baseline | scaled_refit_block | (1,38) at 0 | 300 | 0.963 | [0.936, 0.979] | 0.428 | 0.003 | 0 |
| refit_baseline | scaled_refit_onestep | joint family | 300 | 0.830 | [0.783, 0.868] | 0.821 | nan | 0 |
| refit_baseline | scaled_refit_onestep | (1,1) at 0 | 300 | 0.900 | [0.861, 0.929] | 0.616 | 0.100 | 0 |
| refit_baseline | scaled_refit_onestep | (1,2) at 0 | 300 | 0.953 | [0.923, 0.972] | 0.443 | 0.006 | 0 |
| refit_baseline | scaled_refit_onestep | (1,38) at 0 | 300 | 0.960 | [0.931, 0.977] | 0.406 | 0.003 | 0 |
| refit_baseline | sqrt_refit | joint family | 300 | 0.950 | [0.919, 0.969] | 0.898 | nan | 0 |
| refit_baseline | sqrt_refit | (1,1) at 0 | 300 | 0.920 | [0.884, 0.946] | 0.652 | 0.130 | 0 |
| refit_baseline | sqrt_refit | (1,2) at 0 | 300 | 0.953 | [0.923, 0.972] | 0.474 | 0.012 | 0 |
| refit_baseline | sqrt_refit | (1,38) at 0 | 300 | 0.963 | [0.936, 0.979] | 0.435 | 0.004 | 0 |
| refit_baseline | sqrtlasso | joint family | 300 | 0.940 | [0.907, 0.962] | 0.804 | nan | 0 |
| refit_baseline | sqrtlasso | (1,1) at 0 | 300 | 0.960 | [0.931, 0.977] | 0.591 | 0.024 | 0 |
| refit_baseline | sqrtlasso | (1,2) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.431 | -0.028 | 0 |
| refit_baseline | sqrtlasso | (1,38) at 0 | 300 | 0.953 | [0.923, 0.972] | 0.401 | 0.000 | 0 |
| refit_varying | aclime_block | joint family | 300 | 0.820 | [0.773, 0.859] | 0.759 | nan | 0 |
| refit_varying | aclime_block | (1,1) at 0 | 300 | 0.907 | [0.868, 0.935] | 0.560 | -0.051 | 0 |
| refit_varying | aclime_block | (1,2) at 0 | 300 | 0.847 | [0.802, 0.883] | 0.404 | -0.091 | 0 |
| refit_varying | aclime_block | (1,38) at 0 | 300 | 0.970 | [0.944, 0.984] | 0.385 | -0.001 | 0 |
| refit_varying | aclime_onestep | joint family | 300 | 0.337 | [0.286, 0.392] | 0.477 | nan | 0 |
| refit_varying | aclime_onestep | (1,1) at 0 | 300 | 0.737 | [0.684, 0.783] | 0.365 | -0.095 | 0 |
| refit_varying | aclime_onestep | (1,2) at 0 | 300 | 0.843 | [0.798, 0.880] | 0.259 | -0.058 | 0 |
| refit_varying | aclime_onestep | (1,38) at 0 | 300 | 0.960 | [0.931, 0.977] | 0.254 | -0.001 | 0 |
| refit_varying | aclime_refit_block | joint family | 300 | 0.907 | [0.868, 0.935] | 0.813 | nan | 0 |
| refit_varying | aclime_refit_block | (1,1) at 0 | 300 | 0.947 | [0.915, 0.967] | 0.604 | 0.036 | 0 |
| refit_varying | aclime_refit_block | (1,2) at 0 | 300 | 0.940 | [0.907, 0.962] | 0.433 | -0.015 | 0 |
| refit_varying | aclime_refit_block | (1,38) at 0 | 300 | 0.977 | [0.953, 0.989] | 0.403 | -0.003 | 0 |
| refit_varying | aclime_refit_onestep | joint family | 300 | 0.910 | [0.872, 0.937] | 0.772 | nan | 0 |
| refit_varying | aclime_refit_onestep | (1,1) at 0 | 300 | 0.930 | [0.895, 0.954] | 0.585 | 0.031 | 0 |
| refit_varying | aclime_refit_onestep | (1,2) at 0 | 300 | 0.940 | [0.907, 0.962] | 0.414 | -0.015 | 0 |
| refit_varying | aclime_refit_onestep | (1,38) at 0 | 300 | 0.970 | [0.944, 0.984] | 0.386 | -0.003 | 0 |
| refit_varying | oracle | joint family | 300 | 0.927 | [0.891, 0.951] | 0.808 | nan | 0 |
| refit_varying | oracle | (1,1) at 0 | 300 | 0.967 | [0.940, 0.982] | 0.598 | 0.024 | 0 |
| refit_varying | oracle | (1,2) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.427 | -0.006 | 0 |
| refit_varying | oracle | (1,38) at 0 | 300 | 0.970 | [0.944, 0.984] | 0.398 | -0.002 | 0 |
| refit_varying | scaled_block | joint family | 300 | 0.907 | [0.868, 0.935] | 0.800 | nan | 0 |
| refit_varying | scaled_block | (1,1) at 0 | 300 | 0.940 | [0.907, 0.962] | 0.592 | 0.011 | 0 |
| refit_varying | scaled_block | (1,2) at 0 | 300 | 0.917 | [0.880, 0.943] | 0.424 | -0.054 | 0 |
| refit_varying | scaled_block | (1,38) at 0 | 300 | 0.973 | [0.948, 0.986] | 0.401 | -0.001 | 0 |
| refit_varying | scaled_onestep | joint family | 300 | 0.900 | [0.861, 0.929] | 0.776 | nan | 0 |
| refit_varying | scaled_onestep | (1,1) at 0 | 300 | 0.937 | [0.903, 0.959] | 0.585 | 0.006 | 0 |
| refit_varying | scaled_onestep | (1,2) at 0 | 300 | 0.913 | [0.876, 0.940] | 0.414 | -0.054 | 0 |
| refit_varying | scaled_onestep | (1,38) at 0 | 300 | 0.970 | [0.944, 0.984] | 0.388 | -0.001 | 0 |
| refit_varying | scaled_refit_block | joint family | 300 | 0.940 | [0.907, 0.962] | 0.880 | nan | 0 |
| refit_varying | scaled_refit_block | (1,1) at 0 | 300 | 0.920 | [0.884, 0.946] | 0.645 | 0.113 | 0 |
| refit_varying | scaled_refit_block | (1,2) at 0 | 300 | 0.947 | [0.915, 0.967] | 0.462 | -0.010 | 0 |
| refit_varying | scaled_refit_block | (1,38) at 0 | 300 | 0.967 | [0.940, 0.982] | 0.432 | 0.000 | 0 |
| refit_varying | scaled_refit_onestep | joint family | 300 | 0.820 | [0.773, 0.859] | 0.822 | nan | 0 |
| refit_varying | scaled_refit_onestep | (1,1) at 0 | 300 | 0.877 | [0.835, 0.909] | 0.613 | 0.106 | 0 |
| refit_varying | scaled_refit_onestep | (1,2) at 0 | 300 | 0.937 | [0.903, 0.959] | 0.435 | -0.011 | 0 |
| refit_varying | scaled_refit_onestep | (1,38) at 0 | 300 | 0.963 | [0.936, 0.979] | 0.403 | 0.000 | 0 |
| refit_varying | sqrt_refit | joint family | 300 | 0.930 | [0.895, 0.954] | 0.901 | nan | 0 |
| refit_varying | sqrt_refit | (1,1) at 0 | 300 | 0.917 | [0.880, 0.943] | 0.660 | 0.137 | 0 |
| refit_varying | sqrt_refit | (1,2) at 0 | 300 | 0.950 | [0.919, 0.969] | 0.472 | -0.007 | 0 |
| refit_varying | sqrt_refit | (1,38) at 0 | 300 | 0.963 | [0.936, 0.979] | 0.439 | 0.002 | 0 |
| refit_varying | sqrtlasso | joint family | 300 | 0.913 | [0.876, 0.940] | 0.811 | nan | 0 |
| refit_varying | sqrtlasso | (1,1) at 0 | 300 | 0.937 | [0.903, 0.959] | 0.600 | 0.027 | 0 |
| refit_varying | sqrtlasso | (1,2) at 0 | 300 | 0.933 | [0.899, 0.956] | 0.430 | -0.045 | 0 |
| refit_varying | sqrtlasso | (1,38) at 0 | 300 | 0.973 | [0.948, 0.986] | 0.404 | -0.001 | 0 |

## Data-driven tuning

The initial selector scored different candidates on different validation windows. The final fixed-window implementation corrects that comparison. These are separate seed namespaces; the before/after comparison is not paired. A prediction-based criterion need not optimize coverage error.

| Study | Selector | Target | NMC | Coverage | Wilson 95% CI | Mean width | Failures |
|---|---|---|---|---|---|---|---|
| development | data_driven | joint family | 100 | 0.900 | [0.826, 0.945] | 0.861 | 0 |
| development | data_driven | (1,1) at 0 | 100 | 0.930 | [0.863, 0.966] | 0.638 | 0 |
| development | data_driven | (1,2) at 0 | 100 | 0.970 | [0.915, 0.990] | 0.458 | 0 |
| development | data_driven | (1,38) at 0 | 100 | 0.970 | [0.915, 0.990] | 0.423 | 0 |
| confirmation | selected | joint family | 200 | 0.925 | [0.880, 0.954] | 0.883 | 0 |
| confirmation | selected | (1,1) at 0 | 200 | 0.945 | [0.904, 0.969] | 0.641 | 0 |
| confirmation | selected | (1,2) at 0 | 200 | 0.925 | [0.880, 0.954] | 0.453 | 0 |
| confirmation | selected | (1,38) at 0 | 200 | 0.945 | [0.904, 0.969] | 0.432 | 0 |
| refinement | fixed_window_cv | joint family | 200 | 0.940 | [0.898, 0.965] | 0.746 | 0 |
| refinement | fixed_window_cv | (1,1) at 0 | 200 | 0.945 | [0.904, 0.969] | 0.556 | 0 |
| refinement | fixed_window_cv | (1,2) at 0 | 200 | 0.930 | [0.886, 0.958] | 0.400 | 0 |
| refinement | fixed_window_cv | (1,38) at 0 | 200 | 0.920 | [0.874, 0.950] | 0.376 | 0 |

## Full-matrix errors at zero

The full one-step matrix need not be positive definite. The matrix-norm task differs from inference on a few prespecified entries. These are mean errors, with failed matrices excluded and their counts shown.

| Scenario | Method | Norm | NMC | Mean error | Failures |
|---|---|---|---|---|---|
| block_n600 | aclime_onestep | frobenius | 200 | 2.729 | 0 |
| block_n600 | aclime_onestep | induced_l1 | 200 | 2.810 | 0 |
| block_n600 | aclime_onestep | induced_linf | 200 | 2.810 | 0 |
| block_n600 | aclime_onestep | max_entry | 200 | 0.314 | 0 |
| block_n600 | aclime_onestep | spectral | 200 | 0.973 | 0 |
| block_n600 | aclime_pilot | frobenius | 200 | 1.363 | 0 |
| block_n600 | aclime_pilot | induced_l1 | 200 | 0.923 | 0 |
| block_n600 | aclime_pilot | induced_linf | 200 | 0.923 | 0 |
| block_n600 | aclime_pilot | max_entry | 200 | 0.353 | 0 |
| block_n600 | aclime_pilot | spectral | 200 | 0.803 | 0 |
| block_n600 | scaledlasso_onestep | frobenius | 200 | 3.930 | 0 |
| block_n600 | scaledlasso_onestep | induced_l1 | 200 | 4.122 | 0 |
| block_n600 | scaledlasso_onestep | induced_linf | 200 | 4.122 | 0 |
| block_n600 | scaledlasso_onestep | max_entry | 200 | 0.405 | 0 |
| block_n600 | scaledlasso_onestep | spectral | 200 | 1.301 | 0 |
| block_n600 | scaledlasso_pilot | frobenius | 200 | 0.891 | 0 |
| block_n600 | scaledlasso_pilot | induced_l1 | 200 | 0.660 | 0 |
| block_n600 | scaledlasso_pilot | induced_linf | 200 | 0.660 | 0 |
| block_n600 | scaledlasso_pilot | max_entry | 200 | 0.284 | 0 |
| block_n600 | scaledlasso_pilot | spectral | 200 | 0.443 | 0 |
| high_dimensional | aclime_onestep | frobenius | 200 | 10.954 | 0 |
| high_dimensional | aclime_onestep | induced_l1 | 200 | 10.862 | 0 |
| high_dimensional | aclime_onestep | induced_linf | 200 | 10.862 | 0 |
| high_dimensional | aclime_onestep | max_entry | 200 | 0.419 | 0 |
| high_dimensional | aclime_onestep | spectral | 200 | 2.780 | 0 |
| high_dimensional | aclime_pilot | frobenius | 200 | 3.026 | 0 |
| high_dimensional | aclime_pilot | induced_l1 | 200 | 1.067 | 0 |
| high_dimensional | aclime_pilot | induced_linf | 200 | 1.067 | 0 |
| high_dimensional | aclime_pilot | max_entry | 200 | 0.439 | 0 |
| high_dimensional | aclime_pilot | spectral | 200 | 0.955 | 0 |
| high_dimensional | scaledlasso_onestep | frobenius | 200 | 18.724 | 0 |
| high_dimensional | scaledlasso_onestep | induced_l1 | 200 | 18.976 | 0 |
| high_dimensional | scaledlasso_onestep | induced_linf | 200 | 18.976 | 0 |
| high_dimensional | scaledlasso_onestep | max_entry | 200 | 0.641 | 0 |
| high_dimensional | scaledlasso_onestep | spectral | 200 | 4.433 | 0 |
| high_dimensional | scaledlasso_pilot | frobenius | 200 | 2.076 | 0 |
| high_dimensional | scaledlasso_pilot | induced_l1 | 200 | 0.857 | 0 |
| high_dimensional | scaledlasso_pilot | induced_linf | 200 | 0.857 | 0 |
| high_dimensional | scaledlasso_pilot | max_entry | 200 | 0.452 | 0 |
| high_dimensional | scaledlasso_pilot | spectral | 200 | 0.573 | 0 |
| refit_baseline | aclime_onestep | frobenius | 300 | 2.730 | 0 |
| refit_baseline | aclime_onestep | induced_l1 | 300 | 2.807 | 0 |
| refit_baseline | aclime_onestep | induced_linf | 300 | 2.807 | 0 |
| refit_baseline | aclime_onestep | max_entry | 300 | 0.305 | 0 |
| refit_baseline | aclime_onestep | spectral | 300 | 0.974 | 0 |
| refit_baseline | aclime_pilot | frobenius | 300 | 1.376 | 0 |
| refit_baseline | aclime_pilot | induced_l1 | 300 | 0.928 | 0 |
| refit_baseline | aclime_pilot | induced_linf | 300 | 0.928 | 0 |
| refit_baseline | aclime_pilot | max_entry | 300 | 0.350 | 0 |
| refit_baseline | aclime_pilot | spectral | 300 | 0.807 | 0 |
| refit_baseline | aclime_refit_onestep | frobenius | 300 | 3.893 | 0 |
| refit_baseline | aclime_refit_onestep | induced_l1 | 300 | 4.064 | 0 |
| refit_baseline | aclime_refit_onestep | induced_linf | 300 | 4.064 | 0 |
| refit_baseline | aclime_refit_onestep | max_entry | 300 | 0.391 | 0 |
| refit_baseline | aclime_refit_onestep | spectral | 300 | 1.336 | 0 |
| refit_baseline | aclime_refit_pilot | frobenius | 300 | 0.928 | 0 |
| refit_baseline | aclime_refit_pilot | induced_l1 | 300 | 0.607 | 0 |
| refit_baseline | aclime_refit_pilot | induced_linf | 300 | 0.607 | 0 |
| refit_baseline | aclime_refit_pilot | max_entry | 300 | 0.286 | 0 |
| refit_baseline | aclime_refit_pilot | spectral | 300 | 0.439 | 0 |
| refit_baseline | scaledlasso_onestep | frobenius | 300 | 3.938 | 0 |
| refit_baseline | scaledlasso_onestep | induced_l1 | 300 | 4.115 | 0 |
| refit_baseline | scaledlasso_onestep | induced_linf | 300 | 4.115 | 0 |
| refit_baseline | scaledlasso_onestep | max_entry | 300 | 0.393 | 0 |
| refit_baseline | scaledlasso_onestep | spectral | 300 | 1.307 | 0 |
| refit_baseline | scaledlasso_pilot | frobenius | 300 | 0.890 | 0 |
| refit_baseline | scaledlasso_pilot | induced_l1 | 300 | 0.659 | 0 |
| refit_baseline | scaledlasso_pilot | induced_linf | 300 | 0.659 | 0 |
| refit_baseline | scaledlasso_pilot | max_entry | 300 | 0.276 | 0 |
| refit_baseline | scaledlasso_pilot | spectral | 300 | 0.441 | 0 |
| refit_baseline | scaledlasso_refit_onestep | frobenius | 300 | 4.170 | 0 |
| refit_baseline | scaledlasso_refit_onestep | induced_l1 | 300 | 4.489 | 0 |
| refit_baseline | scaledlasso_refit_onestep | induced_linf | 300 | 4.489 | 0 |
| refit_baseline | scaledlasso_refit_onestep | max_entry | 300 | 0.480 | 0 |
| refit_baseline | scaledlasso_refit_onestep | spectral | 300 | 1.416 | 0 |
| refit_baseline | scaledlasso_refit_pilot | frobenius | 300 | 1.667 | 0 |
| refit_baseline | scaledlasso_refit_pilot | induced_l1 | 300 | 1.131 | 0 |
| refit_baseline | scaledlasso_refit_pilot | induced_linf | 300 | 1.131 | 0 |
| refit_baseline | scaledlasso_refit_pilot | max_entry | 300 | 0.331 | 0 |
| refit_baseline | scaledlasso_refit_pilot | spectral | 300 | 0.649 | 0 |

## Nested-grid multiplier diagnostics

The three nested grids share each dataset and its observation multipliers. Monotonicity of the simulated critical value is checked within replication. Stability is a numerical diagnostic, not a continuum-coverage proof.

| Method | Paired NMC | Failed triples | Critical monotonicity violations | Mean c81-c41 | Coverage81-Coverage41 | Paired MCSE |
|---|---|---|---|---|---|---|
| oracle | 200 | 0 | 0 | 0.003 | 0.000 | 0.000 |
| scaled_refit_block | 200 | 0 | 0 | 0.012 | -0.005 | 0.005 |
| sqrtlasso | 200 | 0 | 0 | 0.003 | -0.005 | 0.005 |
