# Optuna Regression: Task 2 (W10–W40)

## Setup

- **Method**: Optuna hyperparameter optimisation (300 trials) with RFE `fixed` mode (RFE on split 0 only, no leakage)
- **Regressors** (fast mode): Ridge, ElasticNet, QuantileRegressor, KNN, RandomForest, ExtraTrees, DecisionTree
- **Validation**: 70/20/10 Monte Carlo cross-validation (400 iterations)
- **Experiments**: raw (13 features), zscores (9, after dropping zero-variance z_lcc, z_atd), rawzscore (22 = 13 raw + 9 z)
- **Windows**: W10, W20, W30, W40

## Progress

| Experiment | MOT | COG | MOT_V4 | COG_V1 | Status |
|---|---|---|---|---|---|
| **W10_raw_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W10_zscores_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W10_rawzscore_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W20_raw_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W20_zscores_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W20_rawzscore_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W30_raw_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W30_zscores_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
| **W30_rawzscore_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
| **W40_raw_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
| **W40_zscores_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
| **W40_rawzscore_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |

---

## Results

### W10_raw_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | 0.017 [0.004, 0.031] | 0.197 [0.178, 0.216] | 43% | 6.358 [6.304, 6.412] | ExtraTreesRegressor | pe, l3, diameter |
| **COG** | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **MOT_V4** | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] | 52% | 2.778 [2.760, 2.797] | QuantileRegressor (α=0.0058) | pe, atd, density, diameter, asp |
| **COG_V1** | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] | 24% | 1.025 [1.016, 1.034] | QuantileRegressor (α=0.0006) | nodes, edges, lcc |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 230 | ExtraTreesRegressor | 3 | 6.388 [6.352, 6.423] | 7.614 | 0.028 [0.020, 0.036] | 0.205 [0.194, 0.217] | 6.358 [6.304, 6.412] | 7.566 | 0.017 [0.004, 0.031] | 0.197 [0.178, 0.216] |
| COG | 9 | QuantileRegressor (α=5.1826) | 5 | 1.743 [1.732, 1.755] | 2.268 | -0.011 [-0.012, -0.009] | — | 1.718 [1.700, 1.736] | 2.215 | -0.013 [-0.015, -0.012] | — |
| MOT_V4 | 259 | QuantileRegressor (α=0.0058) | 5 | 2.814 [2.802, 2.827] | 3.483 | -0.002 [-0.006, 0.002] | 0.098 [0.084, 0.112] | 2.778 [2.760, 2.797] | 3.438 | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] |
| COG_V1 | 227 | QuantileRegressor (α=0.0006) | 3 | 1.005 [0.999, 1.010] | 1.312 | 0.046 [0.041, 0.052] | 0.234 [0.222, 0.246] | 1.025 [1.016, 1.034] | 1.323 | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] |

---

### W10_zscores_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | 0.023 [0.015, 0.032] | 0.145 [0.126, 0.165] | 33% | 6.423 [6.375, 6.471] | ElasticNet | z_pe, z_l3 |
| **COG** | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | -0.031 [-0.032, -0.029] | — | 100% | 2.806 [2.788, 2.823] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **COG_V1** | -0.004 [-0.004, -0.003] | — | 82% | 1.040 [1.032, 1.048] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 235 | ElasticNet | 2 | 6.455 [6.424, 6.486] | 7.625 | 0.027 [0.022, 0.032] | 0.151 [0.139, 0.164] | 6.423 [6.375, 6.471] | 7.560 | 0.023 [0.015, 0.032] | 0.145 [0.126, 0.165] |
| COG | 9 | QuantileRegressor (α=5.1826) | 4 | 1.743 [1.732, 1.755] | 2.268 | -0.011 [-0.012, -0.009] | — | 1.718 [1.700, 1.736] | 2.215 | -0.013 [-0.015, -0.012] | — |
| MOT_V4 | 9 | QuantileRegressor (α=5.1826) | 4 | 2.845 [2.833, 2.856] | 3.543 | -0.037 [-0.038, -0.036] | — | 2.806 [2.788, 2.823] | 3.495 | -0.031 [-0.032, -0.029] | — |
| COG_V1 | 9 | QuantileRegressor (α=5.1826) | 4 | 1.023 [1.018, 1.028] | 1.348 | -0.005 [-0.005, -0.004] | — | 1.040 [1.032, 1.048] | 1.357 | -0.004 [-0.004, -0.003] | — |

---

### W10_rawzscore_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | 0.000 [-0.010, 0.011] | 0.160 [0.140, 0.179] | 46% | 6.346 [6.290, 6.401] | KNeighborsRegressor | pe, diameter, asp, z_l3 |
| **COG** | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] | 48% | 2.774 [2.755, 2.793] | QuantileRegressor (α=0.0019) | pe, z_l2 |
| **COG_V1** | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] | 25% | 1.025 [1.016, 1.033] | QuantileRegressor (α=0.0005) | nodes, edges, lcc |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 184 | KNeighborsRegressor | 4 | 6.405 [6.367, 6.442] | 7.724 | 0.001 [-0.007, 0.008] | 0.161 [0.148, 0.173] | 6.346 [6.290, 6.401] | 7.644 | 0.000 [-0.010, 0.011] | 0.160 [0.140, 0.179] |
| COG | 9 | QuantileRegressor (α=5.1826) | 8 | 1.743 [1.732, 1.755] | 2.268 | -0.011 [-0.012, -0.009] | — | 1.718 [1.700, 1.736] | 2.215 | -0.013 [-0.015, -0.012] | — |
| MOT_V4 | 232 | QuantileRegressor (α=0.0019) | 2 | 2.809 [2.796, 2.822] | 3.476 | 0.001 [-0.004, 0.006] | 0.116 [0.102, 0.129] | 2.774 [2.755, 2.793] | 3.431 | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] |
| COG_V1 | 137 | QuantileRegressor (α=0.0005) | 3 | 1.005 [0.999, 1.011] | 1.313 | 0.045 [0.040, 0.050] | 0.231 [0.219, 0.243] | 1.025 [1.016, 1.033] | 1.324 | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] |

---

### W20_raw_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | -0.021 [-0.032, -0.011] | 0.117 [0.097, 0.137] | 55% | 6.493 [6.439, 6.548] | KNeighborsRegressor (k=16) | nodes, edges, re, pe, l1, l2, lcc, lsc, atd, density, asp |
| **COG** | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **MOT_V4** | -0.003 [-0.013, 0.006] | 0.140 [0.121, 0.160] | 48.75% | 2.876 [2.853, 2.899] | KNeighborsRegressor (k=20) | nodes, re, pe, l1, l2, l3, lsc, asp |
| **COG_V1** | 0.041 [0.034, 0.048] | 0.250 [0.231, 0.269] | 20.25% | 0.970 [0.962, 0.979] | QuantileRegressor (α=0.0012) | nodes, edges, l2, lcc |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 86 | KNeighborsRegressor (k=16) | 11 | 6.477 [6.443, 6.512] | 7.753 | -0.008 [-0.015, -0.002] | 0.139 [0.127, 0.151] | 6.493 [6.439, 6.548] | 7.763 | -0.021 [-0.032, -0.011] | 0.117 [0.097, 0.137] |
| COG | 9 | QuantileRegressor (α=5.1826) | 5 | 1.726 [1.714, 1.738] | 2.254 | -0.016 [-0.017, -0.015] | — | 1.701 [1.684, 1.718] | 2.191 | -0.013 [-0.015, -0.011] | — |
| MOT_V4 | 187 | KNeighborsRegressor (k=20) | 8 | 2.805 [2.792, 2.818] | 3.451 | -0.003 [-0.009, 0.003] | 0.140 [0.128, 0.152] | 2.876 [2.853, 2.899] | 3.525 | -0.003 [-0.013, 0.006] | 0.140 [0.121, 0.160] |
| COG_V1 | 297 | QuantileRegressor (α=0.0012) | 4 | 0.986 [0.980, 0.992] | 1.306 | 0.035 [0.030, 0.040] | 0.237 [0.224, 0.250] | 0.970 [0.962, 0.979] | 1.298 | 0.041 [0.034, 0.048] | 0.250 [0.231, 0.269] |

---

### W20_zscores_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | 0.031 [0.023, 0.039] | 0.170 [0.153, 0.188] | 35% | 6.400 [6.349, 6.451] | LinearRegression | z_pe, z_density, z_asp |
| **COG** | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | -0.001 [-0.007, 0.005] | 0.104 [0.085, 0.122] | 50.5% | 2.853 [2.834, 2.872] | QuantileRegressor (α=0.00) | z_pe, z_l2 |
| **COG_V1** | -0.005 [-0.006, -0.004] | — | 82.5% | 0.991 [0.983, 0.998] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 1 | LinearRegression | 3 | 6.409 [6.378, 6.440] | 7.585 | 0.035 [0.029, 0.041] | 0.173 [0.160, 0.185] | 6.400 [6.349, 6.451] | 7.538 | 0.031 [0.023, 0.039] | 0.170 [0.153, 0.188] |
| COG | 9 | QuantileRegressor (α=5.1826) | 4 | 1.726 [1.714, 1.738] | 2.254 | -0.016 [-0.017, -0.015] | — | 1.701 [1.684, 1.718] | 2.191 | -0.013 [-0.015, -0.011] | — |
| MOT_V4 | 272 | QuantileRegressor (α=0.00) | 2 | 2.788 [2.777, 2.800] | 3.438 | 0.004 [0.000, 0.008] | 0.104 [0.093, 0.116] | 2.853 [2.834, 2.872] | 3.497 | -0.001 [-0.007, 0.005] | 0.104 [0.085, 0.122] |
| COG_V1 | 9 | QuantileRegressor (α=5.1826) | 4 | 1.013 [1.008, 1.017] | 1.344 | -0.010 [-0.010, -0.009] | — | 0.991 [0.983, 0.998] | 1.321 | -0.005 [-0.006, -0.004] | — |

---

### W20_rawzscore_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | -0.009 [-0.022, 0.004] | 0.165 [0.145, 0.184] | 48.75% | 6.394 [6.336, 6.452] | KNeighborsRegressor (k=10, weights=distance) | edges, pe, l1, l2, lsc, z_re, z_pe, z_l1, z_l2, z_density, z_diameter, z_asp |
| **COG** | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | 0.027 [0.021, 0.033] | 0.133 [0.114, 0.152] | 32% | 2.861 [2.842, 2.880] | ExtraTreesRegressor (n_estimators=430, max_depth=5) | l2, lsc, z_re, z_pe, z_l2 |
| **COG_V1** | 0.042 [0.037, 0.047] | 0.267 [0.248, 0.286] | 18.5% | 0.971 [0.963, 0.979] | QuantileRegressor (α=0.0227) | edges, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 190 | KNeighborsRegressor (k=10, weights=distance) | 12 | 6.352 [6.314, 6.390] | 7.698 | 0.005 [-0.005, 0.014] | 0.184 [0.171, 0.197] | 6.394 [6.336, 6.452] | 7.710 | -0.009 [-0.022, 0.004] | 0.165 [0.145, 0.184] |
| COG | 9 | QuantileRegressor (α=5.1826) | 8 | 1.726 [1.714, 1.738] | 2.254 | -0.016 [-0.017, -0.015] | — | 1.701 [1.684, 1.718] | 2.191 | -0.013 [-0.015, -0.011] | — |
| MOT_V4 | 251 | ExtraTreesRegressor (n_estimators=430, max_depth=5) | 5 | 2.795 [2.784, 2.807] | 3.398 | 0.027 [0.023, 0.031] | 0.136 [0.123, 0.148] | 2.861 [2.842, 2.880] | 3.471 | 0.027 [0.021, 0.033] | 0.133 [0.114, 0.152] |
| COG_V1 | 211 | QuantileRegressor (α=0.0227) | 2 | 0.988 [0.982, 0.994] | 1.305 | 0.037 [0.033, 0.040] | 0.253 [0.241, 0.265] | 0.971 [0.963, 0.979] | 1.297 | 0.042 [0.037, 0.047] | 0.267 [0.248, 0.286] |

---

### W30_raw_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | 0.011 [0.005, 0.017] | 0.125 [0.104, 0.146] | 37.5% | 6.397 [6.354, 6.440] | ElasticNet (α=0.234, l1_ratio=0.755) | l2, asp |
| **COG** | 0.020 [0.014, 0.025] | 0.189 [0.171, 0.207] | 31% | 1.724 [1.705, 1.742] | ExtraTreesRegressor (n_estimators=445, max_depth=18) | nodes, edges, re, l1, l3, lsc |
| **MOT_V4** | -0.016 [-0.017, -0.015] | — | 99.75% | 2.689 [2.671, 2.707] | QuantileRegressor (α=5.183) | lsc, atd, density, diameter, asp |
| **COG_V1** | 0.037 [0.033, 0.041] | 0.279 [0.261, 0.297] | 16.75% | 1.003 [0.994, 1.011] | QuantileRegressor (α=0.0351) | edges, density, diameter, asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 265 | ElasticNet (α=0.234, l1_ratio=0.755) | 2 | 6.337 [6.311, 6.364] | 7.481 | 0.012 [0.008, 0.016] | 0.124 [0.111, 0.137] | 6.397 [6.354, 6.440] | 7.529 | 0.011 [0.005, 0.017] | 0.125 [0.104, 0.146] |
| COG | 206 | ExtraTreesRegressor (n_estimators=445, max_depth=18) | 6 | 1.692 [1.680, 1.704] | 2.195 | 0.030 [0.027, 0.034] | 0.205 [0.194, 0.217] | 1.724 [1.705, 1.742] | 2.191 | 0.020 [0.014, 0.025] | 0.189 [0.171, 0.207] |
| MOT_V4 | 9 | QuantileRegressor (α=5.183) | 5 | 2.779 [2.767, 2.791] | 3.492 | -0.024 [-0.025, -0.023] | — | 2.689 [2.671, 2.707] | 3.363 | -0.016 [-0.017, -0.015] | — |
| COG_V1 | 231 | QuantileRegressor (α=0.0351) | 4 | 1.003 [0.998, 1.009] | 1.316 | 0.041 [0.038, 0.044] | 0.276 [0.264, 0.289] | 1.003 [0.994, 1.011] | 1.318 | 0.037 [0.033, 0.041] | 0.279 [0.261, 0.297] |

---

### W30_zscores_fixed (pending)

All W30 zscores experiments are pending.

---

### W30_rawzscore_fixed (pending)

All W30 rawzscore experiments are pending.

---

### W40_* (pending)

All W40 experiments (raw, zscores, rawzscore) are pending.

---

## Key Findings

### MOT

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | 0.017 [0.004, 0.031] | 0.197 [0.178, 0.216] | 43% | 6.358 [6.304, 6.412] | ExtraTreesRegressor | pe, l3, diameter |
| **W10** | zscores | 0.023 [0.015, 0.032] | 0.145 [0.126, 0.165] | 33% | 6.423 [6.375, 6.471] | ElasticNet | z_pe, z_l3 |
| **W10** | rawzscore | 0.000 [-0.010, 0.011] | 0.160 [0.140, 0.179] | 46% | 6.346 [6.290, 6.401] | KNeighborsRegressor | pe, diameter, asp, z_l3 |
| **W20** | raw | -0.021 [-0.032, -0.011] | 0.117 [0.097, 0.137] | 55% | 6.493 [6.439, 6.548] | KNeighborsRegressor (k=16) | nodes, edges, re, pe, l1, l2, lcc, lsc, atd, density, asp |
| **W20** | zscores | 0.031 [0.023, 0.039] | 0.170 [0.153, 0.188] | 35% | 6.400 [6.349, 6.451] | LinearRegression | z_pe, z_density, z_asp |
| **W20** | rawzscore | -0.009 [-0.022, 0.004] | 0.165 [0.145, 0.184] | 48.75% | 6.394 [6.336, 6.452] | KNeighborsRegressor (k=10, weights=distance) | edges, pe, l1, l2, lsc, z_re, z_pe, z_l1, z_l2, z_density, z_diameter, z_asp |
| **W30** | raw | 0.011 [0.005, 0.017] | 0.125 [0.104, 0.146] | 37.5% | 6.397 [6.354, 6.440] | ElasticNet (α=0.234, l1_ratio=0.755) | l2, asp |

- At W10, zscores produce the highest R² (0.023) and lowest failure rate (33%), while rawzscore eliminates signal (R²≈0). At W20, raw signal turns negative (R²=−0.021, 55% failure), while zscores produce the strongest MOT signal across all experiments (R²=0.031, 35% failure, LinearRegression). W20 rawzscore underperforms both individual experiment types (R²=−0.009). At W30, raw partially recovers (R²=0.011, 37.5% failure) with only 2 features (l2, asp), suggesting a simpler, more robust representation at longer windows.

### COG

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **W10** | zscores | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W10** | rawzscore | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **W20** | raw | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **W20** | zscores | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W20** | rawzscore | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **W30** | raw | **0.020** [0.014, 0.025] | **0.189** [0.171, 0.207] | **31%** | 1.724 [1.705, 1.742] | ExtraTreesRegressor (n_estimators=445, max_depth=18) | nodes, edges, re, l1, l3, lsc |

- R² is consistently negative (≈ −0.013) across W10–W20 experiments. ρ is not computable. >90% of splits fail across all configurations. The results are nearly identical between W10 and W20, confirming no predictive signal at those windows. **However, at W30_raw, COG becomes positive for the first time (R²=0.020 [0.014, 0.025], 31% failure)**, driven by ExtraTreesRegressor with 6 network features (nodes, edges, re, l1, l3, lsc). This suggests that at longer time windows, graph-theoretic features begin to capture COG-relevant variance.

### MOT_V4

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] | 52% | 2.778 [2.760, 2.797] | QuantileRegressor (α=0.0058) | pe, atd, density, diameter, asp |
| **W10** | zscores | -0.031 [-0.032, -0.029] | — | 100% | 2.806 [2.788, 2.823] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W10** | rawzscore | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] | 48% | 2.774 [2.755, 2.793] | QuantileRegressor (α=0.0019) | pe, z_l2 |
| **W20** | raw | -0.003 [-0.013, 0.006] | 0.140 [0.121, 0.160] | 48.75% | 2.876 [2.853, 2.899] | KNeighborsRegressor (k=20) | nodes, re, pe, l1, l2, l3, lsc, asp |
| **W20** | zscores | -0.001 [-0.007, 0.005] | 0.104 [0.085, 0.122] | 50.5% | 2.853 [2.834, 2.872] | QuantileRegressor (α=0.00) | z_pe, z_l2 |
| **W20** | rawzscore | **0.027** [0.021, 0.033] | 0.133 [0.114, 0.152] | **32%** | 2.861 [2.842, 2.880] | ExtraTreesRegressor (n_estimators=430, max_depth=5) | l2, lsc, z_re, z_pe, z_l2 |
| **W30** | raw | -0.016 [-0.017, -0.015] | — | 99.75% | 2.689 [2.671, 2.707] | QuantileRegressor (α=5.183) | lsc, atd, density, diameter, asp |

- R² hovers near zero across most configurations. Zscores improve notably from W10 (−0.031, 100% failure) to W20 (−0.001, 50.5% failure). **W20_rawzscore achieves the best MOT_V4 result overall (R²=0.027, 32% failure, ExtraTrees)** — the first configuration with a clearly positive R². However, W30_raw regresses to constant prediction (−0.016, 99.75% failure), suggesting the effect is window-dependent.

### COG_V1

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] | 24% | 1.025 [1.016, 1.034] | QuantileRegressor (α=0.0006) | nodes, edges, lcc |
| **W10** | zscores | -0.004 [-0.004, -0.003] | — | 82% | 1.040 [1.032, 1.048] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W10** | rawzscore | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] | 25% | 1.025 [1.016, 1.033] | QuantileRegressor (α=0.0005) | nodes, edges, lcc |
| **W20** | raw | 0.041 [0.034, 0.048] | 0.250 [0.231, 0.269] | 20.25% | 0.970 [0.962, 0.979] | QuantileRegressor (α=0.0012) | nodes, edges, l2, lcc |
| **W20** | zscores | -0.005 [-0.006, -0.004] | — | 82.5% | 0.991 [0.983, 0.998] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W20** | rawzscore | 0.042 [0.037, 0.047] | 0.267 [0.248, 0.286] | 18.5% | 0.971 [0.963, 0.979] | QuantileRegressor (α=0.0227) | edges, z_asp |
| **W30** | raw | 0.037 [0.033, 0.041] | **0.279** [0.261, 0.297] | **16.75%** | 1.003 [0.994, 1.011] | QuantileRegressor (α=0.0351) | edges, density, diameter, asp |

- COG_V1 is the only target with consistently positive R² across raw, rawzscore, and now W30 raw. ρ improves from W10 (0.230) to W20 (0.250) to W30 (0.279), and failing splits decrease (24% → 20.25% → 16.75%). W20_rawzscore performs comparably to raw with the simplest feature set (edges, z_asp only). The signal is lost with zscores alone (R²≈−0.004 to −0.005, ~82% failure across both windows).
