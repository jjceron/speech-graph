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
| **W20_rawzscore_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
| **W30_raw_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
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

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **MOT** | 0.031 [0.023, 0.039] | 0.170 [0.153, 0.188] | 35% | LinearRegression | z_pe, z_density, z_asp |
| **COG** | -0.013 [-0.015, -0.011] | — | 93.5% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | -0.001 [-0.007, 0.005] | 0.104 [0.085, 0.122] | 50.5% | QuantileRegressor (α=0.0038) | z_pe, z_l2 |
| **COG_V1** | -0.005 [-0.006, -0.004] | — | 82.5% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |

---

### W20_rawzscore_fixed (pending)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| MOT | — | — | — | — | — |
| COG | — | — | — | — | — |
| MOT_V4 | — | — | — | — | — |
| COG_V1 | — | — | — | — | — |

---

### W30_* (pending)

All W30 experiments (raw, zscores, rawzscore) are pending.

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

- At W10, zscores produce the highest R² (0.023) and lowest failure rate (33%), while rawzscore eliminates signal (R²≈0). At W20, raw signal turns negative (R²=−0.021, 55% failure), while zscores produce the strongest MOT signal across all experiments (R²=0.031, 35% failure, LinearRegression), suggesting zscores generalise better across windows and are more robust than raw features for MOT.

### COG

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **W10** | zscores | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W10** | rawzscore | -0.013 [-0.015, -0.012] | — | 94% | 1.718 [1.700, 1.736] | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **W20** | raw | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **W20** | zscores | -0.013 [-0.015, -0.011] | — | 93.5% | 1.701 [1.684, 1.718] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |

- R² is consistently negative (≈ −0.013) across all experiments and windows. ρ is not computable. >90% of splits fail across all configurations. The results are nearly identical between W10 and W20, confirming no predictive signal.

### MOT_V4

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] | 52% | 2.778 [2.760, 2.797] | QuantileRegressor (α=0.0058) | pe, atd, density, diameter, asp |
| **W10** | zscores | -0.031 [-0.032, -0.029] | — | 100% | 2.806 [2.788, 2.823] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W10** | rawzscore | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] | 48% | 2.774 [2.755, 2.793] | QuantileRegressor (α=0.0019) | pe, z_l2 |
| **W20** | raw | -0.003 [-0.013, 0.006] | 0.140 [0.121, 0.160] | 48.75% | 2.876 [2.853, 2.899] | KNeighborsRegressor (k=20) | nodes, re, pe, l1, l2, l3, lsc, asp |
| **W20** | zscores | -0.001 [-0.007, 0.005] | 0.104 [0.085, 0.122] | 50.5% | 2.853 [2.834, 2.872] | QuantileRegressor (α=0.0038) | z_pe, z_l2 |

- R² hovers near zero across all experiments and windows (W10 raw: 0.002, W10 zscores: −0.031, W10 rawzscore: 0.005, W20 raw: −0.003, W20 zscores: −0.001). Zscores improve notably from W10 (−0.031, 100% failure) to W20 (−0.001, 50.5% failure), comparable to rawzscore at W10. Despite model changes, R² remains at chance level across all configurations.

### COG_V1

| Window | Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|---|
| **W10** | raw | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] | 24% | 1.025 [1.016, 1.034] | QuantileRegressor (α=0.0006) | nodes, edges, lcc |
| **W10** | zscores | -0.004 [-0.004, -0.003] | — | 82% | 1.040 [1.032, 1.048] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **W10** | rawzscore | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] | 25% | 1.025 [1.016, 1.033] | QuantileRegressor (α=0.0005) | nodes, edges, lcc |
| **W20** | raw | 0.041 [0.034, 0.048] | 0.250 [0.231, 0.269] | 20.25% | 0.970 [0.962, 0.979] | QuantileRegressor (α=0.0012) | nodes, edges, l2, lcc |
| **W20** | zscores | -0.005 [-0.006, -0.004] | — | 82.5% | 0.991 [0.983, 0.998] | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |

- COG_V1 is the only target with positive R² across raw and rawzscore at W10 (≈0.04) and raw at W20 (R²=0.041). ρ improves from W10 (0.230) to W20 (0.250), and failing splits decrease (24% → 20.25%). MAE_test drops from 1.025 (W10 raw) to 0.970 (W20 raw). The signal is lost with zscores alone (R²≈−0.004 to −0.005, ~82% failure across both windows). Selected features at W10 (nodes, edges, lcc) expand to include l2 at W20, but the core set (nodes, edges, lcc) is conserved.
