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
| **W20_raw_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
| **W20_zscores_fixed** | ⬜ | ⬜ | ⬜ | ⬜ | pending |
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

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **MOT** | 0.017 [0.004, 0.031] | 0.197 [0.178, 0.216] | 43% | ExtraTreesRegressor | pe, l3, diameter |
| **COG** | -0.013 [-0.015, -0.012] | — | 94% | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **MOT_V4** | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] | 52% | QuantileRegressor (α=0.0058) | pe, atd, density, diameter, asp |
| **COG_V1** | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] | 24% | QuantileRegressor (α=0.0006) | nodes, edges, lcc |

| Target | Trial | Best Model | RFE n | MAE_val | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 230 | ExtraTreesRegressor | 3 | 6.388 | 7.614 | 0.028 [0.020, 0.036] | 0.205 [0.194, 0.217] | 6.358 | 7.566 | 0.017 [0.004, 0.031] | 0.197 [0.178, 0.216] |
| COG | 9 | QuantileRegressor (α=5.1826) | 5 | 1.743 | 2.268 | -0.011 [-0.012, -0.009] | — | 1.718 | 2.215 | -0.013 [-0.015, -0.012] | — |
| MOT_V4 | 259 | QuantileRegressor (α=0.0058) | 5 | 2.814 | 3.483 | -0.002 [-0.006, 0.002] | 0.098 [0.084, 0.112] | 2.778 | 3.438 | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] |
| COG_V1 | 227 | QuantileRegressor (α=0.0006) | 3 | 1.005 | 1.312 | 0.046 [0.041, 0.052] | 0.234 [0.222, 0.246] | 1.025 | 1.323 | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] |

---

### W10_zscores_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **MOT** | 0.023 [0.015, 0.032] | 0.145 [0.126, 0.165] | 33% | ElasticNet | z_pe, z_l3 |
| **COG** | -0.013 [-0.015, -0.012] | — | 94% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | -0.031 [-0.032, -0.029] | — | 100% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **COG_V1** | -0.004 [-0.004, -0.003] | — | 82% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 235 | ElasticNet | 2 | 6.455 | 7.625 | 0.027 [0.022, 0.032] | 0.151 [0.139, 0.164] | 6.423 | 7.560 | 0.023 [0.015, 0.032] | 0.145 [0.126, 0.165] |
| COG | 9 | QuantileRegressor (α=5.1826) | 4 | 1.743 | 2.268 | -0.011 [-0.012, -0.009] | — | 1.718 | 2.215 | -0.013 [-0.015, -0.012] | — |
| MOT_V4 | 9 | QuantileRegressor (α=5.1826) | 4 | 2.845 | 3.543 | -0.037 [-0.038, -0.036] | — | 2.806 | 3.495 | -0.031 [-0.032, -0.029] | — |
| COG_V1 | 9 | QuantileRegressor (α=5.1826) | 4 | 1.023 | 1.348 | -0.005 [-0.005, -0.004] | — | 1.040 | 1.357 | -0.004 [-0.004, -0.003] | — |

---

### W10_rawzscore_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **MOT** | 0.000 [-0.010, 0.011] | 0.160 [0.140, 0.179] | 46% | KNeighborsRegressor | pe, diameter, asp, z_l3 |
| **COG** | -0.013 [-0.015, -0.012] | — | 94% | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] | 48% | QuantileRegressor (α=0.0019) | pe, z_l2 |
| **COG_V1** | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] | 25% | QuantileRegressor (α=0.0005) | nodes, edges, lcc |

| Target | Trial | Best Model | RFE n | MAE_val | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 184 | KNeighborsRegressor | 4 | 6.405 | 7.724 | 0.001 [-0.007, 0.008] | 0.161 [0.148, 0.173] | 6.346 | 7.644 | 0.000 [-0.010, 0.011] | 0.160 [0.140, 0.179] |
| COG | 9 | QuantileRegressor (α=5.1826) | 8 | 1.743 | 2.268 | -0.011 [-0.012, -0.009] | — | 1.718 | 2.215 | -0.013 [-0.015, -0.012] | — |
| MOT_V4 | 232 | QuantileRegressor (α=0.0019) | 2 | 2.809 | 3.476 | 0.001 [-0.004, 0.006] | 0.116 [0.102, 0.129] | 2.774 | 3.431 | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] |
| COG_V1 | 137 | QuantileRegressor (α=0.0005) | 3 | 1.005 | 1.313 | 0.045 [0.040, 0.050] | 0.231 [0.219, 0.243] | 1.025 | 1.324 | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] |

---

### W20_raw_fixed (pending)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| MOT | — | — | — | — | — |
| COG | — | — | — | — | — |
| MOT_V4 | — | — | — | — | — |
| COG_V1 | — | — | — | — | — |

---

### W20_zscores_fixed (pending)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| MOT | — | — | — | — | — |
| COG | — | — | — | — | — |
| MOT_V4 | — | — | — | — | — |
| COG_V1 | — | — | — | — | — |

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

| Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **raw** | 0.017 [0.004, 0.031] | 0.197 [0.178, 0.216] | 43% | ExtraTreesRegressor | pe, l3, diameter |
| **zscores** | 0.023 [0.015, 0.032] | 0.145 [0.126, 0.165] | 33% | ElasticNet | z_pe, z_l3 |
| **rawzscore** | 0.000 [-0.010, 0.011] | 0.160 [0.140, 0.179] | 46% | KNeighborsRegressor | pe, diameter, asp, z_l3 |

- Zscores produce the highest R² (0.023) and lowest failure rate (33%). Rawzscore eliminates the signal (R²≈0).

### COG

| Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **raw** | -0.013 [-0.015, -0.012] | — | 94% | QuantileRegressor (α=5.1826) | lsc, atd, density, diameter, asp |
| **zscores** | -0.013 [-0.015, -0.012] | — | 94% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **rawzscore** | -0.013 [-0.015, -0.012] | — | 94% | QuantileRegressor (α=5.1826) | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |

- R² is consistently negative (≈ -0.013) across all experiments. ρ is not computable. >90% of splits fail across all feature sets.

### MOT_V4

| Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **raw** | 0.002 [-0.004, 0.007] | 0.083 [0.063, 0.103] | 52% | QuantileRegressor (α=0.0058) | pe, atd, density, diameter, asp |
| **zscores** | -0.031 [-0.032, -0.029] | — | 100% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **rawzscore** | 0.005 [-0.001, 0.012] | 0.103 [0.083, 0.123] | 48% | QuantileRegressor (α=0.0019) | pe, z_l2 |

- R² hovers near zero across all experiments (raw: 0.002, zscores: -0.031, rawzscore: 0.005). Zscores produce 100% failing splits; raw and rawzscore are near chance level.

### COG_V1

| Experiment | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | Best Model | Selected Features |
|---|---|---|---|---|---|
| **raw** | 0.043 [0.035, 0.050] | 0.230 [0.210, 0.250] | 24% | QuantileRegressor (α=0.0006) | nodes, edges, lcc |
| **zscores** | -0.004 [-0.004, -0.003] | — | 82% | QuantileRegressor (α=5.1826) | z_lsc, z_density, z_diameter, z_asp |
| **rawzscore** | 0.042 [0.035, 0.050] | 0.232 [0.212, 0.252] | 25% | QuantileRegressor (α=0.0005) | nodes, edges, lcc |

- COG_V1 is the only target with positive R² across raw and rawzscore (≈0.04), with ρ≈0.23 and ~24–25% failing splits. The signal is lost with zscores alone (R²≈−0.004, 82% failure). Selected features (nodes, edges, lcc) are identical for raw and rawzscore.
