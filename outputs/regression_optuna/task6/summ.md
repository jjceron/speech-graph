# Optuna Regression: Task 6 (W30-W200)

## Setup

- **Task**: Narrative recall (story retelling, audio-described text)
- **Method**: Optuna hyperparameter optimisation (300 trials) with RFE `fixed` mode (RFE on split 0 only, no leakage)
- **Regressors** (fast mode): Ridge, ElasticNet, QuantileRegressor, KNN, RandomForest, ExtraTrees, DecisionTree
- **Validation**: 70/20/10 Monte Carlo cross-validation (400 iterations)
- **Experiments**: raw (13 features), zscores (9 z-features)
- **Windows**: W30 (raw, zscores)

## Progress

| Experiment | MOT | COG | MOT_V4 | COG_V1 | Status |
|---|---|---|---|---|---|
| **W30_raw_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W30_zscores_fixed** | ✅ | ✅ | ✅ | ✅ | complete |

---

## Results

### W30_raw_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.012 [−0.017, −0.006] | 0.117 [0.097, 0.136] | 54.5% | 6.504 [6.462, 6.546] | QuantileRegressor (α=0.029) | nodes_T6W30, asp_T6W30 |
| **COG** | −0.007 [−0.008, −0.006] | — | 91% | 1.659 [1.641, 1.677] | QuantileRegressor (α=5.183) | lsc_T6W30, atd_T6W30, density_T6W30, diameter_T6W30, asp_T6W30 |
| **MOT_V4** | 0.004 [−0.006, 0.013] | 0.167 [0.147, 0.186] | 39.5% | 2.809 [2.787, 2.831] | QuantileRegressor (α=0.005) | l1_T6W30, atd_T6W30 |
| **COG_V1** | −0.013 [−0.022, −0.005] | 0.080 [0.060, 0.099] | 47.5% | 1.016 [1.008, 1.024] | QuantileRegressor (α=0.0002) | nodes_T6W30, lcc_T6W30 |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 152 | QuantileRegressor (α=0.029) | 2 | 6.340 [6.314, 6.367] | 7.667 | −0.019 [−0.023, −0.015] | 0.092 [0.080, 0.104] | 6.504 [6.462, 6.546] | 7.823 | −0.012 [−0.017, −0.006] | 0.117 [0.097, 0.136] |
| COG | 9 | QuantileRegressor (α=5.183) | 5 | 1.731 [1.719, 1.743] | 2.252 | −0.009 [−0.010, −0.008] | — | 1.659 [1.641, 1.677] | 2.173 | −0.007 [−0.008, −0.006] | — |
| MOT_V4 | 131 | QuantileRegressor (α=0.005) | 2 | 2.794 [2.780, 2.808] | 3.445 | 0.001 [−0.006, 0.007] | 0.163 [0.150, 0.175] | 2.809 [2.787, 2.831] | 3.449 | 0.004 [−0.006, 0.013] | 0.167 [0.147, 0.186] |
| COG_V1 | 252 | QuantileRegressor (α=0.0002) | 2 | 1.054 [1.049, 1.060] | 1.382 | −0.002 [−0.007, 0.003] | 0.097 [0.083, 0.111] | 1.016 [1.008, 1.024] | 1.342 | −0.013 [−0.022, −0.005] | 0.080 [0.060, 0.099] |

---

### W30_zscores_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.042 [−0.053, −0.031] | 0.083 [0.064, 0.103] | 60.5% | 6.587 [6.531, 6.643] | QuantileRegressor (α=0.0001) | z_lsc_T6W30, z_diameter_T6W30 |
| **COG** | −0.007 [−0.008, −0.006] | — | 91% | 1.659 [1.641, 1.677] | QuantileRegressor (α=5.183) | z_lsc_T6W30, z_density_T6W30, z_diameter_T6W30, z_asp_T6W30 |
| **MOT_V4** | −0.013 [−0.014, −0.012] | — | 99.25% | 2.822 [2.805, 2.839] | QuantileRegressor (α=5.183) | z_lsc_T6W30, z_density_T6W30, z_diameter_T6W30, z_asp_T6W30 |
| **COG_V1** | −0.007 [−0.008, −0.006] | — | 87.75% | 1.016 [1.009, 1.024] | QuantileRegressor (α=5.183) | z_lsc_T6W30, z_density_T6W30, z_diameter_T6W30, z_asp_T6W30 |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 270 | QuantileRegressor (α=0.0001) | 2 | 6.448 [6.414, 6.481] | 7.806 | −0.057 [−0.065, −0.049] | 0.068 [0.055, 0.081] | 6.587 [6.531, 6.643] | 7.935 | −0.042 [−0.053, −0.031] | 0.083 [0.064, 0.103] |
| COG | 9 | QuantileRegressor (α=5.183) | 4 | 1.731 [1.719, 1.743] | 2.252 | −0.009 [−0.010, −0.008] | — | 1.659 [1.641, 1.677] | 2.173 | −0.007 [−0.008, −0.006] | — |
| MOT_V4 | 9 | QuantileRegressor (α=5.183) | 4 | 2.821 [2.810, 2.832] | 3.491 | −0.024 [−0.025, −0.024] | — | 2.822 [2.805, 2.839] | 3.482 | −0.013 [−0.014, −0.012] | — |
| COG_V1 | 9 | QuantileRegressor (α=5.183) | 4 | 1.060 [1.054, 1.065] | 1.384 | −0.003 [−0.004, −0.003] | — | 1.016 [1.009, 1.024] | 1.340 | −0.007 [−0.008, −0.006] | — |

---

## Key Findings

- **Task 6 results are markedly weaker than Task 2** across all targets and experiment types.
- Only **MOT_V4 at W30_raw** shows a marginally positive R² (0.004 [−0.006, 0.013], 39.5% failure), driven by QuantileRegressor with 2 features (l1_T6W30, atd_T6W30). This is the sole positive signal in Task 6.
- **COG and COG_V1** are consistently near-zero negative (R²≈−0.007 to −0.013) with high failure rates (47.5–91%). COG and COG_V1 at W30_zscores collapse to near-constant prediction.
- **MOT at W30_raw** (R²=−0.012, 54.5% failure) is worse than all Task 2 MOT configurations. Zscores degrade further (R²=−0.042, 60.5% failure).
- **Raw features outperform zscores** in all cases, consistent with Task 2. However, even raw features fail to produce meaningful positive R² values.
- **Window limitation**: Only W30 is available for Task 6 regression. The narrative recall task may require different window ranges or feature sets to capture predictive signal.
- **Best model**: QuantileRegressor dominates across all targets and experiment types (unlike Task 2 where RandomForest/ExtraTrees captured some signal), suggesting the Task 6 signal is too weak for more complex models to exploit.
