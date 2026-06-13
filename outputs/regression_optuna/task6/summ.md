# Optuna Regression: Task 6 (W30-W200)

## Setup

- **Task**: Narrative recall (story retelling, audio-described text)
- **Method**: Optuna hyperparameter optimisation (300 trials) with RFE `fixed` mode (RFE on split 0 only, no leakage)
- **Regressors** (fast mode): LinearRegression, ElasticNet, QuantileRegressor, KNeighbors, RandomForest, ExtraTrees, DecisionTree, XGBRegressor
- **Validation**: 70/20/10 Monte Carlo cross-validation (400 iterations)
- **Experiments**: raw (13 features), zscores (9 z-features), rawzscore (22 features)
- **Windows**: W30, W40, W50…W200 (raw, zscores, rawzscore)

## Progress

| Experiment | MOT | COG | MOT_V4 | COG_V1 | Status |
|---|---|---|---|---|---|---|
| **W30_raw_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W30_zscores_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W30_rawzscore_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W40_raw_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W40_zscores_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W40_rawzscore_fixed** | ✅ | ✅ | ✅ | ✅ | complete |
| **W50_raw_fixed** | ✅ | ❌ | ❌ | ❌ | partial (MOT only) |
| **W50_zscores_fixed** | — | — | — | — | pending |
| **W50_rawzscore_fixed** | — | — | — | — | pending |

---

## Results

### W30_raw_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.012 [−0.017, −0.006] | 0.117 [0.097, 0.136] | 54.5% | 6.504 [6.462, 6.546] | QuantileRegressor | nodes, asp |
| **COG** | −0.007 [−0.009, −0.006] | — | 91.0% | 1.659 [1.641, 1.677] | QuantileRegressor | lsc, atd, density, diameter, asp |
| **MOT_V4** | 0.004 [−0.006, 0.013] | 0.167 [0.147, 0.186] | 39.5% | 2.809 [2.787, 2.831] | QuantileRegressor | l1, atd |
| **COG_V1** | −0.013 [−0.022, −0.005] | 0.080 [0.060, 0.100] | 47.5% | 1.016 [1.008, 1.024] | QuantileRegressor | nodes, lcc |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 152 | QuantileRegressor | 2 | 6.340 [6.314, 6.367] | 7.667 | −0.019 [−0.023, −0.015] | 0.092 [0.080, 0.104] | 6.504 [6.462, 6.546] | 7.823 | −0.012 [−0.017, −0.006] | 0.117 [0.097, 0.136] |
| COG | 9 | QuantileRegressor | 5 | 1.731 [1.719, 1.743] | 2.252 | −0.009 [−0.010, −0.008] | — | 1.659 [1.641, 1.677] | 2.173 | −0.007 [−0.009, −0.006] | — |
| MOT_V4 | 131 | QuantileRegressor | 2 | 2.794 [2.780, 2.808] | 3.445 | 0.001 [−0.006, 0.007] | 0.163 [0.150, 0.175] | 2.809 [2.787, 2.831] | 3.449 | 0.004 [−0.006, 0.013] | 0.167 [0.147, 0.186] |
| COG_V1 | 252 | QuantileRegressor | 2 | 1.054 [1.049, 1.060] | 1.382 | −0.002 [−0.007, 0.003] | 0.097 [0.083, 0.111] | 1.016 [1.008, 1.024] | 1.342 | −0.013 [−0.022, −0.005] | 0.080 [0.060, 0.100] |

---

### W30_zscores_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.042 [−0.053, −0.031] | 0.083 [0.064, 0.103] | 60.5% | 6.587 [6.531, 6.643] | QuantileRegressor | z_lsc, z_diameter |
| **COG** | −0.007 [−0.009, −0.006] | — | 91.0% | 1.659 [1.641, 1.677] | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | −0.013 [−0.014, −0.012] | — | 99.3% | 2.822 [2.805, 2.839] | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |
| **COG_V1** | −0.007 [−0.008, −0.006] | — | 87.7% | 1.016 [1.009, 1.024] | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 270 | QuantileRegressor | 2 | 6.448 [6.414, 6.482] | 7.806 | −0.057 [−0.065, −0.049] | 0.068 [0.055, 0.082] | 6.587 [6.531, 6.643] | 7.935 | −0.042 [−0.053, −0.031] | 0.083 [0.064, 0.103] |
| COG | 9 | QuantileRegressor | 4 | 1.731 [1.719, 1.743] | 2.252 | −0.009 [−0.010, −0.008] | — | 1.659 [1.641, 1.677] | 2.173 | −0.007 [−0.009, −0.006] | — |
| MOT_V4 | 9 | QuantileRegressor | 4 | 2.821 [2.810, 2.832] | 3.491 | −0.024 [−0.025, −0.024] | — | 2.822 [2.805, 2.839] | 3.482 | −0.013 [−0.014, −0.012] | — |
| COG_V1 | 9 | QuantileRegressor | 4 | 1.060 [1.054, 1.065] | 1.384 | −0.003 [−0.004, −0.003] | — | 1.016 [1.009, 1.024] | 1.340 | −0.007 [−0.008, −0.006] | — |

---

### W30_rawzscore_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.012 [−0.017, −0.006] | 0.115 [0.096, 0.135] | 54.5% | 6.504 [6.462, 6.546] | QuantileRegressor | nodes, z_asp |
| **COG** | −0.007 [−0.009, −0.006] | — | 91.0% | 1.659 [1.641, 1.677] | QuantileRegressor | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **MOT_V4** | −0.013 [−0.014, −0.012] | — | 99.3% | 2.822 [2.805, 2.839] | QuantileRegressor | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |
| **COG_V1** | −0.007 [−0.008, −0.006] | — | 87.7% | 1.016 [1.009, 1.024] | QuantileRegressor | z_pe, z_l1, z_l2, z_l3, z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 198 | QuantileRegressor | 2 | 6.340 [6.314, 6.367] | 7.667 | −0.019 [−0.023, −0.015] | 0.090 [0.078, 0.103] | 6.504 [6.462, 6.546] | 7.824 | −0.012 [−0.017, −0.006] | 0.115 [0.096, 0.135] |
| COG | 9 | QuantileRegressor | 8 | 1.731 [1.719, 1.743] | 2.252 | −0.009 [−0.010, −0.008] | — | 1.659 [1.641, 1.677] | 2.173 | −0.007 [−0.009, −0.006] | — |
| MOT_V4 | 9 | QuantileRegressor | 8 | 2.821 [2.810, 2.832] | 3.491 | −0.024 [−0.025, −0.024] | — | 2.822 [2.805, 2.839] | 3.482 | −0.013 [−0.014, −0.012] | — |
| COG_V1 | 9 | QuantileRegressor | 8 | 1.060 [1.054, 1.065] | 1.384 | −0.003 [−0.004, −0.003] | — | 1.016 [1.009, 1.024] | 1.340 | −0.007 [−0.008, −0.006] | — |

---

### W40_raw_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.009 [−0.014, −0.003] | 0.111 [0.092, 0.129] | 52.5% | 6.517 [6.474, 6.561] | QuantileRegressor | lcc, lsc |
| **COG** | **0.037** [0.028, 0.047] | **0.225** [0.209, 0.242] | **31.5%** | 1.661 [1.642, 1.680] | **ExtraTreesRegressor** | re, pe, l2, l3, lcc, lsc |
| **MOT_V4** | −0.013 [−0.014, −0.012] | — | 99.3% | 2.822 [2.805, 2.839] | QuantileRegressor | l3, lcc, lsc, atd, density, diameter, asp |
| **COG_V1** | 0.001 [−0.006, 0.007] | 0.077 [0.058, 0.096] | 41.0% | 1.012 [1.004, 1.020] | QuantileRegressor | nodes, lcc |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 55 | QuantileRegressor | 2 | 6.361 [6.334, 6.388] | 7.665 | −0.018 [−0.022, −0.014] | 0.081 [0.069, 0.093] | 6.517 [6.474, 6.561] | 7.811 | −0.009 [−0.014, −0.003] | 0.111 [0.092, 0.129] |
| COG | 161 | ExtraTreesRegressor | 6 | 1.713 [1.701, 1.726] | 2.183 | 0.050 [0.043, 0.056] | 0.237 [0.225, 0.248] | 1.661 [1.642, 1.680] | 2.118 | 0.037 [0.028, 0.047] | 0.225 [0.209, 0.242] |
| MOT_V4 | 16 | QuantileRegressor | 7 | 2.821 [2.810, 2.832] | 3.491 | −0.024 [−0.025, −0.024] | — | 2.822 [2.805, 2.839] | 3.482 | −0.013 [−0.014, −0.012] | — |
| COG_V1 | 52 | QuantileRegressor | 2 | 1.053 [1.047, 1.059] | 1.376 | 0.008 [0.004, 0.012] | 0.070 [0.057, 0.084] | 1.012 [1.004, 1.020] | 1.334 | 0.001 [−0.006, 0.007] | 0.077 [0.058, 0.096] |

---

### W40_zscores_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.032 [−0.033, −0.030] | — | 100.0% | 6.620 [6.583, 6.656] | QuantileRegressor | z_density, z_diameter, z_asp |
| **COG** | −0.007 [−0.009, −0.006] | — | 91.0% | 1.659 [1.641, 1.677] | QuantileRegressor | z_diameter, z_asp |
| **MOT_V4** | −0.013 [−0.014, −0.012] | — | 99.3% | 2.822 [2.805, 2.839] | QuantileRegressor | z_diameter, z_asp |
| **COG_V1** | −0.007 [−0.008, −0.006] | — | 88.0% | 1.016 [1.009, 1.024] | QuantileRegressor | z_diameter, z_asp |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 13 | QuantileRegressor | 3 | 6.446 [6.424, 6.469] | 7.731 | −0.035 [−0.037, −0.034] | — | 6.620 [6.583, 6.656] | 7.905 | −0.032 [−0.033, −0.030] | — |
| COG | 8 | QuantileRegressor | 2 | 1.731 [1.719, 1.743] | 2.252 | −0.009 [−0.010, −0.008] | — | 1.659 [1.641, 1.677] | 2.173 | −0.007 [−0.009, −0.006] | — |
| MOT_V4 | 8 | QuantileRegressor | 2 | 2.821 [2.810, 2.832] | 3.491 | −0.024 [−0.025, −0.024] | — | 2.822 [2.805, 2.839] | 3.482 | −0.013 [−0.014, −0.012] | — |
| COG_V1 | 8 | QuantileRegressor | 2 | 1.060 [1.054, 1.065] | 1.384 | −0.003 [−0.004, −0.003] | — | 1.016 [1.009, 1.024] | 1.340 | −0.007 [−0.008, −0.006] | — |

---

### W40_rawzscore_fixed (complete)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** | −0.006 [−0.013, 0.000] | 0.117 [0.098, 0.135] | 49.0% | 6.511 [6.467, 6.554] | QuantileRegressor | lcc, z_density |
| **COG** | **0.064** [0.055, 0.072] | **0.252** [0.235, 0.269] | **22.8%** | 1.642 [1.623, 1.660] | **XGBRegressor** | re, pe, l2, l3 |
| **MOT_V4** | 0.007 [−0.000, 0.015] | 0.147 [0.128, 0.167] | 43.5% | 2.807 [2.787, 2.827] | QuantileRegressor | density, z_l1 |
| **COG_V1** | 0.001 [−0.006, 0.007] | 0.075 [0.056, 0.094] | 41.0% | 1.012 [1.004, 1.020] | QuantileRegressor | nodes, lcc |

| Target | Trial | Best Model | RFE n | MAE_val [IC 95%] | RMSE_val | R²_val [IC 95%] | ρ_val [IC 95%] | MAE_test [IC 95%] | RMSE_test | R²_test [IC 95%] | ρ_test [IC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT | 263 | QuantileRegressor | 2 | 6.346 [6.318, 6.374] | 7.649 | −0.014 [−0.018, −0.010] | 0.094 [0.082, 0.106] | 6.511 [6.467, 6.554] | 7.802 | −0.006 [−0.013, 0.000] | 0.117 [0.098, 0.135] |
| COG | 223 | XGBRegressor | 4 | 1.691 [1.679, 1.704] | 2.155 | 0.074 [0.069, 0.080] | 0.264 [0.252, 0.276] | 1.642 [1.623, 1.660] | 2.091 | 0.064 [0.055, 0.072] | 0.252 [0.235, 0.269] |
| MOT_V4 | 179 | QuantileRegressor | 2 | 2.796 [2.783, 2.809] | 3.444 | 0.002 [−0.003, 0.008] | 0.144 [0.131, 0.157] | 2.807 [2.787, 2.827] | 3.444 | 0.007 [−0.000, 0.015] | 0.147 [0.128, 0.167] |
| COG_V1 | 259 | QuantileRegressor | 2 | 1.053 [1.047, 1.059] | 1.376 | 0.008 [0.004, 0.012] | 0.070 [0.056, 0.083] | 1.012 [1.004, 1.020] | 1.334 | 0.001 [−0.006, 0.007] | 0.075 [0.056, 0.094] |

---

### W50_raw_fixed (partial — MOT only)

| Target | R²_test [IC 95%] | ρ_test [IC 95%] | % R²<0 | MAE_test [IC 95%] | Best Model | Selected Features |
|---|---|---|---|---|---|---|
| **MOT** |  |  |  |  | TBD | TBD |
| **COG** | — | — | — | — | — | — |
| **MOT_V4** | — | — | — | — | — | — |
| **COG_V1** | — | — | — | — | — | — |

---

## Key Findings

- **Task 6 results are markedly weaker than Task 2** across all targets and experiment types, but W40 shows improvement over W30.
- **Best result: COG at W40_rawzscore_fixed** achieves R²=0.064 [0.055, 0.072] with **XGBRegressor** (22.8% failure rate) and 4 features (re, pe, l2, l3). This is the strongest signal yet in Task 6.
- **COG at W40_raw_fixed** also shows positive signal: R²=0.037 [0.028, 0.047] with **ExtraTreesRegressor** (31.5% failure).
- **Window effect**: W40 clearly outperforms W30 for COG targets. At W30, COG is near-zero negative across all experiment types; at W40, COG achieves positive R² with both ExtraTrees (+rawzscore).
- **rawzscore experiments** outperform both raw and zscores individually at W40 (COG: R²=0.064 vs 0.037 vs −0.007; MOT: −0.006 vs −0.009 vs −0.032), showing feature complementarity.
- **MOT remains challenging**: best result is W40_rawzscore (R²=−0.006 [−0.013, 0.000], 49% failure), marginally better than W30_raw.
- **zscores alone continue to fail**: W30 and W40 zscores are consistently near-zero negative for all targets except MOT (weak ρ).
- **Best models**: QuantileRegressor dominates most configurations, but **XGBRegressor** (COG at W40_rawzscore) and **ExtraTreesRegressor** (COG at W40_raw) show that tree-based models can capture signal at larger windows with combined features.
