# Optuna Regression: Task 2 (W10-W40)

## Setup

- **Method**: Optuna hyperparameter optimisation (300 trials) with RFE `fixed` mode (RFE on split 0 only, no leakage)
- **Regressors** (fast mode): Ridge, Lasso, ElasticNet, SVR, KNN, RandomForest, ExtraTrees
- **Validation**: 70/20/10 Monte Carlo cross-validation (400 iterations)
- **Experiments**: raw (13 features), zscores (9, after dropping zero-variance z_lcc, z_atd), rawzscore (22 = 13 raw + 9 z)
- **Windows**: W10, W20, W30, W40

## Execution Status

| Experiment | MOT | COG | MOT_V4 | COG_V1 | Status |
|-----------|-----|-----|--------|--------|--------|
| **W10_raw_fixed** | done | done | done | done | complete |
| **W10_zscores_fixed** | done | done | done | done | complete |
| **W10_rawzscore_fixed** | done | done | — | — | partial (2/4) |
| **W20_raw_fixed** | done | done | done | done | complete |
| **W20_zscores_fixed** | — | — | done | done | partial (2/4) |
| **W20_rawzscore_fixed** | — | — | — | — | pending |
| **W30_raw_fixed** | — | — | — | — | pending |
| **W30_zscores_fixed** | — | — | — | — | pending |
| **W30_rawzscore_fixed** | — | — | — | — | pending |
| **W40_raw_fixed** | — | — | — | — | pending |
| **W40_zscores_fixed** | — | — | — | — | pending |
| **W40_rawzscore_fixed** | — | — | — | — | pending |

## Results

### W10_raw_fixed (complete)

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| **COG_V1** | **0.043** | **0.230** | **24%** | QuantileRegressor | nodes, edges, lcc |
| MOT | 0.017 | 0.197 | 43% | ExtraTrees | pe, l3, diameter |
| MOT_V4 | 0.002 | 0.083 | 53% | QuantileRegressor | pe, atd, density, diameter, asp |
| COG | -0.013 | — | 94% | QuantileRegressor | lsc, atd, density, diameter, asp |

| Target | Trial | Regressor | RFE n | MAE_val | RMSE_val |
|--------|-------|-----------|-------|---------|----------|
| COG_V1 | 227 | QuantileRegressor (alpha=0.0006) | 3 | 1.005 | 1.312 |
| MOT | 230 | ExtraTrees (142 trees, depth=19) | 3 | 6.388 | 7.614 |
| MOT_V4 | 259 | QuantileRegressor (alpha=0.0058) | 5 | 2.814 | 3.678 |
| COG | 9 | QuantileRegressor (alpha=5.18) | 5 | 1.743 | 2.199 |

### W10_zscores_fixed (complete)

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| MOT | 0.023 | 0.145 | 33% | ElasticNet (alpha=0.02, l1=0.38) | z_pe, z_l3 |
| COG | -0.013 | — | 94% | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |
| MOT_V4 | -0.031 | — | 100% | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |
| COG_V1 | -0.004 | — | 82% | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Regressor | RFE n | MAE_val | RMSE_val |
|--------|-------|-----------|-------|---------|----------|
| MOT | 235 | ElasticNet (alpha=0.02, l1=0.38) | 2 | 6.455 | 7.625 |
| COG | 9 | QuantileRegressor (alpha=5.18) | 4 | 1.743 | 2.268 |
| MOT_V4 | 9 | QuantileRegressor (alpha=5.18) | 4 | 2.845 | 3.543 |
| COG_V1 | 9 | QuantileRegressor (alpha=5.18) | 4 | 1.023 | 1.348 |

### W10_rawzscore_fixed (partial: MOT, COG)

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| MOT | 0.000 | 0.160 | 46% | KNeighborsRegressor | pe, diameter, asp, z_l3 |
| COG | -0.013 | — | 94% | QuantileRegressor | z_pe, z_l1, z_l2, z_l3, z_lsc |

| Target | Trial | Regressor | RFE n | MAE_val | RMSE_val |
|--------|-------|-----------|-------|---------|----------|
| MOT | 184 | KNeighbors (n=15, weights=uniform, metric=manhattan) | 4 | 6.405 | 7.724 |
| COG | 9 | QuantileRegressor (alpha=5.18) | 5 | 1.743 | 2.268 |

### W20_raw_fixed (complete)

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| **COG_V1** | **0.041** | **0.250** | **20%** | QuantileRegressor | nodes, edges, l2, lcc |
| MOT | 0.004 | 0.163 | 45% | KNeighborsRegressor | nodes, l1, l2, lcc, lsc |
| MOT_V4 | -0.003 | 0.106 | 53% | QuantileRegressor | pe, l2 |
| COG | -0.013 | — | 94% | QuantileRegressor | lsc, atd, density, diameter, asp |

| Target | Trial | Regressor | RFE n | MAE_val | RMSE_val |
|--------|-------|-----------|-------|---------|----------|
| COG_V1 | 297 | QuantileRegressor (alpha=0.001) | 4 | 0.986 | 1.306 |
| MOT | 514 | KNeighbors (n=19, weights=distance) | 5 | 6.422 | 7.664 |
| MOT_V4 | 551 | QuantileRegressor (alpha=0.014) | 2 | 2.772 | 3.441 |
| COG | 9 | QuantileRegressor (alpha=5.18) | 5 | 1.726 | 2.254 |

### W20_zscores_fixed (partial: MOT_V4, COG_V1)

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| MOT_V4 | -0.001 | 0.104 | 50% | QuantileRegressor | z_pe, z_l2 |
| COG_V1 | -0.005 | — | 82% | QuantileRegressor | z_lsc, z_density, z_diameter, z_asp |

| Target | Trial | Regressor | RFE n | MAE_val | RMSE_val |
|--------|-------|-----------|-------|---------|----------|
| MOT_V4 | 272 | QuantileRegressor (alpha=0.004) | 2 | 2.788 | 3.438 |
| COG_V1 | 9 | QuantileRegressor (alpha=5.18) | 4 | 1.005 | 1.332 |

## Key Findings

1. **COG_V1 is the only target with consistent predictive signal**: R2 approx 0.04 at both W10 and W20 for raw features, Spearman rho 0.23–0.25, only 20–24% of splits yield R2 < 0. Selected features: nodes, edges, lcc/l2.

2. **MOT shows weak signal**: R2 ≈ 0.02 at W10 raw (rho ≈ 0.20) but degrades at W20 (R2 ≈ 0.00). High instability: 43–45% of splits have R2 < 0.

3. **Zscores underperform vs raw**: W10 zscores only show signal for MOT (R2 ≈ 0.02, similar to raw). COG, MOT_V4, COG_V1 all yield R2 ≤ 0. W20 zscores show no signal for any target.

4. **Rawzscore (combined) does not improve**: W10 rawzscore results are comparable to raw alone — no benefit from adding both feature sets.

5. **COG and MOT_V4 are noise**: All experiments yield R2 ≈ 0 or negative, with >50% of splits failing.

6. **Overall**, speech-graph features explain at most ~4% of the variance in impulsivity scores. The only consistent signal is COG_V1 with raw features.
