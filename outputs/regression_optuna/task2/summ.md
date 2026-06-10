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
| **W10_zscores_fixed** | done | — | — | — | partial (1/4) |
| **W10_rawzscore_fixed** | — | — | — | — | pending |
| **W20_raw_fixed** | done | — | — | — | partial (1/4) |
| **W20_zscores_fixed** | — | — | — | — | pending |
| **W20_rawzscore_fixed** | — | — | — | — | pending |
| **W30_raw_fixed** | — | — | — | — | pending |
| **W30_zscores_fixed** | — | — | — | — | pending |
| **W30_rawzscore_fixed** | — | — | — | — | pending |
| **W40_raw_fixed** | — | — | — | — | pending |
| **W40_zscores_fixed** | — | — | — | — | pending |
| **W40_rawzscore_fixed** | — | — | — | — | pending |

## Completed: W10_raw_fixed

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| **COG_V1** | **0.043** | **0.230** | **24%** | QuantileRegressor | nodes, edges, lcc |
| MOT | 0.017 | 0.197 | 43% | ExtraTrees | pe, l3, diameter |
| MOT_V4 | 0.002 | 0.083 | 53% | QuantileRegressor | pe, atd, density, diameter, asp |
| COG | -0.013 | — | 94% | QuantileRegressor | lsc, atd, density, diameter, asp |

### Best Model Details

| Target | Trial | Regressor | RFE n | MAE_val | RMSE_val |
|--------|-------|-----------|-------|---------|----------|
| COG_V1 | 227 | QuantileRegressor (alpha=0.0006) | 3 | 1.005 | 1.312 |
| MOT | 230 | ExtraTrees (142 trees, depth=19) | 3 | 6.388 | 7.614 |
| MOT_V4 | 259 | QuantileRegressor (alpha=0.0058) | 5 | 2.814 | 3.678 |
| COG | 9 | QuantileRegressor (alpha=5.18) | 5 | 1.743 | 2.199 |

## Completed: W10_zscores_fixed (partial)

| Target | R2_test | rho_test | % R2 < 0 | Best Model | Selected Features |
|--------|---------|----------|----------|------------|------------------|
| MOT | 0.023 | 0.145 | 33% | ElasticNet (alpha=0.02, l1=0.38) | z_pe, z_l3 |

## Key Findings

1. **COG_V1 is the only target with consistent predictive signal**: R2 approx 0.04, Spearman rho approx 0.23, only 24% of cross-validation splits yield negative R2. The RFE-selected features (nodes, edges, lcc) align with the bivariate correlation finding (edges vs COG_V1, rho approx 0.28).

2. **MOT shows marginal signal** (R2 approx 0.02, rho approx 0.20) but with high instability: 43% of splits have R2 < 0.

3. **COG and MOT_V4 are noise**: R2 approx 0 or negative, with >50% of splits failing.

4. **Zscores do not outperform raw** features for MOT (comparable R2 approx 0.02).

5. **Overall**, speech-graph features explain at most ~4% of the variance in impulsivity scores. The relationship is weak but non-zero for COG_V1.
