# 1. SpeechGraph

## 1.1 Overview

SpeechGraph is a speech-analysis project that represents participant answers as directed lexical graphs and uses graph topology to study two metadata targets, MOT and COG. The repository contains source code for transcript parsing, graph construction, raw metric extraction, permutation-baseline z-score computation, correlation analysis, Monte Carlo regression, RidgeCV regression, and model-level permutation tests.

The current evidence is stronger for COG than for MOT. COG is best supported by a raw multi-activity pattern that combines Activity 2, Activity 6, and Activity 7 graph information. MOT has weaker and more heterogeneous raw evidence, but Activity 2 z-score metrics show a coherent normalized pattern.

## 1.2 Main Goals

The project has three main goals: quantify speech organization from transcript-derived directed graphs, compare raw graph properties with permutation-normalized z-score properties, and evaluate whether graph features contain predictive signal for MOT and COG beyond simple descriptive association.

## 1.3 Available Repository Content

The inspected archive contains the source tree and output folders, but it does not contain the raw transcripts, metadata workbook, processed metric tables, or a standalone permutation-test script. The available outputs are sufficient to review correlations, OLS/RidgeCV summaries, coefficients, predictions, figures, and permutation-test summaries.

| Path | Content |
|---|---|
| `src/preprocessing/` | Transcript loading, annotation normalization, and tokenization. |
| `src/graphs/` | Directed graph construction, sliding windows, and graph metrics. |
| `src/pipeline/` | Raw SpeechGraph pipeline and z-score SpeechGraph pipeline. |
| `src/analysis/` | Correlation analysis, OLS Monte Carlo regression, RidgeCV Monte Carlo regression, and random graph utilities. |
| `src/visualization/` | Heatmaps and regression diagnostic figures. |
| `outputs/correlations/` | Simple Spearman and partial Spearman results for raw metrics in Activities 2, 6, and 7, and z-score metrics for Activity 2. |
| `outputs/linear_regression/ols_mc/` | Monte Carlo OLS summaries, coefficients, predictions, and figures. |
| `outputs/linear_regression/ridgecv_mc/` | Monte Carlo RidgeCV summaries, alpha selections, coefficients, predictions, and figures. |
| `outputs/linear_regression/perm_test/` | Model-level permutation tests for one COG model and one MOT model. |

## 1.4 Availability of Raw and Z-Score Results

Raw metrics are complete in the available correlation outputs for Activities 2, 6, and 7. Z-score correlation outputs are available only for Activity 2 and only for the windows retained in the z-score correlation files. Z-score outputs for Activities 6 and 7 are pending, so the current z-score interpretation is incomplete.

| Metric type   | Activity   | Available windows in correlation outputs   | Status                                                    |
|:--------------|:-----------|:-------------------------------------------|:----------------------------------------------------------|
| raw           | Activity 2 | T2W10, T2W20, T2W30, T2W40                 | complete in available outputs                             |
| raw           | Activity 6 | T6W30, T6W40, T6W50, T6W150, T6W200        | complete in available outputs                             |
| raw           | Activity 7 | T7W20, T7W30, T7W40, T7W50                 | complete in available outputs                             |
| z-score       | Activity 2 | T2W10, T2W20                               | available; W30/W40 are not present after output filtering |
| z-score       | Activity 6 | none found                                 | pending                                                   |
| z-score       | Activity 7 | none found                                 | pending                                                   |

# 2. Pipeline

## 2.1 Transcript Loading and Speaker Filtering

The pipeline reads transcript `.txt` files, extracts activity blocks such as `Actividad2`, `Actividad6`, and `Actividad7`, and keeps only selected speakers. The default speaker filter is `spk_1`. Subject codes are recovered from transcript filenames and matched to metadata through the `Cod` column.

## 2.2 Annotation Normalization and Tokenization

Annotations are canonicalized before graph construction. The preserved graph token is `[[EE]]`. Break annotations such as `[[PAUSA]]`, `[[DI]]`, and `[[DP]]` define segment boundaries. Dropped annotations include labels such as `[[IF]]`, `[[PS]]`, and `[[PNC]]`. Text is lowercased by default, structural transcript tags are removed, and words are extracted with Unicode-aware tokenization.

## 2.3 Sliding Window Extraction

For each activity, the transcript is flattened into a token sequence while preserving segment-boundary flags. Sliding windows are extracted with step size 1. Raw outputs are aggregated by subject as mean metric values per file and per window. The configured windows are Activity 2: 10, 20, 30, and 40 tokens; Activity 6: 30, 40, 50, 150, and 200 tokens; and Activity 7: 20, 30, 40, and 50 tokens.

## 2.4 Directed Speech Graph Construction

Each speech window is represented as a directed graph $G=(V,E)$. Tokens are graph nodes. Directed edges are consecutive-token transitions within the same segment. Segment boundaries prevent edges from being drawn across pauses or discontinuities. Multiple occurrences of the same transition are retained as edge counts before metric computation.

## 2.5 Raw Graph Metrics

| Metric   | Technical meaning                                                             |
|:---------|:------------------------------------------------------------------------------|
| wc       | Word count in the analyzed window.                                            |
| nodes    | Number of unique tokens.                                                      |
| edges    | Total number of directed token transitions.                                   |
| re       | Repeated directed transitions beyond the first occurrence.                    |
| pe       | Reciprocal recurrence estimated from two-step closed walks.                   |
| l1       | Self-loops.                                                                   |
| l2       | Parallel reciprocal directed edges between two nodes.                         |
| l3       | Directed three-node cycles.                                                   |
| lcc      | Largest weakly connected component size.                                      |
| lsc      | Largest strongly connected component size.                                    |
| atd      | Average total degree, computed as $2|E|/|V|$.                                 |
| density  | Undirected density after removing self-loops.                                 |
| diameter | Diameter of the largest undirected connected component.                       |
| asp      | Average shortest path in the largest undirected connected component.          |
| cc       | Average clustering coefficient in the largest undirected connected component. |

## 2.6 Z-Score Graph Metrics

Z-score metrics compare the observed graph metric in a window against random graphs generated by shuffling tokens within the same segment boundaries. This preserves segment length and token inventory while disrupting local word order. The z-score for metric $m$ is:

$$
z_m = \frac{m_\mathrm{obs} - \mu(m_\mathrm{perm})}{\sigma(m_\mathrm{perm})}
$$

The default z-score pipeline uses 100 random shuffles per window and seed 42. Metrics such as `wc`, `nodes`, and `edges` can become invariant or near-invariant under within-segment permutation, which explains why some z-score fields are absent or uninformative in the available correlation outputs.

# 3. Statistical Analysis

## 3.1 Simple Spearman Correlation

Simple Spearman correlation evaluates monotonic association between each graph feature and each target without additional adjustment. The analysis reports $\rho$, p-value, sample size, target, activity, window, and feature type.

## 3.2 Partial Spearman Correlation

Partial Spearman correlation estimates the association between a graph feature and the target after adjusting for one covariate. The available partial analyses adjust separately for School year, Age, and Educational level. Non-numeric adjustment variables are encoded ordinally by sorted category labels in the current code.

## 3.3 Linear Regression and RidgeCV

The OLS models use Monte Carlo train-test splits with 400 iterations and test size 0.2. RidgeCV uses the same Monte Carlo framework and chooses $\alpha$ inside each split from a 20-value grid spanning $10^{-2}$ to $10^3$. The linear prediction model is:

$$
\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j
$$

RidgeCV minimizes:

$$
\sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j \beta_j^2
$$

The current RidgeCV implementation calls `RidgeCV` directly. It does not wrap predictor standardization in a `StandardScaler` pipeline, so coefficient magnitudes must be interpreted cautiously when predictors have different scales.

## 3.4 Permutation Tests

The available permutation tests validate complete fitted models by comparing a real model-level $R^2$ with a null distribution obtained after target permutation. These tests validate the full model specification. When a covariate is included, the test does not isolate the incremental contribution of graph metrics above that covariate.

# 4. Correlation Results

## 4.1 Evidence Status Guide

Correlation-level evidence is treated as exploratory because many features, windows, targets, and adjustment variables are screened. Results marked as stable exploratory have p-values below 0.05 for both simple and partial Spearman in the same direction, but they are not corrected for multiple comparisons. Model-level permutation tests provide stronger evidence of non-random full-model signal, while still showing low predictive power.

## 4.2 Strongest Correlation Results by Target, Metric Type, and Adjustment

| Target   | Metric type   | Correlation      | Adjustment        | Activity   | Window   | Feature   |   n |    rho |   p-value | Direction   | Evidence                 |
|:---------|:--------------|:-----------------|:------------------|:-----------|:---------|:----------|----:|-------:|----------:|:------------|:-------------------------|
| MOT      | raw           | Simple Spearman  | NA                | Activity 7 | T7W30    | lsc       | 250 |  0.174 |    0.006  | positive    | exploratory, uncorrected |
| MOT      | raw           | Partial Spearman | School year       | Activity 7 | T7W30    | lsc       | 250 |  0.173 |    0.006  | positive    | stable exploratory       |
| MOT      | raw           | Partial Spearman | Age               | Activity 7 | T7W30    | lsc       | 250 |  0.173 |    0.006  | positive    | stable exploratory       |
| MOT      | raw           | Partial Spearman | Educational level | Activity 7 | T7W30    | lsc       | 250 |  0.174 |    0.006  | positive    | stable exploratory       |
| MOT      | z             | Simple Spearman  | NA                | Activity 2 | T2W20    | z_cc      | 237 | -0.163 |    0.012  | negative    | exploratory, uncorrected |
| MOT      | z             | Partial Spearman | School year       | Activity 2 | T2W20    | z_cc      | 237 | -0.167 |    0.01   | negative    | stable exploratory       |
| MOT      | z             | Partial Spearman | Age               | Activity 2 | T2W20    | z_cc      | 237 | -0.168 |    0.01   | negative    | stable exploratory       |
| MOT      | z             | Partial Spearman | Educational level | Activity 2 | T2W20    | z_cc      | 237 | -0.164 |    0.012  | negative    | stable exploratory       |
| COG      | raw           | Simple Spearman  | NA                | Activity 6 | T6W30    | l3        | 252 | -0.208 |    0.0009 | negative    | exploratory, uncorrected |
| COG      | raw           | Partial Spearman | School year       | Activity 2 | T2W20    | edges     | 237 |  0.18  |    0.005  | positive    | stable exploratory       |
| COG      | raw           | Partial Spearman | Age               | Activity 2 | T2W10    | edges     | 240 |  0.186 |    0.004  | positive    | stable exploratory       |
| COG      | raw           | Partial Spearman | Educational level | Activity 6 | T6W30    | l3        | 252 | -0.199 |    0.001  | negative    | stable exploratory       |
| COG      | z             | Simple Spearman  | NA                | Activity 2 | T2W10    | z_atd     | 240 | -0.135 |    0.036  | negative    | exploratory, uncorrected |
| COG      | z             | Partial Spearman | School year       | Activity 2 | T2W10    | z_l3      | 240 | -0.143 |    0.027  | negative    | exploratory              |
| COG      | z             | Partial Spearman | Age               | Activity 2 | T2W10    | z_l3      | 240 | -0.14  |    0.03   | negative    | exploratory              |
| COG      | z             | Partial Spearman | Educational level | Activity 2 | T2W10    | z_atd     | 240 | -0.124 |    0.055  | negative    | exploratory              |

## 4.3 Simple Spearman Results

### 4.3.1 MOT raw Simple Spearman

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                     |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:-------------------------------------------------------------------------|
| Activity 7 | T7W30    | lsc       | 250 |        0.174 |            0.006 |         0.173 |             0.006 | School year  | positive    | stable exploratory | Higher higher raw strongly connected span is associated with higher MOT. |
| Activity 7 | T7W40    | cc        | 248 |        0.145 |            0.022 |         0.151 |             0.018 | School year  | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W30    | cc        | 250 |        0.143 |            0.024 |         0.147 |             0.02  | School year  | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 2 | T2W30    | l2        | 230 |        0.138 |            0.037 |         0.145 |             0.028 | School year  | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W50    | edges     | 243 |        0.136 |            0.034 |         0.138 |             0.031 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 7 | T7W50    | cc        | 243 |        0.136 |            0.034 |         0.142 |             0.028 | School year  | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W40    | edges     | 248 |        0.135 |            0.034 |         0.137 |             0.032 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 2 | T2W30    | pe        | 230 |        0.132 |            0.045 |         0.138 |             0.037 | School year  | positive    | stable exploratory | Higher higher raw reciprocal recurrence is associated with higher MOT.   |
| Activity 2 | T2W10    | l2        | 240 |        0.13  |            0.044 |         0.136 |             0.036 | School year  | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W30    | edges     | 250 |        0.129 |            0.042 |         0.131 |             0.039 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |

### 4.3.2 MOT z Simple Spearman

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                                         |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:---------------------------------------------------------------------------------------------|
| Activity 2 | T2W20    | z_cc      | 237 |       -0.163 |            0.012 |        -0.167 |             0.01  | School year  | negative    | stable exploratory | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W10    | z_l2      | 240 |        0.154 |            0.017 |         0.154 |             0.017 | School year  | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W10    | z_pe      | 240 |        0.15  |            0.02  |         0.151 |             0.02  | School year  | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W20    | z_l3      | 237 |       -0.146 |            0.024 |        -0.15  |             0.021 | School year  | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W20    | z_l2      | 237 |        0.136 |            0.036 |         0.136 |             0.036 | School year  | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W20    | z_pe      | 237 |        0.131 |            0.044 |         0.131 |             0.044 | School year  | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W10    | z_l3      | 240 |       -0.128 |            0.047 |        -0.132 |             0.042 | School year  | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W10    | z_cc      | 240 |       -0.124 |            0.055 |        -0.127 |             0.049 | School year  | negative    | exploratory        | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W20    | z_density | 237 |       -0.098 |            0.133 |        -0.104 |             0.111 | School year  | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |
| Activity 2 | T2W10    | z_density | 240 |       -0.075 |            0.25  |        -0.083 |             0.203 | School year  | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |

### 4.3.3 COG raw Simple Spearman

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                 |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:---------------------------------------------------------------------|
| Activity 6 | T6W30    | l3        | 252 |       -0.208 |           0.0009 |        -0.157 |             0.013 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.    |
| Activity 6 | T6W50    | l3        | 252 |       -0.208 |           0.0009 |        -0.151 |             0.017 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.    |
| Activity 6 | T6W40    | l3        | 252 |       -0.205 |           0.001  |        -0.15  |             0.018 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.    |
| Activity 2 | T2W10    | edges     | 240 |        0.202 |           0.002  |         0.18  |             0.005 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG.  |
| Activity 2 | T2W40    | edges     | 222 |        0.2   |           0.003  |         0.178 |             0.008 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG.  |
| Activity 2 | T2W20    | edges     | 237 |        0.2   |           0.002  |         0.18  |             0.005 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG.  |
| Activity 6 | T6W200   | l3        | 248 |       -0.196 |           0.002  |        -0.132 |             0.038 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.    |
| Activity 2 | T2W30    | edges     | 230 |        0.194 |           0.003  |         0.173 |             0.009 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG.  |
| Activity 6 | T6W150   | l3        | 252 |       -0.194 |           0.002  |        -0.129 |             0.04  | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.    |
| Activity 6 | T6W50    | re        | 252 |       -0.174 |           0.006  |        -0.1   |             0.113 | School year  | negative    | exploratory        | Higher higher raw repeated transitions is associated with lower COG. |

### 4.3.4 COG z Simple Spearman

| Activity   | Window   | Feature    |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence         | Brief interpretation                                                                           |
|:-----------|:---------|:-----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-----------------|:-----------------------------------------------------------------------------------------------|
| Activity 2 | T2W10    | z_atd      | 240 |       -0.135 |            0.036 |        -0.088 |             0.177 | School year  | negative    | exploratory      | Higher higher than permutation-baseline average total degree is associated with lower COG.     |
| Activity 2 | T2W10    | z_l3       | 240 |       -0.117 |            0.071 |        -0.143 |             0.027 | School year  | negative    | exploratory      | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W10    | z_asp      | 240 |       -0.112 |            0.084 |        -0.064 |             0.327 | School year  | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W10    | z_cc       | 240 |       -0.106 |            0.1   |        -0.131 |             0.043 | School year  | negative    | exploratory      | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_cc       | 237 |       -0.103 |            0.114 |        -0.131 |             0.044 | School year  | negative    | exploratory      | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_diameter | 237 |       -0.101 |            0.12  |        -0.08  |             0.222 | School year  | negative    | weak exploratory | Higher higher than permutation-baseline maximum path length is associated with lower COG.      |
| Activity 2 | T2W20    | z_asp      | 237 |       -0.1   |            0.125 |        -0.073 |             0.264 | School year  | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_l3       | 237 |       -0.099 |            0.127 |        -0.129 |             0.047 | School year  | negative    | exploratory      | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W20    | z_lsc      | 237 |        0.083 |            0.205 |         0.067 |             0.302 | School year  | positive    | weak exploratory | Higher higher than permutation-baseline strongly connected span is associated with higher COG. |
| Activity 2 | T2W10    | z_lsc      | 240 |        0.074 |            0.252 |         0.042 |             0.519 | School year  | positive    | weak exploratory | Higher higher than permutation-baseline strongly connected span is associated with higher COG. |

## 4.4 Partial Spearman Results Adjusted for School Year

### 4.4.1 MOT raw Partial Spearman Adjusted for School Year

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                     |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:-------------------------------------------------------------------------|
| Activity 7 | T7W30    | lsc       | 250 |        0.174 |            0.006 |         0.173 |             0.006 | School year  | positive    | stable exploratory | Higher higher raw strongly connected span is associated with higher MOT. |
| Activity 7 | T7W40    | cc        | 248 |        0.145 |            0.022 |         0.151 |             0.018 | School year  | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W30    | cc        | 250 |        0.143 |            0.024 |         0.147 |             0.02  | School year  | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 2 | T2W30    | l2        | 230 |        0.138 |            0.037 |         0.145 |             0.028 | School year  | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W50    | cc        | 243 |        0.136 |            0.034 |         0.142 |             0.028 | School year  | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W50    | edges     | 243 |        0.136 |            0.034 |         0.138 |             0.031 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 2 | T2W30    | pe        | 230 |        0.132 |            0.045 |         0.138 |             0.037 | School year  | positive    | stable exploratory | Higher higher raw reciprocal recurrence is associated with higher MOT.   |
| Activity 7 | T7W40    | edges     | 248 |        0.135 |            0.034 |         0.137 |             0.032 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 2 | T2W10    | l2        | 240 |        0.13  |            0.044 |         0.136 |             0.036 | School year  | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 6 | T6W30    | nodes     | 252 |       -0.11  |            0.081 |        -0.134 |             0.035 | School year  | negative    | exploratory        | Higher higher raw lexical diversity is associated with lower MOT.        |

### 4.4.2 MOT z Partial Spearman Adjusted for School Year

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                                         |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:---------------------------------------------------------------------------------------------|
| Activity 2 | T2W20    | z_cc      | 237 |       -0.163 |            0.012 |        -0.167 |             0.01  | School year  | negative    | stable exploratory | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W10    | z_l2      | 240 |        0.154 |            0.017 |         0.154 |             0.017 | School year  | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W10    | z_pe      | 240 |        0.15  |            0.02  |         0.151 |             0.02  | School year  | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W20    | z_l3      | 237 |       -0.146 |            0.024 |        -0.15  |             0.021 | School year  | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W20    | z_l2      | 237 |        0.136 |            0.036 |         0.136 |             0.036 | School year  | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W10    | z_l3      | 240 |       -0.128 |            0.047 |        -0.132 |             0.042 | School year  | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W20    | z_pe      | 237 |        0.131 |            0.044 |         0.131 |             0.044 | School year  | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W10    | z_cc      | 240 |       -0.124 |            0.055 |        -0.127 |             0.049 | School year  | negative    | exploratory        | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W20    | z_density | 237 |       -0.098 |            0.133 |        -0.104 |             0.111 | School year  | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |
| Activity 2 | T2W10    | z_density | 240 |       -0.075 |            0.25  |        -0.083 |             0.203 | School year  | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |

### 4.4.3 COG raw Partial Spearman Adjusted for School Year

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:--------------------------------------------------------------------|
| Activity 2 | T2W20    | edges     | 237 |        0.2   |           0.002  |         0.18  |             0.005 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W10    | edges     | 240 |        0.202 |           0.002  |         0.18  |             0.005 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W40    | edges     | 222 |        0.2   |           0.003  |         0.178 |             0.008 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W30    | edges     | 230 |        0.194 |           0.003  |         0.173 |             0.009 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 7 | T7W50    | edges     | 243 |        0.142 |           0.027  |         0.164 |             0.01  | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 7 | T7W40    | edges     | 248 |        0.136 |           0.033  |         0.157 |             0.013 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 6 | T6W30    | l3        | 252 |       -0.208 |           0.0009 |        -0.157 |             0.013 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 7 | T7W30    | edges     | 250 |        0.131 |           0.039  |         0.152 |             0.016 | School year  | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 6 | T6W50    | l3        | 252 |       -0.208 |           0.0009 |        -0.151 |             0.017 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 6 | T6W40    | l3        | 252 |       -0.205 |           0.001  |        -0.15  |             0.018 | School year  | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |

### 4.4.4 COG z Partial Spearman Adjusted for School Year

| Activity   | Window   | Feature    |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence         | Brief interpretation                                                                           |
|:-----------|:---------|:-----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-----------------|:-----------------------------------------------------------------------------------------------|
| Activity 2 | T2W10    | z_l3       | 240 |       -0.117 |            0.071 |        -0.143 |             0.027 | School year  | negative    | exploratory      | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W10    | z_cc       | 240 |       -0.106 |            0.1   |        -0.131 |             0.043 | School year  | negative    | exploratory      | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_cc       | 237 |       -0.103 |            0.114 |        -0.131 |             0.044 | School year  | negative    | exploratory      | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_l3       | 237 |       -0.099 |            0.127 |        -0.129 |             0.047 | School year  | negative    | exploratory      | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W10    | z_atd      | 240 |       -0.135 |            0.036 |        -0.088 |             0.177 | School year  | negative    | exploratory      | Higher higher than permutation-baseline average total degree is associated with lower COG.     |
| Activity 2 | T2W20    | z_diameter | 237 |       -0.101 |            0.12  |        -0.08  |             0.222 | School year  | negative    | weak exploratory | Higher higher than permutation-baseline maximum path length is associated with lower COG.      |
| Activity 2 | T2W20    | z_asp      | 237 |       -0.1   |            0.125 |        -0.073 |             0.264 | School year  | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_lsc      | 237 |        0.083 |            0.205 |         0.067 |             0.302 | School year  | positive    | weak exploratory | Higher higher than permutation-baseline strongly connected span is associated with higher COG. |
| Activity 2 | T2W10    | z_asp      | 240 |       -0.112 |            0.084 |        -0.064 |             0.327 | School year  | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_atd      | 237 |        0.052 |            0.422 |         0.062 |             0.34  | School year  | positive    | weak exploratory | Higher higher than permutation-baseline average total degree is associated with higher COG.    |

## 4.5 Partial Spearman Results Adjusted for Age

### 4.5.1 MOT raw Partial Spearman Adjusted for Age

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                     |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:-------------------------------------------------------------------------|
| Activity 7 | T7W30    | lsc       | 250 |        0.174 |            0.006 |         0.173 |             0.006 | Age          | positive    | stable exploratory | Higher higher raw strongly connected span is associated with higher MOT. |
| Activity 7 | T7W40    | cc        | 248 |        0.145 |            0.022 |         0.151 |             0.017 | Age          | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W30    | cc        | 250 |        0.143 |            0.024 |         0.147 |             0.02  | Age          | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 2 | T2W30    | l2        | 230 |        0.138 |            0.037 |         0.147 |             0.027 | Age          | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W50    | cc        | 243 |        0.136 |            0.034 |         0.142 |             0.027 | Age          | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W50    | edges     | 243 |        0.136 |            0.034 |         0.139 |             0.03  | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 2 | T2W30    | pe        | 230 |        0.132 |            0.045 |         0.139 |             0.035 | Age          | positive    | stable exploratory | Higher higher raw reciprocal recurrence is associated with higher MOT.   |
| Activity 2 | T2W10    | l2        | 240 |        0.13  |            0.044 |         0.138 |             0.033 | Age          | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W40    | edges     | 248 |        0.135 |            0.034 |         0.138 |             0.031 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 2 | T2W20    | l2        | 237 |        0.127 |            0.051 |         0.134 |             0.04  | Age          | positive    | exploratory        | Higher higher raw two-node cycles is associated with higher MOT.         |

### 4.5.2 MOT z Partial Spearman Adjusted for Age

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                                         |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:---------------------------------------------------------------------------------------------|
| Activity 2 | T2W20    | z_cc      | 237 |       -0.163 |            0.012 |        -0.168 |             0.01  | Age          | negative    | stable exploratory | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W10    | z_l2      | 240 |        0.154 |            0.017 |         0.154 |             0.017 | Age          | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W10    | z_pe      | 240 |        0.15  |            0.02  |         0.151 |             0.02  | Age          | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W20    | z_l3      | 237 |       -0.146 |            0.024 |        -0.151 |             0.021 | Age          | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W20    | z_l2      | 237 |        0.136 |            0.036 |         0.137 |             0.036 | Age          | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W10    | z_l3      | 240 |       -0.128 |            0.047 |        -0.133 |             0.04  | Age          | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W20    | z_pe      | 237 |        0.131 |            0.044 |         0.132 |             0.043 | Age          | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W10    | z_cc      | 240 |       -0.124 |            0.055 |        -0.129 |             0.047 | Age          | negative    | exploratory        | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W20    | z_density | 237 |       -0.098 |            0.133 |        -0.108 |             0.097 | Age          | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |
| Activity 2 | T2W10    | z_density | 240 |       -0.075 |            0.25  |        -0.086 |             0.184 | Age          | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |

### 4.5.3 COG raw Partial Spearman Adjusted for Age

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence           | Brief interpretation                                                |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-------------------|:--------------------------------------------------------------------|
| Activity 2 | T2W10    | edges     | 240 |        0.202 |           0.002  |         0.186 |             0.004 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W20    | edges     | 237 |        0.2   |           0.002  |         0.186 |             0.004 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W40    | edges     | 222 |        0.2   |           0.003  |         0.184 |             0.006 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W30    | edges     | 230 |        0.194 |           0.003  |         0.179 |             0.007 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 6 | T6W30    | l3        | 252 |       -0.208 |           0.0009 |        -0.171 |             0.007 | Age          | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 7 | T7W50    | edges     | 243 |        0.142 |           0.027  |         0.166 |             0.01  | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 6 | T6W50    | l3        | 252 |       -0.208 |           0.0009 |        -0.166 |             0.008 | Age          | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 6 | T6W40    | l3        | 252 |       -0.205 |           0.001  |        -0.165 |             0.009 | Age          | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 7 | T7W40    | edges     | 248 |        0.136 |           0.033  |         0.158 |             0.013 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 7 | T7W30    | edges     | 250 |        0.131 |           0.039  |         0.152 |             0.016 | Age          | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |

### 4.5.4 COG z Partial Spearman Adjusted for Age

| Activity   | Window   | Feature    |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment   | Direction   | Evidence         | Brief interpretation                                                                           |
|:-----------|:---------|:-----------|----:|-------------:|-----------------:|--------------:|------------------:|:-------------|:------------|:-----------------|:-----------------------------------------------------------------------------------------------|
| Activity 2 | T2W10    | z_l3       | 240 |       -0.117 |            0.071 |        -0.14  |             0.03  | Age          | negative    | exploratory      | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W10    | z_cc       | 240 |       -0.106 |            0.1   |        -0.131 |             0.043 | Age          | negative    | exploratory      | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_cc       | 237 |       -0.103 |            0.114 |        -0.126 |             0.053 | Age          | negative    | weak exploratory | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_l3       | 237 |       -0.099 |            0.127 |        -0.124 |             0.058 | Age          | negative    | weak exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W10    | z_atd      | 240 |       -0.135 |            0.036 |        -0.094 |             0.149 | Age          | negative    | exploratory      | Higher higher than permutation-baseline average total degree is associated with lower COG.     |
| Activity 2 | T2W20    | z_diameter | 237 |       -0.101 |            0.12  |        -0.078 |             0.233 | Age          | negative    | weak exploratory | Higher higher than permutation-baseline maximum path length is associated with lower COG.      |
| Activity 2 | T2W20    | z_asp      | 237 |       -0.1   |            0.125 |        -0.07  |             0.282 | Age          | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_atd      | 237 |        0.052 |            0.422 |         0.069 |             0.294 | Age          | positive    | weak exploratory | Higher higher than permutation-baseline average total degree is associated with higher COG.    |
| Activity 2 | T2W10    | z_asp      | 240 |       -0.112 |            0.084 |        -0.065 |             0.318 | Age          | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_lsc      | 237 |        0.083 |            0.205 |         0.064 |             0.325 | Age          | positive    | weak exploratory | Higher higher than permutation-baseline strongly connected span is associated with higher COG. |

## 4.6 Partial Spearman Results Adjusted for Educational Level

### 4.6.1 MOT raw Partial Spearman Adjusted for Educational Level

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment        | Direction   | Evidence           | Brief interpretation                                                     |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:------------------|:------------|:-------------------|:-------------------------------------------------------------------------|
| Activity 7 | T7W30    | lsc       | 250 |        0.174 |            0.006 |         0.174 |             0.006 | Educational level | positive    | stable exploratory | Higher higher raw strongly connected span is associated with higher MOT. |
| Activity 7 | T7W40    | cc        | 248 |        0.145 |            0.022 |         0.145 |             0.023 | Educational level | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W30    | cc        | 250 |        0.143 |            0.024 |         0.142 |             0.025 | Educational level | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 2 | T2W30    | l2        | 230 |        0.138 |            0.037 |         0.138 |             0.038 | Educational level | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W50    | edges     | 243 |        0.136 |            0.034 |         0.136 |             0.034 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 7 | T7W50    | cc        | 243 |        0.136 |            0.034 |         0.136 |             0.035 | Educational level | positive    | stable exploratory | Higher higher raw local clustering is associated with higher MOT.        |
| Activity 7 | T7W40    | edges     | 248 |        0.135 |            0.034 |         0.135 |             0.034 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |
| Activity 2 | T2W30    | pe        | 230 |        0.132 |            0.045 |         0.132 |             0.046 | Educational level | positive    | stable exploratory | Higher higher raw reciprocal recurrence is associated with higher MOT.   |
| Activity 2 | T2W10    | l2        | 240 |        0.13  |            0.044 |         0.131 |             0.043 | Educational level | positive    | stable exploratory | Higher higher raw two-node cycles is associated with higher MOT.         |
| Activity 7 | T7W30    | edges     | 250 |        0.129 |            0.042 |         0.129 |             0.042 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher MOT.      |

### 4.6.2 MOT z Partial Spearman Adjusted for Educational Level

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment        | Direction   | Evidence           | Brief interpretation                                                                         |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:------------------|:------------|:-------------------|:---------------------------------------------------------------------------------------------|
| Activity 2 | T2W20    | z_cc      | 237 |       -0.163 |            0.012 |        -0.164 |             0.012 | Educational level | negative    | stable exploratory | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W10    | z_l2      | 240 |        0.154 |            0.017 |         0.153 |             0.018 | Educational level | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W10    | z_pe      | 240 |        0.15  |            0.02  |         0.15  |             0.02  | Educational level | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W20    | z_l3      | 237 |       -0.146 |            0.024 |        -0.146 |             0.025 | Educational level | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W20    | z_l2      | 237 |        0.136 |            0.036 |         0.136 |             0.037 | Educational level | positive    | stable exploratory | Higher higher than permutation-baseline two-node cycles is associated with higher MOT.       |
| Activity 2 | T2W20    | z_pe      | 237 |        0.131 |            0.044 |         0.131 |             0.044 | Educational level | positive    | stable exploratory | Higher higher than permutation-baseline reciprocal recurrence is associated with higher MOT. |
| Activity 2 | T2W10    | z_l3      | 240 |       -0.128 |            0.047 |        -0.128 |             0.048 | Educational level | negative    | stable exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower MOT.      |
| Activity 2 | T2W10    | z_cc      | 240 |       -0.124 |            0.055 |        -0.125 |             0.054 | Educational level | negative    | weak exploratory   | Higher higher than permutation-baseline local clustering is associated with lower MOT.       |
| Activity 2 | T2W20    | z_density | 237 |       -0.098 |            0.133 |        -0.098 |             0.132 | Educational level | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |
| Activity 2 | T2W10    | z_density | 240 |       -0.075 |            0.25  |        -0.076 |             0.245 | Educational level | negative    | weak exploratory   | Higher higher than permutation-baseline graph density is associated with lower MOT.          |

### 4.6.3 COG raw Partial Spearman Adjusted for Educational Level

| Activity   | Window   | Feature   |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment        | Direction   | Evidence           | Brief interpretation                                                |
|:-----------|:---------|:----------|----:|-------------:|-----------------:|--------------:|------------------:|:------------------|:------------|:-------------------|:--------------------------------------------------------------------|
| Activity 6 | T6W30    | l3        | 252 |       -0.208 |           0.0009 |        -0.199 |             0.001 | Educational level | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 6 | T6W50    | l3        | 252 |       -0.208 |           0.0009 |        -0.197 |             0.002 | Educational level | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 6 | T6W40    | l3        | 252 |       -0.205 |           0.001  |        -0.195 |             0.002 | Educational level | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 2 | T2W10    | edges     | 240 |        0.202 |           0.002  |         0.194 |             0.003 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W40    | edges     | 222 |        0.2   |           0.003  |         0.193 |             0.004 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W20    | edges     | 237 |        0.2   |           0.002  |         0.193 |             0.003 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 2 | T2W30    | edges     | 230 |        0.194 |           0.003  |         0.189 |             0.004 | Educational level | positive    | stable exploratory | Higher higher raw speech transitions is associated with higher COG. |
| Activity 6 | T6W200   | l3        | 248 |       -0.196 |           0.002  |        -0.182 |             0.004 | Educational level | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 6 | T6W150   | l3        | 252 |       -0.194 |           0.002  |        -0.177 |             0.005 | Educational level | negative    | stable exploratory | Higher higher raw three-node cycles is associated with lower COG.   |
| Activity 6 | T6W30    | cc        | 252 |       -0.171 |           0.007  |        -0.158 |             0.012 | Educational level | negative    | stable exploratory | Higher higher raw local clustering is associated with lower COG.    |

### 4.6.4 COG z Partial Spearman Adjusted for Educational Level

| Activity   | Window   | Feature    |   n |   Simple rho |   Simple p-value |   Partial rho |   Partial p-value | Adjustment        | Direction   | Evidence         | Brief interpretation                                                                           |
|:-----------|:---------|:-----------|----:|-------------:|-----------------:|--------------:|------------------:|:------------------|:------------|:-----------------|:-----------------------------------------------------------------------------------------------|
| Activity 2 | T2W10    | z_atd      | 240 |       -0.135 |            0.036 |        -0.124 |             0.055 | Educational level | negative    | exploratory      | Higher higher than permutation-baseline average total degree is associated with lower COG.     |
| Activity 2 | T2W10    | z_l3       | 240 |       -0.117 |            0.071 |        -0.118 |             0.07  | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W10    | z_cc       | 240 |       -0.106 |            0.1   |        -0.114 |             0.079 | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W20    | z_cc       | 237 |       -0.103 |            0.114 |        -0.112 |             0.087 | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline local clustering is associated with lower COG.         |
| Activity 2 | T2W10    | z_asp      | 240 |       -0.112 |            0.084 |        -0.106 |             0.103 | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_diameter | 237 |       -0.101 |            0.12  |        -0.099 |             0.127 | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline maximum path length is associated with lower COG.      |
| Activity 2 | T2W20    | z_l3       | 237 |       -0.099 |            0.127 |        -0.099 |             0.129 | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline three-node cycles is associated with lower COG.        |
| Activity 2 | T2W20    | z_asp      | 237 |       -0.1   |            0.125 |        -0.099 |             0.129 | Educational level | negative    | weak exploratory | Higher higher than permutation-baseline average shortest path is associated with lower COG.    |
| Activity 2 | T2W20    | z_lsc      | 237 |        0.083 |            0.205 |         0.083 |             0.207 | Educational level | positive    | weak exploratory | Higher higher than permutation-baseline strongly connected span is associated with higher COG. |
| Activity 2 | T2W10    | z_lsc      | 240 |        0.074 |            0.252 |         0.071 |             0.274 | Educational level | positive    | weak exploratory | Higher higher than permutation-baseline strongly connected span is associated with higher COG. |

# 5. Main Findings

## 5.1 COG Findings

The current evidence favors a raw multi-activity model for COG. Activity 2 `edges_T2W10` and Activity 7 `edges_T7W50` contribute positive signal, consistent with a higher effective number of transitions or greater discourse continuity. Activity 6 `l3_T6W30` contributes negative signal, consistent with lower order-3 cyclic structure in reflective or counterfactual responses being associated with higher COG. The strongest raw COG correlations are stable across simple and partial Spearman analyses, especially for Activity 2 edges and Activity 6 three-node cycles.

## 5.2 MOT Findings

The raw MOT signal is weaker and more heterogeneous. Raw associations emphasize Activity 7 `lsc`, `cc`, and `edges`, plus Activity 2 reciprocal-cycle features, but effect sizes are modest. The clearer conceptual pattern comes from Activity 2 z-scores: `z_l2` and `z_pe` are positive, while `z_cc` and `z_l3` are negative. This suggests higher reciprocal simple recurrence relative to the permutation baseline, but lower clustering or local closure relative to that baseline.

## 5.3 Incomplete Z-Score Interpretation

The z-score interpretation must remain incomplete because z-score results for Activities 6 and 7 are not present in the archive. Activity 2 z-scores are informative for MOT and mildly informative for COG, but conclusions about normalized discourse structure across all activities require the pending Activity 6 and Activity 7 z-score outputs.

# 6. Regression and RidgeCV Results

## 6.1 COG Regression Models

| Model type   | Metric type   | Run                    | Features                                                                                                          | Covariates   |   n |   Predictors |   Mean $R^2$ |   SD $R^2$ |   Mean RMSE |   Mean Spearman rho |   SD Spearman rho | $R^2 < 0$   | Mean coefficients                                                                                                                                                                | Interpretation                                                                                                 |
|:-------------|:--------------|:-----------------------|:------------------------------------------------------------------------------------------------------------------|:-------------|----:|-------------:|-------------:|-----------:|------------:|--------------------:|------------------:|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| RidgeCV      | mixed         | COG_COG_3raw+z1_cov1   | edges_T2W10, l3_T6W30, edges_T7W50, z_l3_T2W10, School year                                                       | School year  | 232 |            5 |        0.046 |      0.093 |       2.102 |               0.292 |             0.121 | 26.8%       | z_l3_T2W10=-0.961; edges_T7W50=0.872; edges_T2W10=0.636; l3_T6W30=-0.480; School year=0.173                                                                                      | low positive cross-validated signal; consistent with multi-activity COG pattern.                               |
| RidgeCV      | mixed         | COG_COG_3raw+z3_cov1   | edges_T2W10, l3_T6W30, edges_T7W50, z_l3_T2W20, School year                                                       | School year  | 230 |            5 |        0.041 |      0.103 |       2.106 |               0.291 |             0.116 | 29.8%       | edges_T2W10=1.056; edges_T7W50=0.916; z_l3_T2W20=-0.515; l3_T6W30=-0.490; School year=0.181                                                                                      | low positive cross-validated signal; consistent with multi-activity COG pattern.                               |
| RidgeCV      | mixed         | COG_COG_3raw+z2_cov1   | edges_T2W10, l3_T6W30, edges_T7W50, z_cc_T2W10, School year                                                       | School year  | 232 |            5 |        0.034 |      0.089 |       2.116 |               0.276 |             0.122 | 29.2%       | edges_T7W50=0.837; z_cc_T2W10=-0.783; edges_T2W10=0.588; l3_T6W30=-0.470; School year=0.170                                                                                      | low positive cross-validated signal; consistent with multi-activity COG pattern.                               |
| RidgeCV      | raw           | COG_3feat_sch_raw_cov1 | edges_T2W10, l3_T6W30, edges_T7W50, School year                                                                   | School year  | 232 |            4 |        0.027 |      0.08  |       2.124 |               0.268 |             0.116 | 33.0%       | edges_T7W50=0.857; edges_T2W10=0.535; l3_T6W30=-0.421; School year=0.167                                                                                                         | low positive cross-validated signal; consistent with multi-activity COG pattern.                               |
| RidgeCV      | raw           | COG_2feat_raw          | edges_T2W10, l3_T6W30                                                                                             | none         | 240 |            2 |        0.019 |      0.07  |       2.158 |               0.244 |             0.123 | 32.2%       | edges_T2W10=1.271; l3_T6W30=-0.654                                                                                                                                               | very low positive cross-validated signal; consistent with multi-activity COG pattern.                          |
| OLS          | raw           | COG_2feat_raw          | edges_T2W10, l3_T6W30                                                                                             | none         | 240 |            2 |        0.018 |      0.075 |       2.157 |               0.248 |             0.124 | 33.2%       | edges_T2W10=1.700; l3_T6W30=-0.661                                                                                                                                               | very low positive cross-validated signal; consistent with multi-activity COG pattern.                          |
| OLS          | raw           | COG_3feat_raw          | edges_T2W10, l3_T6W30, edges_T7W50                                                                                | none         | 232 |            3 |        0.004 |      0.095 |       2.149 |               0.271 |             0.119 | 38.2%       | edges_T2W10=1.706; edges_T7W50=1.190; l3_T6W30=-0.624                                                                                                                            | very low positive cross-validated signal; consistent with multi-activity COG pattern.                          |
| OLS          | z             | COG_2feat_z            | z_l3_T2W10, z_cc_T2W10                                                                                            | none         | 240 |            2 |       -0.031 |      0.065 |       2.211 |               0.06  |             0.116 | 64.8%       | z_l3_T2W10=-2.194; z_cc_T2W10=1.572                                                                                                                                              | below mean-baseline by $R^2$.                                                                                  |
| RidgeCV      | raw           | COG_top10_raw          | edges_T2W20, edges_T2W10, edges_T2W40, edges_T2W30, lsc_T2W20, lsc_T2W30, l3_T2W10, cc_T2W10, lcc_T2W10, cc_T2W20 | none         | 222 |           10 |       -0.033 |      0.053 |       2.227 |               0.103 |             0.144 | 70.2%       | l3_T2W10=-0.166; edges_T2W40=0.112; lsc_T2W30=0.040; edges_T2W30=0.035; cc_T2W10=-0.031; lsc_T2W20=0.024; cc_T2W20=-0.021; edges_T2W20=0.006; lcc_T2W10=0.002; edges_T2W10=0.001 | below mean-baseline by $R^2$; larger top-k model likely redundant; consistent with multi-activity COG pattern. |
| RidgeCV      | z             | COG_3feat_z            | z_l3_T2W10, z_cc_T2W10, z_cc_T2W20                                                                                | none         | 237 |            3 |       -0.039 |      0.055 |       2.233 |               0.109 |             0.134 | 80.2%       | z_l3_T2W10=-0.400; z_cc_T2W20=-0.074; z_cc_T2W10=0.038                                                                                                                           | below mean-baseline by $R^2$.                                                                                  |
| OLS          | z             | COG_top5_z             | z_atd_T2W10, z_l3_T2W10, z_cc_T2W10, z_cc_T2W20, z_asp_T2W10                                                      | none         | 237 |            5 |       -0.04  |      0.079 |       2.233 |               0.109 |             0.125 | 64.2%       | z_l3_T2W10=-1.958; z_atd_T2W10=-1.154; z_cc_T2W10=0.832; z_asp_T2W10=-0.239; z_cc_T2W20=0.171                                                                                    | below mean-baseline by $R^2$.                                                                                  |
| OLS          | z             | COG_3feat_z            | z_l3_T2W10, z_cc_T2W10, z_cc_T2W20                                                                                | none         | 237 |            3 |       -0.042 |      0.068 |       2.235 |               0.042 |             0.129 | 70.2%       | z_l3_T2W10=-2.252; z_cc_T2W10=1.787; z_cc_T2W20=-0.106                                                                                                                           | below mean-baseline by $R^2$.                                                                                  |

## 6.2 MOT Regression Models

| Model type   | Metric type   | Run                  | Features                                                                                                                         | Covariates   |   n |   Predictors |   Mean $R^2$ |   SD $R^2$ |   Mean RMSE |   Mean Spearman rho |   SD Spearman rho | $R^2 < 0$   | Mean coefficients                                                                                                                                                                                  | Interpretation                                                                                                     |
|:-------------|:--------------|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------|:-------------|----:|-------------:|-------------:|-----------:|------------:|--------------------:|------------------:|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| RidgeCV      | z             | MOT_3feat_z          | z_cc_T2W20, z_l2_T2W10, z_pe_T2W10                                                                                               | none         | 237 |            3 |        0.004 |      0.061 |       7.614 |               0.154 |             0.124 | 37.8%       | z_pe_T2W10=2.516; z_cc_T2W20=-2.164; z_l2_T2W10=1.295                                                                                                                                              | very low positive cross-validated signal; consistent with Activity 2 z-score MOT pattern.                          |
| RidgeCV      | mixed         | MOT_MOT_3z           | z_cc_T2W20, z_l2_T2W10, z_pe_T2W10                                                                                               | none         | 237 |            3 |        0.004 |      0.061 |       7.614 |               0.154 |             0.124 | 37.8%       | z_pe_T2W10=2.516; z_cc_T2W20=-2.164; z_l2_T2W10=1.295                                                                                                                                              | very low positive cross-validated signal; consistent with Activity 2 z-score MOT pattern.                          |
| OLS          | z             | MOT_2feat_z          | z_cc_T2W20, z_l2_T2W10                                                                                                           | none         | 237 |            2 |        0.002 |      0.071 |       7.621 |               0.159 |             0.125 | 40.2%       | z_l2_T2W10=4.954; z_cc_T2W20=-2.730                                                                                                                                                                | very low positive cross-validated signal; consistent with Activity 2 z-score MOT pattern.                          |
| RidgeCV      | mixed         | MOT_MOT_mix          | z_cc_T2W20, z_l2_T2W10, lsc_T7W30                                                                                                | none         | 235 |            3 |        0.002 |      0.065 |       7.676 |               0.185 |             0.125 | 39.0%       | z_l2_T2W10=3.188; z_cc_T2W20=-2.133; lsc_T7W30=0.515                                                                                                                                               | very low positive cross-validated signal; consistent with Activity 2 z-score MOT pattern.                          |
| OLS          | z             | MOT_3feat_z          | z_cc_T2W20, z_l2_T2W10, z_pe_T2W10                                                                                               | none         | 237 |            3 |       -0.003 |      0.072 |       7.641 |               0.133 |             0.124 | 42.5%       | z_pe_T2W10=6.997; z_l2_T2W10=-2.635; z_cc_T2W20=-2.521                                                                                                                                             | below mean-baseline by $R^2$; consistent with Activity 2 z-score MOT pattern.                                      |
| RidgeCV      | z             | MOT_top10_z          | z_cc_T2W20, z_l2_T2W10, z_pe_T2W10, z_l3_T2W20, z_l2_T2W20, z_l3_T2W10, z_pe_T2W20, z_cc_T2W10, z_density_T2W20, z_density_T2W10 | none         | 237 |           10 |       -0.005 |      0.058 |       7.649 |               0.13  |             0.125 | 43.8%       | z_pe_T2W20=0.730; z_cc_T2W20=-0.634; z_l2_T2W20=0.514; z_l3_T2W20=-0.457; z_pe_T2W10=0.387; z_density_T2W20=-0.381; z_l2_T2W10=0.279; z_l3_T2W10=-0.250; z_cc_T2W10=-0.213; z_density_T2W10=-0.125 | below mean-baseline by $R^2$; larger top-k model likely redundant; consistent with Activity 2 z-score MOT pattern. |
| RidgeCV      | mixed         | MOT_MOT_2raw         | lsc_T7W30, cc_T7W40                                                                                                              | none         | 236 |            2 |       -0.006 |      0.069 |       7.728 |               0.217 |             0.117 | 43.0%       | cc_T7W40=35.710; lsc_T7W30=0.733                                                                                                                                                                   | below mean-baseline by $R^2$.                                                                                      |
| RidgeCV      | z             | MOT_2feat_sch_z_cov1 | z_cc_T2W20, z_l2_T2W10, School year                                                                                              | School year  | 237 |            3 |       -0.008 |      0.064 |       7.659 |               0.138 |             0.126 | 45.8%       | z_l2_T2W10=3.657; z_cc_T2W20=-2.485; School year=0.061                                                                                                                                             | below mean-baseline by $R^2$; consistent with Activity 2 z-score MOT pattern.                                      |
| OLS          | raw           | MOT_2feat_raw        | lsc_T7W30, l2_T2W30                                                                                                              | none         | 228 |            2 |       -0.009 |      0.079 |       7.576 |               0.187 |             0.139 | 45.8%       | l2_T2W30=2.519; lsc_T7W30=0.511                                                                                                                                                                    | below mean-baseline by $R^2$.                                                                                      |
| OLS          | z             | MOT_top5_z           | z_cc_T2W20, z_l2_T2W10, z_pe_T2W10, z_l3_T2W20, z_l2_T2W20                                                                       | none         | 237 |            5 |       -0.015 |      0.077 |       7.683 |               0.113 |             0.124 | 51.2%       | z_l2_T2W10=-6.629; z_pe_T2W10=5.886; z_l2_T2W20=2.981; z_cc_T2W20=-2.868; z_l3_T2W20=0.097                                                                                                         | below mean-baseline by $R^2$; consistent with Activity 2 z-score MOT pattern.                                      |
| OLS          | raw           | MOT_3feat_raw        | lsc_T7W30, l2_T2W30, asp_T6W30                                                                                                   | none         | 228 |            3 |       -0.016 |      0.081 |       7.6   |               0.161 |             0.134 | 50.5%       | l2_T2W30=2.449; asp_T6W30=-1.108; lsc_T7W30=0.486                                                                                                                                                  | below mean-baseline by $R^2$.                                                                                      |
| RidgeCV      | raw           | MOT_top10_raw        | lsc_T7W30, cc_T7W40, cc_T7W30, cc_T7W50, edges_T7W50, edges_T7W40, edges_T7W30, edges_T7W20, cc_T7W20, diameter_T7W30            | none         | 243 |           10 |       -0.043 |      0.06  |       7.79  |               0.138 |             0.113 | 74.8%       | cc_T7W50=5.761; cc_T7W40=5.593; cc_T7W30=3.954; edges_T7W30=-3.670; cc_T7W20=2.552; edges_T7W50=2.207; edges_T7W20=0.870; lsc_T7W30=0.424; edges_T7W40=0.373; diameter_T7W30=-0.069                | below mean-baseline by $R^2$; larger top-k model likely redundant.                                                 |

## 6.3 Prediction Compression in Diagnostic Figures

Observed-versus-predicted figures and the prediction CSV files show that predictions are compressed around the target mean. The scatter points aggregate predictions from repeated Monte Carlo splits, so the points are not independent observations.

| Model                             | Observed range   | Predicted range   |   Observed SD |   Predicted SD |   Aggregated Spearman rho | Interpretation                                                                                                               |
|:----------------------------------|:-----------------|:------------------|--------------:|---------------:|--------------------------:|:-----------------------------------------------------------------------------------------------------------------------------|
| COG RidgeCV 3raw+z1 + School year | 2 to 15          | 4.62 to 9.95      |         2.19  |          0.675 |                     0.27  | Predictions are compressed around the target mean and should not be treated as independent points across Monte Carlo splits. |
| MOT RidgeCV z_cc + z_l2 + z_pe    | 0 to 32          | 4.31 to 24.02     |         7.723 |          1.459 |                     0.119 | Predictions are compressed around the target mean and should not be treated as independent points across Monte Carlo splits. |

## 6.4 Regression Interpretation

The regression results show non-random signal but low predictive power. COG is consistently stronger than MOT: the best available COG RidgeCV model reaches mean $R^2=0.046$ and mean Spearman $\rho=0.292$, while the best available MOT RidgeCV z-score model reaches mean $R^2=0.004$ and mean Spearman $\rho=0.154$. Models with many variables or large top-k feature sets tend to perform worse, probably because graph features are redundant across windows and activities and because feature screening was performed before cross-validation. The recommended strategy is to use reduced, target-specific models, and for COG to prioritize multi-activity raw models.

# 7. Permutation Tests

## 7.1 Available Model-Level Permutation Tests

| Run         | Target   | Model   | Metric type in output   | Features                                       | Covariates   |   n |   Predictors |   Permutations |   Real $R^2$ |   Null mean |   Null SD | Null 95% interval   |   Empirical p-value | Significant   | Command stored in outputs   | Interpretation                                                                                                       | Note                                                                                                                                |
|:------------|:---------|:--------|:------------------------|:-----------------------------------------------|:-------------|----:|-------------:|---------------:|-------------:|------------:|----------:|:--------------------|--------------------:|:--------------|:----------------------------|:---------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| COG_3raw+z1 | COG      | RidgeCV | raw                     | edges_T2W10, l3_T6W30, edges_T7W50, z_l3_T2W10 | School year  | 232 |            5 |           5000 |        0.125 |      -0.035 |     0.05  | -0.171 to 0.019     |              0.0002 | True          | not available               | Full model exceeds the null distribution; this does not isolate incremental graph-feature value above the covariate. | This is the closest available COG permutation output; it includes the three raw COG features, one z-score feature, and School year. |
| MOT_3z      | MOT      | RidgeCV | raw                     | z_cc_T2W20, z_l2_T2W10, z_pe_T2W10             | none         | 237 |            3 |           5000 |        0.018 |      -0.032 |     0.047 | -0.163 to 0.013     |              0.02   | True          | not available               | Full z-score model exceeds the null distribution with a small real $R^2$.                                            | The output column labels feature_type as raw, but the listed predictors are z-score features.                                       |

## 7.2 Interpretation of the COG Permutation Test

The archive does not contain a pure raw-only COG permutation test for `edges_T2W10`, `l3_T6W30`, `edges_T7W50`, and School year alone. The closest available output is `COG_3raw+z1`, which validates a RidgeCV model containing those three raw graph features, `z_l3_T2W10`, and School year. Its real $R^2=0.125$ is above the null 95% interval of $-0.171$ to $0.019$, with empirical p-value $0.0002$. This supports a non-random full-model signal, but it does not prove that graph metrics add value beyond School year.

## 7.3 Interpretation of the MOT Permutation Test

The available MOT permutation test is `MOT_3z`, with predictors `z_cc_T2W20`, `z_l2_T2W10`, and `z_pe_T2W10`. Its real $R^2=0.018$ is slightly above the null 95% interval of $-0.163$ to $0.013$, with empirical p-value $0.0196$. This supports non-random full-model signal for the Activity 2 z-score pattern, but the absolute predictive effect remains small.

## 7.4 Recommended Incremental Permutation Analysis

A future permutation analysis should compare a base model containing only the covariate against a full model containing the covariate plus graph metrics. The key statistic should be $\Delta R^2 = R^2_\mathrm{full} - R^2_\mathrm{base}$. This would test whether graph structure contributes incremental information beyond School year, Age, or Educational level.

# 8. Methodological Limitations and Pending Work

## 8.1 Feature Selection Leakage Risk

Feature selection was informed by correlations or heatmaps computed on the full dataset. This can leak test-set information into model design. The next implementation should perform feature selection inside each cross-validation split, preferably with a nested cross-validation design.

## 8.2 Predictor Standardization for RidgeCV

The current RidgeCV code does not apply `StandardScaler` inside a scikit-learn `Pipeline`. Ridge regularization is scale-sensitive, so future RidgeCV models should standardize predictors within each training split before model fitting.

## 8.3 Z-Score Invariance and Missing Windows

Some graph quantities are invariant or weakly informative under within-segment permutation. `wc`, `nodes`, and often `edges` can remain unchanged after shuffling within fixed windows and fixed segment boundaries. This can produce zero variance in the null distribution or non-informative z-scores. Activity 2 z-score correlation outputs include only W10 and W20, and z-score outputs for Activities 6 and 7 are pending.

## 8.4 Multiple Testing and Effect Size

The correlation analysis screens many metrics, windows, targets, metric types, and adjustment variables. P-values should therefore be treated as uncorrected exploratory evidence. The strongest individual correlations are modest in magnitude, mostly around $|\rho|=0.13$ to $0.21$.

## 8.5 Regression Diagnostics

The observed-versus-predicted scatter plots aggregate predictions from multiple Monte Carlo splits. A participant can appear multiple times across split iterations, so plotted points are not independent. These figures are useful diagnostics for prediction compression and rank association, not independent-sample scatter plots.

## 8.6 Pending Analyses

The main pending analyses are z-score extraction and correlation for Activities 6 and 7, nested feature selection inside cross-validation, standardized RidgeCV pipelines, incremental covariate-versus-full permutation tests, and corrected or stability-based inference for the correlation screening stage.

# 9. Conclusion

## 9.1 General Conclusion

COG is the target with the strongest evidence so far, mainly through a reduced raw multi-activity model that combines positive transition-continuity signal from Activity 2 and Activity 7 with negative three-cycle signal from Activity 6. MOT shows weaker predictive performance, but Activity 2 z-scores form a coherent conceptual pattern: more reciprocal simple recurrence and less clustering or local closure relative to the permutation baseline. The z-score results for Activities 6 and 7 are necessary before final conclusions can be made about normalized discourse structure across the full SpeechGraph protocol.
