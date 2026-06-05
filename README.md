# SpeechGraph

SpeechGraph is a computational pipeline for analyzing the structure of transcribed speech using directed lexical graphs. It was designed to extract graph-theoretic and recurrence-based features from orthographically transcribed speech and evaluate their association with impulsivity dimensions measured by the Barratt Impulsiveness Scale.

The pipeline:

1. Parses participant speech from raw transcripts.
2. Converts token sequences into directed lexical transition graphs.
3. Computes graph metrics over sliding windows.
4. Builds null distributions through within-segment token permutation.
5. Computes z-scores for graph metrics relative to randomized baselines.
6. Tests associations with impulsivity dimensions using Spearman and partial Spearman correlations.
7. Evaluates predictive models using Monte Carlo cross-validated linear regression.

---

## Overview of the Pipeline

SpeechGraph transforms speech into a directed graph representation:

- **Nodes** represent unique lexical tokens.
- **Directed edges** represent transitions between consecutive tokens.
- **Edge weights** count how often each ordered transition occurs.
- **Segment boundaries** prevent transitions from being created across pause-delimited speech units.

This allows speech structure to be quantified through graph topology, recurrence, connectivity, and path-based measures.

---

## Graph Construction from Speech Transcriptions

Raw transcripts are parsed with `load_transcript_txt()` from `src.preprocessing.loaders`. The parser extracts activity-specific text blocks and retains only participant speech.

Annotation markers such as:

- `[[PAUSA]]`
- `[[DI]]`
- `[[DP]]`

are normalized into segment boundaries by `tokenize_segments()` from `src.preprocessing.tokenizer`.

This function returns:

- a list of token segments;
- a boolean boundary vector indicating where segment breaks occur.

A directed speech graph is then defined as:

$$
G = (V, E)
$$

where each vertex $v \in V$ is a unique lexical token and each directed edge $(v_i, v_{i+1}) \in E$ represents a transition between consecutive tokens within the same segment.

Edge weights are computed with `edge_counts()` from `src.graphs.builder`:

$$
w(v_i, v_j) = \text{number of times the transition } v_i \rightarrow v_j \text{ occurs}
$$

The weighted adjacency matrix is built with `adjacency_matrix()`:

$$
A_{ij} = w(v_i, v_j)
$$

Segment boundaries ensure that no edge is created across pause-delimited units, preserving the natural segmentation of discourse.

---

## Graph Metrics

Graph metrics are computed with `compute_metrics()` from `src.graphs.metrics`.

For a token sequence and its segment boundary vector, the function computes fifteen graph-theoretic measures. Let:

- $n = |V|$ be the number of unique lexical tokens;
- $m = \sum_{i,j} w_{ij}$ be the total directed edge weight;
- $\text{wc}$ be the number of tokens in the window.

The computed metrics include:

| Metric | Description |
|---|---|
| `wc` | Word count, defined as the total number of tokens. |
| `nodes` | Vocabulary size, defined as the number of unique tokens. |
| `edges` | Total number of directed transitions. |
| `re` | Repeated edges, defined as transitions observed more than once. |
| `pe` | Parallel or reciprocal edge count, computed from the adjacency matrix as implemented in `compute_metrics()`. |
| `l1` | Self-loops, computed as $\operatorname{tr}(\mathbf{A})$. |
| `l2` | Two-cycles, corresponding to reciprocal directed pairs. |
| `l3` | Three-cycles, computed as $\frac{1}{3}\operatorname{tr}(\tilde{\mathbf{A}}^3)$, where $\tilde{\mathbf{A}}$ is the adjacency matrix with a zeroed diagonal. |
| `lcc` | Size of the largest connected component in the undirected projection. |
| `lsc` | Size of the largest strongly connected component in the directed graph. |
| `atd` | Average token degree, computed as $2m/n$. |
| `density` | Proportion of observed undirected non-self pairs relative to all possible pairs. |
| `diameter` | Diameter of the largest undirected connected component. |
| `asp` | Average shortest path length of the largest undirected connected component. |
| `cc` | Average clustering coefficient of the largest undirected connected component. |

Path-based measures are computed on the largest undirected connected component when its size is greater than one. Connectivity measures are computed using `networkx`.

---

## Sliding Window Decomposition

The function `sliding_windows()` from `src.graphs.windowing` decomposes a token sequence $T$ of length $N$ into overlapping windows of size $w$ and step size $s$.

Each window is defined as:

$$
W_k = \{t_{1+(k-1)s}, \ldots, t_{w+(k-1)s}\}
$$

for:

$$
k = 1, \ldots, \left\lfloor \frac{N - w}{s} \right\rfloor + 1
$$

For each window, the function returns:

- the window tokens;
- the start and end indices;
- a local boundary vector.

Segment boundaries are propagated from the global boundary vector to each local window so that graph metrics continue to respect pause-delimited segmentation.

Window-level metrics are then aggregated per subject by computing:

- the mean across windows;
- the median across windows.

This produces one row per subject for each task-window configuration.

---

## Z-Score Computation via Permutation Testing

The module `src.pipeline.zscore_sg` extends the raw metric pipeline by computing z-scores relative to randomized null distributions.

For each window $W_k$, the observed token sequence and its segment boundaries are passed to `generate_random_graphs()` from `src.analysis.random`.

Random graphs are generated with `shuffle_within_segments()`, which:

1. partitions the token sequence according to segment boundaries;
2. randomly permutes tokens independently within each segment;
3. reassembles the shuffled segments into a full token sequence;
4. preserves the original segment structure.

This procedure preserves pause-delimited segmentation while disrupting the original lexical transition order.

For each of the $R$ randomized realizations, all graph metrics are recomputed.

Given an observed metric value $x_{\text{obs}}$ and randomized values:

$$
\{x^{(r)}\}_{r=1}^{R}
$$

the z-score is computed with `compute_z_scores()` as:

$$
z_x = \frac{x_{\text{obs}} - \mu_{\text{rand}}}{\sigma_{\text{rand}}}
$$

where:

$$
\mu_{\text{rand}} = \frac{1}{R} \sum_{r=1}^{R} x^{(r)}
$$

and:

$$
\sigma_{\text{rand}}^2 = \frac{1}{R} \sum_{r=1}^{R} \left(x^{(r)} - \mu_{\text{rand}}\right)^2
$$

A z-score of zero indicates that the observed value is centered in the null distribution. Positive and negative values indicate departures from the randomized baseline in standard deviation units.

The resulting z-score tables are saved as:

- `z_params_table`
- `z_means_params_table`
- `z_median_params_table`

These files are written to the same task-specific output directories as the raw metrics, using the same naming convention with a `z_` prefix.

---

## Correlation Analysis

The module `src.analysis.correlation_analysis` evaluates associations between graph-derived features and Barratt impulsivity dimensions.

The target dimensions are:

- `MOT`
- `COG`

Per-subject mean metrics from each task-window combination are merged with subject metadata using subject codes extracted from transcript filenames by `get_subject_code()`.

For each feature $f$ and target dimension $y$, the simple Spearman rank correlation is computed with `scipy.stats.spearmanr`:

$$
r_s(f, y)
$$

Partial Spearman correlation controlling for school year $S$ is computed with `pingouin.partial_corr()` using the Spearman method:

$$
r_{s,\text{partial}}(f, y \mid S)
$$

This is equivalent to computing the Spearman correlation between the residuals of rank-based regressions of $f$ and $y$ on $S$:

$$
r_{s,\text{partial}}(f, y \mid S)
=
r_s\left(f - \hat{f}(S),\; y - \hat{y}(S)\right)
$$

Features are ranked separately by the absolute values of:

- simple Spearman correlations;
- partial Spearman correlations.

Rankings are produced independently for:

- raw graph metrics;
- z-score graph metrics.

The top-ranked features for each target and feature type are exported as CSV files and used as candidate predictors in the regression stage.

---

## Monte Carlo Cross-Validated Linear Regression

The module `src.analysis.linear_regression_mc` implements predictive modeling of `MOT` and `COG` using the top $K$ features identified by correlation analysis.

For each target $y$ and feature type, the selected features form the design matrix:

$$
\mathbf{X} \in \mathbb{R}^{N \times K}
$$

A linear model is estimated using ordinary least squares with `sklearn.linear_model.LinearRegression`:

$$
\hat{y} = \beta_0 + \mathbf{X}\boldsymbol{\beta}
$$

Model performance is assessed through Monte Carlo cross-validation with $M$ iterations.

At each iteration $j$:

1. subjects are randomly split into training and test sets;
2. the training set contains 80% of the data;
3. the test set contains the remaining 20%;
4. the model is fitted on the training set;
5. predictions are evaluated on the held-out test set.

Three performance metrics are recorded per iteration.

### Coefficient of Determination

$$
R^2_j =
1 -
\frac{
\sum_{i \in \text{test}} (y_i - \hat{y}_i)^2
}{
\sum_{i \in \text{test}} (y_i - \bar{y}_{\text{train}})^2
}
$$

### Root Mean Squared Error

$$
\text{RMSE}_j =
\sqrt{
\frac{1}{n_{\text{test}}}
\sum_{i \in \text{test}} (y_i - \hat{y}_i)^2
}
$$

### Spearman Correlation Between Observed and Predicted Values

$$
\rho_j = r_s(y_{\text{test}}, \hat{y}_{\text{test}})
$$

The distribution of each metric across Monte Carlo iterations is summarized by its mean and standard deviation.

The pipeline also reports the proportion of iterations with:

$$
R^2_j < 0
$$

This value quantifies how often the model performs worse than a mean-baseline predictor.

---

## Visualization

SpeechGraph includes visualization utilities for both correlation analysis and regression performance.

### Correlation Heatmaps

The module `src.visualization.corr_metrics` generates heatmaps of correlation matrices using `seaborn`.

For each feature type, raw and z-score, the module produces a combined figure with:

- simple Spearman correlations;
- partial Spearman correlations;
- features as rows;
- task-window-target combinations as columns.

Each cell displays the correlation coefficient with three decimal places. The color scale represents both the magnitude and sign of the correlation.

### Regression Diagnostics

The module `src.visualization.lr_metrics` generates four diagnostic plots for each regression model:

1. histogram of Monte Carlo $R^2$ values;
2. histogram of RMSE values;
3. histogram of Spearman $\rho$ values;
4. scatter plot of observed versus predicted values.

The observed-versus-predicted scatter plot includes:

- the identity line;
- Pearson correlation between observed and predicted values;
- Spearman correlation between observed and predicted values.

Correlations are computed using predictions aggregated across Monte Carlo iterations.

---

## Main Modules

| Module | Purpose |
|---|---|
| `src.preprocessing.loaders` | Loads and parses transcript files. |
| `src.preprocessing.tokenizer` | Tokenizes speech and identifies segment boundaries. |
| `src.graphs.builder` | Builds lexical transition graphs and adjacency matrices. |
| `src.graphs.metrics` | Computes graph-theoretic speech metrics. |
| `src.graphs.windowing` | Applies sliding window decomposition. |
| `src.analysis.random` | Generates randomized token sequences within segment boundaries. |
| `src.pipeline.zscore_sg` | Computes z-score-normalized graph metrics. |
| `src.analysis.correlation_analysis` | Computes simple and partial Spearman correlations. |
| `src.analysis.linear_regression_mc` | Runs Monte Carlo cross-validated linear regression. |
| `src.visualization.corr_metrics` | Produces correlation heatmaps. |
| `src.visualization.lr_metrics` | Produces regression diagnostic figures. |

---

## Outputs

The pipeline produces subject-level tables for each task-window configuration, including:

- raw graph metrics;
- mean-aggregated graph metrics;
- median-aggregated graph metrics;
- z-score graph metrics;
- mean-aggregated z-score metrics;
- median-aggregated z-score metrics;
- correlation ranking tables;
- regression performance summaries;
- diagnostic visualizations.

Z-score outputs follow the same naming convention as raw metric outputs, with the prefix `z_`.

---

## Summary

SpeechGraph provides a reproducible framework for quantifying speech structure through graph-based analysis. By combining lexical transition graphs, sliding window decomposition, permutation-based z-scores, correlation analysis, and Monte Carlo regression, the pipeline enables systematic evaluation of how discourse organization relates to impulsivity-related behavioral dimensions.