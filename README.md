# SpeechGraph

A computational pipeline for the analysis of language structure through directed graph modeling of transcribed speech, with application to impulsivity assessment via the Barratt Impulsiveness Scale. The pipeline transforms orthographically transcribed speech into lexical transition graphs, extracts topological and recurrence-based graph metrics via sliding window decomposition, generates null distributions through within-segment token permutation, and evaluates the association between graph-derived features and impulsivity dimensions using correlation analysis and Monte Carlo cross-validated linear regression.

## Graph Construction from Speech Transcriptions

Raw transcripts are parsed by `load_transcript_txt()` from `src.preprocessing.loaders` into activity-specific text blocks, retaining only participant speech. Annotation markers such as `[[PAUSA]]`, `[[DI]]`, and `[[DP]]` are normalized into segment boundaries via `tokenize_segments()` from `src.preprocessing.tokenizer`, which returns a list of token segments and a boolean boundary vector indicating where segment breaks occur.

A directed speech graph $G = (V, E)$ is constructed from each token sequence such that each vertex $v \in V$ represents a unique lexical token and each directed edge $(v_i, v_{i+1}) \in E$ encodes a sequential transition between consecutive tokens within the same segment. Edge weights $w(v_i, v_j)$ count the number of times the ordered pair is observed, computed by `edge_counts()` from `src.graphs.builder`. The adjacency matrix $\mathbf{A} \in \mathbb{N}^{|V| \times |V|}$ is built by `adjacency_matrix()` with $A_{ij} = w(v_i, v_j)$. Segment boundaries ensure that no edge is created across pause-delimited speech units, preserving the natural segmentation of discourse.

## Graph Metrics

The function `compute_metrics()` from `src.graphs.metrics` computes fifteen graph-theoretic measures for a given token sequence and its segment boundaries. Let $n = |V|$ and $m = \sum_{i,j} w_{ij}$ denote the number of nodes and the total edge weight sum respectively. Word count $\text{wc}$ is the number of tokens in the window. Nodes $n$ is vocabulary size. Edges $m$ is the sum of all directed transition counts. Repeated edges $\text{re}$ is the number of transitions observed more than once. Parallel edges $\text{pe}$ counts reciprocal directed pairs and is computed as half the trace of the squared adjacency matrix with zeroed diagonal, $\frac{1}{2}\operatorname{tr}(\tilde{\mathbf{A}}^2)$. Self-loops $l_1$ is $\operatorname{tr}(\mathbf{A})$. Two-cycles $l_2$ counts reciprocal edge pairs. Three-cycles $l_3$ is $\frac{1}{3}\operatorname{tr}(\tilde{\mathbf{A}}^3)$.

The largest connected component $\text{lcc}$ and largest strongly connected component $\text{lsc}$ are computed via `networkx` on the undirected projection and the directed graph respectively. Average token degree $\text{atd}$ is $2m/n$. Density $d$ is the proportion of observed undirected non-self pairs relative to all possible pairs. Diameter $D$ and average shortest path length $L$ are computed on the largest undirected connected component when its size exceeds one. The clustering coefficient $C$ is the average clustering coefficient of that component.

## Sliding Window Decomposition

The function `sliding_windows()` from `src.graphs.windowing` decomposes a token sequence $T$ of length $N$ into overlapping windows of size $w$ with step $s$. Each window $W_k$ consists of tokens $t_{1+(k-1)s}$ through $t_{w+(k-1)s}$ for $k = 1, \ldots, \lfloor (N - w)/s \rfloor + 1$. Segment boundaries are propagated from the global boundary vector to each window so that the graph metrics respect pause-delimited segmentation within every window. For each window, the function yields the window tokens, start and end indices, and a local boundary vector. These window-level metric vectors are aggregated per subject by taking the mean and median across all windows produced from that subject's activity transcript, yielding one row per subject per task-window combination.

## Z-Score Computation via Permutation Testing

The module `src.pipeline.zscore_sg` extends the raw metric pipeline by computing z-scores relative to a null distribution generated through within-segment token shuffling. For each window $W_k$, the original token sequence and its segment boundaries are passed to `generate_random_graphs()` from `src.analysis.random`, which constructs $R$ random realizations by calling `shuffle_within_segments()`. This function partitions tokens by their segment boundaries and applies a random permutation independently within each segment, reassembling the tokens into a shuffled sequence that preserves the original segment structure and therefore the graph's boundary-respecting topology. For each of the $R$ realizations, all fifteen metrics are computed.

Given an observed metric value $x_{\text{obs}}$ and the set of $R$ random values $\{x^{(r)}\}_{r=1}^{R}$, the z-score is computed by `compute_z_scores()` as

$$
z_x = \frac{x_{\text{obs}} - \mu_{\text{rand}}}{\sigma_{\text{rand}}}
$$

where $\mu_{\text{rand}} = \frac{1}{R} \sum_{r=1}^{R} x^{(r)}$ and $\sigma_{\text{rand}}^2 = \frac{1}{R} \sum_{r=1}^{R} (x^{(r)} - \mu_{\text{rand}})^2$. A z-score of zero indicates that the observed metric is at the center of the null distribution; positive and negative values indicate departures from the randomized baseline in standard deviation units.

The resulting z-scores are saved as `z_params_table`, `z_means_params_table`, and `z_median_params_table` in the same task-specific output directories as the raw metrics, following the same file naming convention with a `z_` prefix.

## Correlation Analysis

The module `src.analysis.correlation_analysis` evaluates the association between each graph feature and the Barratt impulsivity dimensions MOT and COG. Per-subject mean metrics from every task-window combination are merged with subject metadata using subject codes extracted from transcript filenames by `get_subject_code()`.

For each feature $f$ and each target dimension $y \in \{\text{MOT}, \text{COG}\}$, the simple Spearman rank correlation $r_s(f, y)$ is computed using `scipy.stats.spearmanr`. The partial Spearman correlation controlling for School year $S$ is computed using `pingouin.partial_corr()` with the Spearman method, which is equivalent to computing the Spearman correlation between the residuals of rank-regressing $f$ on $S$ and $y$ on $S$,

$$
r_{s,\text{partial}}(f, y \mid S) = r_s\bigl(f - \hat{f}(S),\; y - \hat{y}(S)\bigr)
$$

where $\hat{f}(S)$ and $\hat{y}(S)$ are the monotonic predictions from rank-based regressions on $S$. Features are ranked by the absolute value of their simple and partial correlation coefficients separately for raw metrics and for z-score metrics. The top-ranked features for each target and feature type are exported as CSV tables to be used as predictors in the regression stage.

## Monte Carlo Cross-Validated Linear Regression

The module `src.analysis.linear_regression_mc` implements predictive modeling of MOT and COG using the top $K$ features identified by the correlation analysis. For each target $y$ and feature type (raw or z-score), the selected features form the design matrix $\mathbf{X} \in \mathbb{R}^{N \times K}$. A linear model of the form

$$
\hat{y} = \beta_0 + \mathbf{X} \boldsymbol{\beta}
$$

is estimated by ordinary least squares via `sklearn.linear_model.LinearRegression`.

Model performance is assessed through Monte Carlo cross-validation with $M$ iterations. In each iteration $j \in \{1, \ldots, M\}$, the $N$ subjects are randomly partitioned into a training set containing $80\%$ of the data and a test set containing the remaining $20\%$. The model is fitted on the training set and evaluated on the held-out test set. Three metrics are recorded per iteration: the coefficient of determination

$$
R^2_j = 1 - \frac{\sum_{i \in \text{test}} (y_i - \hat{y}_i)^2}{\sum_{i \in \text{test}} (y_i - \bar{y}_{\text{train}})^2},
$$

the root mean squared error

$$
\text{RMSE}_j = \sqrt{\frac{1}{n_{\text{test}}} \sum_{i \in \text{test}} (y_i - \hat{y}_i)^2},
$$

and the Spearman rank correlation $\rho_j = r_s(y_{\text{test}}, \hat{y}_{\text{test}})$ between observed and predicted values. The distribution of each metric across $M$ iterations is summarized by its mean and standard deviation, and the proportion of iterations with $R^2_j < 0$ is reported to quantify the frequency with which the model performs worse than the mean baseline.

## Visualization

The module `src.visualization.corr_metrics` produces heatmaps of the correlation matrices using `seaborn`. For each feature type (raw and z-score), a combined figure displays side-by-side the simple and partial Spearman correlation matrices with features as rows and task-window-target combinations as columns. Each cell displays the correlation coefficient with three decimal places, and the color scale reflects the magnitude and sign of the coefficient.

The module `src.visualization.lr_metrics` produces four diagnostic figures for each regression model: a histogram of $R^2$ values across Monte Carlo iterations, a histogram of RMSE values, a histogram of Spearman $\rho$ values, and a scatter plot of observed versus predicted values. The scatter plot includes the identity line and reports the Pearson and Spearman correlation coefficients between observed and predicted values aggregated across all iterations.
