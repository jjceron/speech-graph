# README — Speech-Graph NLP Features and Barratt Dimensions

## Analytical objective

This analysis evaluated whether **speech-graph NLP features** extracted from structured interview transcripts improved the prediction of four Barratt impulsivity dimensions: **TOTAL**, **NPLAN**, **MOT**, and **COG**. The central question was not whether the transcripts contained any statistical association with Barratt scores, but whether graph-based features of speech organization improved out-of-sample prediction beyond a mean baseline.

The main finding is precise: speech-graph features showed **weak, localized predictive signal** for **MOT** and **COG**, almost no useful predictive signal for **TOTAL**, and **no predictive signal for NPLAN**.

---

## Pipeline logic

The pipeline treated each transcript as ordered behavioral language data. Only the participant's speech was retained. Interviewer speech and irrelevant transcription labels were excluded. Each transcript was segmented into seven activities using the activity markers present in the files. Within each activity, speech was cleaned, lowercased, punctuation was removed, and pause/interruption markers were treated as boundaries so that graph edges were not created across discontinuous speech.

For example, a cleaned sequence such as:

```text
había una vez un pollo
```

was converted into directed lexical transitions:

```text
había → una → vez → un → pollo
```

Thus, each activity became a directed speech graph, where nodes represent lexical items and edges represent observed word-to-word transitions. This preserves local recurrence, connectedness, and graph topology rather than treating the transcript as a bag of words.

Graph metrics were extracted using sliding windows of **10**, **20**, and **30** words with step size **1**:

\[
W_{10}: (t_1,\ldots,t_{10}), (t_2,\ldots,t_{11}), (t_3,\ldots,t_{12}), \ldots
\]

The resulting window-level graph metrics were aggregated to the **subject × activity × window-size** level. This aggregation is important because overlapping windows are highly correlated; the final regression models did not treat each overlapping window as an independent subject.

The final modeling unit was:

\[
X_{s,a,w} \rightarrow y_{s,k}
\]

where \(s\) is the subject, \(a\) is the activity, \(w \in \{10,20,30\}\) is the window size, and \(k\) is one Barratt target among TOTAL, NPLAN, MOT, or COG.

Metadata were used only for merging and interpretation. Demographic or grouping variables such as age, gender, school, educational level, school year, and impulsivity group were **not used as predictors** in the primary NLP-to-Barratt models. This avoids inflating model performance through non-linguistic or derived psychometric information.

---

## Data integrity after correction

The final activity-window feature table was successfully deduplicated. Each subject appears at most once per activity and window size.

| Check | Final result | Interpretation |
|---|---:|---|
| Activity-window feature rows | **4,534** | Final modeling table after filtering and aggregation |
| Duplicate `code × activity × window` rows | **0** | Deduplication correction worked |
| Maximum unique subjects in a scheme | **252** | Consistent with available Barratt metadata |
| Number of activities | **7** | Activity-specific modeling was preserved |
| Window sizes | **10, 20, 30** | All planned temporal scales were evaluated |
| Targets | **TOTAL, NPLAN, MOT, COG** | Four Barratt dimensions modeled separately |

Subject counts varied by activity and window size because shorter responses cannot support larger windows.

| Activity | W10 n | W20 n | W30 n | Comment |
|---:|---:|---:|---:|---|
| 1 | 244 | 185 | 109 | Larger windows reduce sample size |
| 2 | 238 | 232 | 218 | Stable across windows |
| 3 | 249 | 226 | 188 | Moderate loss at W30 |
| 4 | **252** | **252** | **252** | Fully retained across windows |
| 5 | 234 | 112 | **37** | **W30 is underpowered and exploratory only** |
| 6 | **252** | **252** | **252** | Fully retained across windows |
| 7 | 250 | 250 | 250 | Stable across windows |

---

## Model specification

For each Barratt target, activity, and window size, a separate Ridge regression model was fitted:

\[
y_{s,k} = \beta_0 + X_{s,a,w}\beta + \varepsilon_s
\]

with Ridge regularization:

\[
\hat{\beta} = \arg\min_{\beta}\left[\sum_s (y_s - \beta_0 - X_s\beta)^2 + \alpha\lVert\beta\rVert_2^2\right]
\]

Performance was evaluated using cross-validated \(R^2\):

\[
R^2_{CV} = 1 - \frac{\sum_s (y_s - \hat{y}_s)^2}{\sum_s (y_s - \bar{y}_{train})^2}
\]

Interpretation of \(R^2_{CV}\):

| R² behavior | Meaning |
|---|---|
| **R² > 0** | NLP features improve prediction relative to the mean baseline |
| **R² ≈ 0** | NLP features add negligible predictive value |
| **R² < 0** | The model performs worse than the mean baseline |

A total of **84 models** were evaluated:

\[
4\;\text{targets} \times 7\;\text{activities} \times 3\;\text{window sizes} = 84\;\text{models}
\]

Each model used **13 non-constant speech-graph features**. Word count and edge count were effectively constant within fixed window sizes and therefore did not contribute as predictors.

---

## Model performance by Barratt dimension

| Target | Models tested | Positive R² models | Positive R² with n ≥ 100 | Best R² | Mean R² | Median R² | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| **MOT** | 21 | 3 | 3 | **0.0171** | -0.0367 | -0.0122 | **Weakest but clearest predictive signal** |
| **COG** | 21 | 4 | 4 | **0.0123** | -0.0194 | -0.0134 | **Weak localized signal** |
| TOTAL | 21 | 1 | 1 | 0.0018 | -0.0277 | -0.0144 | Negligible predictive value |
| **NPLAN** | 21 | **0** | **0** | **-0.0041** | -0.0347 | -0.0136 | **No evidence of predictive value** |

The result is not a strong predictive finding. The best models explain only a very small proportion of out-of-sample variance. The analysis supports the presence of **weak speech-graph signal**, mainly for **MOT** and **COG**, but does not support strong individual-level prediction.

---

## Positive models: where NLP features helped above baseline

Only models with \(R^2 > 0\) and \(n \geq 100\) are shown.

| Rank | Target | Activity | Window | n | R² | Pearson r | Pearson p | Spearman r | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| **1** | **MOT** | **7** | **30** | 250 | **0.0171** | **0.1357** | **0.0320** | 0.1190 | **Best overall model; weak but detectable signal** |
| **2** | **COG** | **3** | **10** | 249 | **0.0123** | 0.1157 | 0.0683 | 0.1011 | Best COG model; weak signal |
| 3 | MOT | 1 | 20 | 185 | 0.0063 | 0.0793 | 0.2834 | 0.1228 | Very small predictive gain |
| 4 | COG | 7 | 10 | 250 | 0.0060 | 0.0786 | 0.2155 | 0.0847 | Very small predictive gain |
| 5 | MOT | 2 | 20 | 232 | 0.0052 | 0.1031 | 0.1173 | 0.1285 | Very small predictive gain |
| 6 | COG | 4 | 30 | 252 | 0.0023 | 0.0506 | 0.4234 | 0.0542 | Negligible gain |
| 7 | TOTAL | 2 | 20 | 232 | 0.0018 | 0.0857 | 0.1936 | 0.0940 | Practically negligible |
| 8 | COG | 4 | 10 | 252 | 0.0000 | 0.0582 | 0.3572 | 0.0647 | Effectively null |

The only model with both positive \(R^2\) and a nominally significant Pearson association between predicted and observed scores was **MOT, Activity 7, W30**. Even there, the magnitude was small.

---

## Improvement over the mean baseline

The practical gain was assessed by comparing model error against a mean-prediction baseline.

\[
\Delta RMSE = RMSE_{baseline} - RMSE_{model}
\]

\[
\Delta MAE = MAE_{baseline} - MAE_{model}
\]

Positive values indicate that the speech-graph model improved prediction.

| Target | Activity | Window | R² | RMSE model | RMSE baseline | ΔRMSE | MAE model | MAE baseline | ΔMAE | Practical reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **MOT** | **7** | **30** | **0.0171** | 7.6070 | 7.6728 | **0.0658** | 6.4189 | 6.5448 | **0.1259** | **Largest practical improvement** |
| COG | 3 | 10 | 0.0123 | 2.2302 | 2.2440 | 0.0138 | 1.7218 | 1.7258 | 0.0040 | Small improvement |
| MOT | 1 | 20 | 0.0063 | 7.5297 | 7.5534 | 0.0237 | 6.3818 | 6.3854 | 0.0036 | Small improvement |
| COG | 7 | 10 | 0.0060 | 2.2312 | 2.2379 | 0.0067 | 1.7294 | 1.7210 | -0.0084 | RMSE improves, MAE worsens |
| MOT | 2 | 20 | 0.0052 | 7.6645 | 7.6844 | 0.0199 | 6.4911 | 6.5577 | 0.0666 | Small improvement |
| COG | 4 | 30 | 0.0023 | 2.2410 | 2.2435 | 0.0025 | 1.7351 | 1.7285 | -0.0066 | Negligible |
| TOTAL | 2 | 20 | 0.0018 | 11.9415 | 11.9524 | 0.0109 | 10.2449 | 10.4393 | 0.1944 | Numerically positive but R² negligible |
| COG | 4 | 10 | 0.0000 | 2.2435 | 2.2435 | 0.0000 | 1.7425 | 1.7285 | -0.0140 | Null |

The error table confirms that the positive \(R^2\) models produced only **minor numerical improvements** over baseline. The best model reduced RMSE by only **0.0658** points and MAE by **0.1259** points.

---

## Feature-level interpretation

Feature relevance was evaluated through mean absolute contribution within the positive models. These values indicate how strongly each standardized feature contributed to model predictions, not whether the feature is causal.

### Aggregated feature relevance across positive models

| Target | Main contributing features | Interpretation |
|---|---|---|
| **MOT** | **cc**, **pe**, **lsc**, **l3**, **re**, **diameter** | MOT signal was mainly related to clustering, repeated/parallel transitions, connectedness, recurrence, and graph path structure |
| **COG** | **l1**, **lsc**, **pe**, **l3**, **re**, **diameter** | COG signal was mainly related to short-range recurrence, strongly connected structure, repeated transitions, and graph scale/path structure |
| TOTAL | pe, lsc, cc, l2, re, diameter | Contributions occurred only in one very weak model, so they should not be interpreted strongly |
| **NPLAN** | None | **No positive predictive model was found** |

### Top aggregated features by target

| Target | Feature | Models | Mean absolute contribution | Mean share | Best model R² |
|---|---|---:|---:|---:|---:|
| **MOT** | **cc** | 3 | **0.3677** | **0.1578** | **0.0171** |
| **MOT** | **pe** | 3 | **0.2286** | 0.1142 | **0.0171** |
| MOT | lsc | 3 | 0.1707 | 0.0825 | 0.0171 |
| MOT | l3 | 3 | 0.1504 | 0.0697 | 0.0171 |
| MOT | re | 3 | 0.1449 | 0.0740 | 0.0171 |
| MOT | diameter | 3 | 0.1425 | 0.0785 | 0.0171 |
| **COG** | **l1** | 4 | **0.0326** | **0.1312** | **0.0123** |
| **COG** | **lsc** | 4 | **0.0254** | 0.1148 | **0.0123** |
| COG | pe | 4 | 0.0252 | 0.0934 | 0.0123 |
| COG | l3 | 4 | 0.0222 | 0.0875 | 0.0123 |
| COG | re | 4 | 0.0178 | 0.0695 | 0.0123 |
| COG | diameter | 4 | 0.0176 | 0.0780 | 0.0123 |

The most coherent pattern is that predictive signal, when present, is not driven by simple word volume. It is associated with **recurrence**, **connectedness**, **clustering**, and **path structure** in speech graphs.

---

## Feature contributions in the strongest models

### Best MOT model: Activity 7, W30

| Feature | Mean absolute contribution | Share within model | Univariate r | p | Reading |
|---|---:|---:|---:|---:|---|
| **cc** | **0.7919** | **0.3168** | **0.1337** | **0.0346** | **Main model driver; weak positive univariate association** |
| l3 | 0.3836 | 0.1534 | 0.0042 | 0.9471 | Important multivariately but not univariately |
| **lsc** | 0.2555 | 0.1022 | **0.1434** | **0.0233** | Weak positive univariate association |
| re | 0.2485 | 0.0994 | -0.0876 | 0.1672 | Multivariate contribution without clear univariate effect |
| diameter | 0.2066 | 0.0826 | -0.1033 | 0.1031 | Weak negative tendency |
| asp | 0.1498 | 0.0599 | -0.0901 | 0.1554 | Weak negative tendency |

This model provides the strongest evidence that speech-graph features help predict a Barratt dimension, but the effect remains small: **R² = 0.0171**.

### Best COG model: Activity 3, W10

| Feature | Mean absolute contribution | Share within model | Univariate r | p | Reading |
|---|---:|---:|---:|---:|---|
| **l1** | **0.0334** | **0.1660** | -0.0585 | 0.3582 | Main model contributor, not independently correlated |
| **lsc** | 0.0283 | 0.1405 | **0.1434** | **0.0237** | Weak positive univariate association |
| **cc** | 0.0260 | 0.1293 | **0.1364** | **0.0315** | Weak positive univariate association |
| diameter | 0.0205 | 0.1020 | -0.1139 | 0.0727 | Weak negative tendency |
| **l3** | 0.0204 | 0.1014 | **0.1350** | **0.0332** | Weak positive univariate association |
| asp | 0.0179 | 0.0889 | -0.1062 | 0.0946 | Weak negative tendency |

The COG signal is weaker than the MOT signal but more distributed across recurrence and connectedness features.

### Activity 1, W20 for MOT

| Feature | Mean absolute contribution | Univariate r | p | Reading |
|---|---:|---:|---:|---|
| **atd** | **0.0998** | **0.2140** | **0.0034** | Strongest univariate association in this model |
| **density** | 0.0910 | **0.2123** | **0.0037** | Higher graph density associated with MOT |
| **nodes** | 0.0815 | **-0.2075** | **0.0046** | More unique nodes associated with lower MOT |
| **lcc** | 0.0815 | **-0.2075** | **0.0046** | Same pattern as nodes |
| **asp** | 0.0524 | **-0.2076** | **0.0046** | Longer paths associated with lower MOT |
| **diameter** | 0.0455 | **-0.1942** | **0.0081** | Larger graph extent associated with lower MOT |

This activity-window combination shows clearer univariate correlations than its multivariate predictive performance. This suggests that the graph metrics contain association, but not enough independent information to generate strong cross-validated prediction.

---

## Correlational structure versus predictive performance

The analysis shows a consistent distinction between **association** and **prediction**.

Some individual graph metrics showed correlations around \(|r| \approx 0.15\) to \(0.23\), especially for MOT and COG. However, the multivariate models remained close to zero in cross-validated \(R^2\). This implies that the features may be partially redundant, collinear, or unstable across folds.

A feature can have a visible univariate correlation but still fail to improve prediction when combined with other graph metrics. Conversely, a feature can contribute multivariately without showing a strong univariate correlation, as observed for some recurrence features in the best MOT model.

---

## Scientific interpretation

The results support the following interpretation:

| Claim | Supported? | Evidence |
|---|---|---|
| Speech-graph NLP features predict Barratt TOTAL | Weakly / practically no | Best R² = 0.0018 |
| Speech-graph NLP features predict NPLAN | **No** | Best R² was negative |
| Speech-graph NLP features predict MOT | **Weakly yes** | Best R² = **0.0171**, best Pearson r = **0.1357** |
| Speech-graph NLP features predict COG | **Weakly yes** | Best R² = **0.0123** |
| Effects are activity-dependent | **Yes** | Best models occur in specific activity-window schemes |
| Effects are window-dependent | **Yes** | Different targets peak at W10, W20, or W30 |
| Results are ready for strong clinical prediction claims | **No** | R² values are very small and many models are negative |

The strongest evidence points to **motor impulsivity (MOT)** and **cognitive impulsivity (COG)**. The graph properties most often involved are recurrence, repeated transitions, clustering, strongly connected components, and graph path structure. These may reflect differences in how speech revisits lexical states, forms local cycles, and organizes transitions over short discourse windows.

However, the magnitude of prediction is small. The best model explains only about **1.71%** of out-of-sample variance, and most models perform worse than the mean baseline. Therefore, these results should be described as **exploratory evidence of weak behavioral-language signal**, not as a clinically deployable prediction model.

---

## Methodological cautions

| Issue | Consequence | Recommendation |
|---|---|---|
| Many models tested | Higher chance of false positives | Apply FDR correction or permutation testing |
| Small R² values | Limited predictive utility | Report as weak signal, not strong prediction |
| Collinearity among graph metrics | Feature importance may be unstable | Consider feature reduction or grouped metrics |
| Activity 5 W30 has n = 37 | Underpowered estimate | Treat as exploratory only |
| Some univariate correlations exceed model performance | Association does not imply prediction | Emphasize cross-validated R² over p-values |
| No external validation sample | Generalizability unknown | Validate on independent data or repeated nested CV |

---

## Final conclusion

The final deduplicated analysis indicates that speech-graph NLP features contain **detectable but weak** information about selected Barratt dimensions. The clearest signal was observed for **MOT**, especially in **Activity 7 with a 30-word window**. A smaller signal was observed for **COG**, especially in **Activity 3 with a 10-word window**. **TOTAL** showed negligible predictive improvement, and **NPLAN** showed no evidence of predictive improvement.

The most relevant NLP features were not simple length measures, but graph-structural properties: **clustering coefficient**, **parallel edges**, **largest strongly connected component**, **recurrence metrics**, **repeated edges**, **diameter**, and **average shortest path**. These findings suggest that the organization and recurrence structure of speech may carry limited behavioral information related to impulsivity, but the current predictive strength is too small for strong individual-level or clinical claims.

The results should therefore be reported as an exploratory speech-graph analysis showing **weak task- and window-specific associations with MOT and COG**, requiring permutation testing, correction for multiple comparisons, and external validation before stronger conclusions are drawn.
