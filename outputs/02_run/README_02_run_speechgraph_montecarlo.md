# README — Monte Carlo Speech-Graph Models, School-Year Control, and Barratt Dimensions

## Analytical objective

This second analysis evaluated whether **speech-graph NLP features** extracted from structured interview transcripts provided stable and incremental information for predicting four Barratt impulsivity dimensions: **TOTAL**, **NPLAN**, **MOT**, and **COG**.

The first run suggested weak and localized predictive signal for **MOT** and **COG**. The present run tested the robustness of that finding by replacing a single 5-fold cross-validation split with **Monte Carlo cross-validation using 400 random train/test partitions**, by adding **School year** as a control predictor, by comparing different model sets, and by estimating **partial correlations adjusted for School year**.

The central question was therefore not only:

> Do speech-graph features predict Barratt scores?

but more specifically:

> Do speech-graph features add predictive information beyond School year, and are their associations with Barratt dimensions stable across repeated train/test partitions?

The main finding is precise: **the apparent NLP signal observed in the first run did not remain stable under Monte Carlo validation**. The clearest positive predictive performance in the second run came from **School year**, especially for **COG**. Speech-graph features still showed weak exploratory associations with Barratt dimensions, particularly in selected activity-window schemes, but they did **not** provide robust incremental predictive value beyond School year.

---

## Pipeline logic

The linguistic representation used in this run was the same as in the previous analysis. Each transcript was treated as ordered behavioral language data. Only the participant's speech was retained. Interviewer speech and irrelevant transcription labels were excluded. Each transcript was segmented into seven activities, cleaned, lowercased, and transformed into directed lexical-transition graphs.

For example, a cleaned sequence such as:

```text
había una vez un pollo
```

was converted into the directed transitions:

```text
había → una → vez → un → pollo
```

Thus, each activity became a directed speech graph where nodes represent lexical items and edges represent observed word-to-word transitions. This representation preserves recurrence, local connectedness, short cycles, graph compactness, and path structure rather than treating the transcript as a bag of independent words.

Graph metrics were extracted using sliding windows of **10**, **20**, and **30** words with step size **1**:

$$
W_{10}: (t_1,\ldots,t_{10}), (t_2,\ldots,t_{11}), (t_3,\ldots,t_{12}), \ldots
$$

The window-level metrics were aggregated to the **subject × activity × window-size** level. Therefore, overlapping windows were **not** treated as independent observations in the regression models.

The modeling unit remained:

$$
X_{s,a,w} \rightarrow y_{s,k}
$$

where $s$ is the subject, $a$ is the activity, $w \in \{10,20,30\}$ is the window size, and $k$ is one Barratt target among **TOTAL**, **NPLAN**, **MOT**, and **COG**.

---

## Methodological extension in Run 02

Run 02 extended the first analysis in three ways.

First, the validation strategy was changed from a single 5-fold cross-validation design to **Monte Carlo cross-validation** with 400 random partitions and an 80/20 train/test split. This produced a distribution of model performance rather than a single cross-validated estimate.

Second, three predictor sets were evaluated:

| Model set | Predictors | Number of features |
|---|---|---:|
| `nlp_only` | SpeechGraph features only | 13 |
| `school_year_only` | School year only | 1 |
| `school_year_plus_nlp` | School year + SpeechGraph features | 14 |

Third, the analysis added **partial Spearman correlations** between SpeechGraph metrics and Barratt dimensions while controlling for **School year**. This was done to distinguish raw speech-graph associations from associations that may be partly explained by grade-related differences in language development or impulsivity scores.

The executed command was conceptually equivalent to:

```text
py -m src.pipeline.run_montecarlo_cv \
  --input-csv outputs/01_run/analysis/activity_window_features.csv \
  --output-dir outputs/02_run \
  --targets TOTAL,NPLAN,MOT,COG \
  --control-cols "School year" \
  --model-sets nlp,school_year,school_year_nlp \
  --n-repeats 400 \
  --test-size 0.2 \
  --alphas 0.1,1,10,100,1000 \
  --random-state 42 \
  --skip-plots
```

Plots were later generated with:

```text
py -m src.visualization.plotting --run-dir outputs/02_run
```

---

## Data integrity and model grid

The input table for this run was the deduplicated activity-window feature table created in Run 01. Each subject appeared at most once for each activity and window size.

| Check | Final result | Interpretation |
|---|---:|---|
| Activity-window feature rows | **4,534** | Same corrected modeling table used after Run 01 deduplication |
| Duplicate `code × activity × window` rows | **0** | The correction from Run 01 remained valid |
| Maximum unique subjects in a scheme | **252** | Consistent with available Barratt metadata |
| Number of activities | **7** | Activity-specific modeling was preserved |
| Window sizes | **10, 20, 30** | All planned temporal scales were evaluated |
| Targets | **TOTAL, NPLAN, MOT, COG** | Four Barratt dimensions modeled separately |
| Model sets | **3** | `nlp_only`, `school_year_only`, `school_year_plus_nlp` |
| Monte Carlo repetitions | **400** | Repeated train/test evaluation |
| Test proportion | **0.20** | Approximately 20% of available subjects per split |

The full model grid was:

$$
4\;\text{targets} \times 7\;\text{activities} \times 3\;\text{window sizes} \times 3\;\text{model sets} = 252\;\text{model summaries}.
$$

The analysis generated the following major result tables:

| Output table | Rows | Interpretation |
|---|---:|---|
| `models/mc_cv_summary.csv` | **252** | Monte Carlo performance summaries for all target × activity × window × model-set combinations |
| `models/mc_model_comparisons.csv` | **252** | Pairwise model-set comparisons across repeated splits |
| `analysis/partial_correlations_by_activity_window_min_n100.csv` | **1,200** | Partial and simple Spearman correlations, restricted to activity-window schemes with at least 100 subjects |
| `shap/shap_summary.csv` | **2,352** | Exact linear SHAP-style contribution summaries for linear models across repeats |

Subject counts still varied across activities and windows because shorter responses cannot support larger windows. Activity 5 with W30 remained especially underpowered and should be interpreted as exploratory only.

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

For each target, activity, window size, and model set, Ridge regression was fitted repeatedly across Monte Carlo train/test splits.

The general model was:

$$
y_{s,k} = \beta_0 + X_{s,a,w}\beta + \varepsilon_s
$$

with Ridge regularization:

$$
\hat{\beta} = \arg\min_{\beta}\left[\sum_s (y_s - \beta_0 - X_s\beta)^2 + \alpha\lVert\beta\rVert_2^2\right]
$$

where $\alpha \in \{0.1,1,10,100,1000\}$ was selected within the model-fitting pipeline.

For each Monte Carlo repetition, the model was evaluated on a held-out test set using:

$$
R^2 = 1 - \frac{\sum_{i \in test}(y_i - \hat{y}_i)^2}{\sum_{i \in test}(y_i - \bar{y}_{train})^2}
$$

where $\bar{y}_{train}$ is the target mean in the training set. This definition compares the model against a mean-prediction baseline estimated only from the training data.

Interpretation of test-set $R^2$:

| R² behavior | Meaning |
|---|---|
| **R² > 0** | The model improves over the train-mean baseline on held-out data |
| **R² ≈ 0** | The model adds negligible predictive value |
| **R² < 0** | The model performs worse than the train-mean baseline |

The analysis summarized the full Monte Carlo distribution using:

$$
\overline{R^2},\; median(R^2),\; SD(R^2),\; P_{2.5}(R^2),\; P_{97.5}(R^2),\; Pr(R^2 > 0)
$$

and corresponding error metrics:

$$
\Delta RMSE = RMSE_{baseline} - RMSE_{model}
$$

$$
\Delta MAE = MAE_{baseline} - MAE_{model}
$$

Positive values of $\Delta RMSE$ or $\Delta MAE$ indicate lower model error than the baseline.

---

## Model-set comparisons

Three comparisons were used to evaluate whether SpeechGraph and School year contributed different kinds of information.

| Comparison | Definition | Scientific interpretation |
|---|---|---|
| `nlp_vs_school_year` | $R^2_{nlp} - R^2_{school}$ | Whether SpeechGraph alone outperformed School year alone |
| `added_nlp_over_school_year` | $R^2_{school+nlp} - R^2_{school}$ | Whether SpeechGraph added incremental information beyond School year |
| `added_school_year_over_nlp` | $R^2_{school+nlp} - R^2_{nlp}$ | Whether School year added incremental information beyond SpeechGraph |

The second comparison was the most important for evaluating the independent predictive contribution of SpeechGraph:

$$
\Delta R^2_{NLP|School} = R^2_{School+NLP} - R^2_{School}
$$

A positive value indicates that adding SpeechGraph features improved performance relative to a School-year-only model. However, because performance was estimated over repeated splits, the mean $\Delta R^2$ was interpreted together with the median, empirical 2.5th and 97.5th percentiles, and the proportion of splits with $\Delta R^2 > 0$.

---

## Monte Carlo model performance

The strongest positive mean $R^2$ values were concentrated in **COG** models using **School year**, not in SpeechGraph-only models.

### Best models by mean R²

| Rank | Target | Activity | Window | Model set | Features | Mean R² | Median R² | 2.5% R² | 97.5% R² | Prop. R² > 0 | Interpretation |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **COG** | 5 | 10 | `school_year_only` | 1 | **0.0231** | 0.0320 | -0.1371 | 0.1168 | 0.6875 | Best overall mean performance; School-year-driven |
| 2 | **COG** | 3 | 10 | `school_year_only` | 1 | **0.0197** | 0.0332 | -0.1271 | 0.1098 | 0.6700 | Positive but small |
| 3 | **COG** | 1 | 10 | `school_year_only` | 1 | **0.0185** | 0.0296 | -0.1462 | 0.1070 | 0.6500 | Positive but small |
| 4 | **COG** | 3 | 10 | `school_year_plus_nlp` | 14 | **0.0179** | 0.0293 | -0.1136 | 0.1128 | 0.6650 | Combined model, but not better than the best School-year-only model |
| 5 | **COG** | 7 | 10/20/30 | `school_year_only` | 1 | **0.0171** | 0.0320 | -0.1665 | 0.1087 | 0.6550 | Same sample across windows; same School-year-only result |
| 6 | **COG** | 4/6 | 10/20/30 | `school_year_only` | 1 | **0.0156** | 0.0317 | -0.1596 | 0.1090 | 0.6650 | Same retained sample across windows |

Although several COG models had positive mean $R^2$, their empirical intervals still included negative values. Therefore, these results support only **weak predictive signal**, mainly linked to School year.

### Best model by target and model set

| Target | Model set | Best activity-window | Best mean R² | Interpretation |
|---|---|---|---:|---|
| **COG** | `school_year_only` | A5-W10 | **0.0231** | Best overall model; School-year effect |
| **COG** | `school_year_plus_nlp` | A3-W10 | **0.0179** | Positive, but not clearly superior to School year alone |
| **COG** | `nlp_only` | A3-W10 | **-0.0237** | SpeechGraph-only model did not generalize |
| TOTAL | `school_year_plus_nlp` | A2-W20 | -0.0141 | Negative mean R² |
| TOTAL | `nlp_only` | A2-W20 | -0.0143 | Negative mean R² |
| MOT | `nlp_only` | A2-W20 | -0.0214 | Negative mean R²; Run 01 MOT signal did not stabilize |
| NPLAN | `school_year_only` | A1-W10 | -0.0256 | No useful predictive signal |
| NPLAN | `nlp_only` | A6-W30 | -0.0281 | No useful predictive signal |

The most important change relative to Run 01 is that the previously promising **MOT SpeechGraph signal** did not remain positive when performance was averaged across 400 Monte Carlo partitions.

---

## Incremental value of SpeechGraph beyond School year

The primary incremental question was whether adding SpeechGraph features improved a School-year-only model.

### Top `added_nlp_over_school_year` comparisons

| Rank | Target | Activity | Window | Mean ΔR² | Median ΔR² | 2.5% ΔR² | 97.5% ΔR² | Prop. ΔR² > 0 | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | TOTAL | 2 | 20 | **0.0156** | 0.0199 | -0.0795 | 0.0755 | 0.7250 | Small possible NLP gain, but interval crosses zero |
| 2 | NPLAN | 5 | 30 | 0.0118 | 0.0037 | -0.3729 | 0.4462 | 0.5775 | Underpowered and unstable; exploratory only |
| 3 | COG | 5 | 30 | 0.0087 | -0.0050 | -0.3607 | 0.4573 | 0.4275 | Underpowered and unstable; exploratory only |
| 4 | MOT | 1 | 20 | 0.0064 | 0.0126 | -0.0580 | 0.0405 | 0.7275 | Small candidate signal, not robust |
| 5 | TOTAL | 1 | 10 | 0.0030 | 0.0064 | -0.0444 | 0.0374 | 0.6225 | Negligible |

The incremental effects of SpeechGraph over School year were numerically small and their empirical intervals crossed zero. Therefore, this run did **not** provide strong evidence that SpeechGraph features add stable predictive value beyond School year.

---

## Incremental value of School year beyond SpeechGraph

The reverse comparison evaluated whether adding School year improved an NLP-only model.

### Top `added_school_year_over_nlp` comparisons

| Rank | Target | Activity | Window | Mean ΔR² | Median ΔR² | 2.5% ΔR² | 97.5% ΔR² | Prop. ΔR² > 0 | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | MOT | 5 | 30 | **0.1006** | 0.0005 | -0.2340 | 1.0802 | 0.5750 | Very unstable; underpowered A5-W30 scheme |
| 2 | TOTAL | 5 | 30 | 0.0538 | -0.0005 | -0.0563 | 0.5556 | 0.3100 | Very unstable; underpowered A5-W30 scheme |
| 3 | **COG** | 1 | 10 | **0.0438** | 0.0540 | -0.1050 | 0.1150 | 0.8400 | Consistent advantage of adding School year |
| 4 | **COG** | 3 | 10 | **0.0416** | 0.0494 | -0.0431 | 0.1000 | 0.8425 | Consistent advantage of adding School year |
| 5 | **COG** | 5 | 10 | **0.0416** | 0.0499 | -0.1451 | 0.1571 | 0.7825 | School-year contribution, with larger uncertainty |
| 6 | **COG** | 3 | 20 | **0.0380** | 0.0441 | -0.0667 | 0.1024 | 0.8475 | Consistent advantage of adding School year |
| 7 | **COG** | 6 | 20 | **0.0350** | 0.0391 | -0.0658 | 0.1158 | 0.8575 | Consistent advantage of adding School year |
| 8 | **COG** | 6 | 30 | **0.0336** | 0.0394 | -0.0491 | 0.0889 | 0.8550 | Consistent advantage of adding School year |

This comparison showed a clearer pattern than the reverse one: **School year improved several NLP-only models, especially for COG**. This indicates that some of the variance relevant to COG is more strongly linked to school grade than to SpeechGraph metrics.

---

## Interpretation of the COG pattern

The COG results were the clearest positive finding of Run 02, but their interpretation differs from the first run.

In Run 01, the best COG model was an NLP-only model using Activity 3, W10. In Run 02, the best COG results were primarily obtained by `school_year_only` models, and the combined `school_year_plus_nlp` models did not consistently improve over School year alone.

This suggests the following interpretation:

| Observation | Interpretation |
|---|---|
| COG had the strongest positive mean R² values | COG is the Barratt dimension most amenable to weak prediction in this dataset |
| Best COG models were `school_year_only` | Grade-related variation explains part of the COG signal |
| `school_year_plus_nlp` was not consistently better than `school_year_only` | SpeechGraph did not add robust independent predictive value |
| COG `nlp_only` remained negative on average | The apparent NLP signal did not generalize across Monte Carlo splits |

Thus, COG should be described as showing **weak grade-related predictive signal**, not as showing robust SpeechGraph-driven prediction.

---

## Interpretation of the MOT pattern

In Run 01, the strongest positive model was **MOT, Activity 7, W30**, with a small positive $R^2$. In Run 02, this model did not remain positive on average under Monte Carlo validation.

The best MOT `nlp_only` result in Run 02 was still negative:

| Target | Model set | Best activity-window | Mean R² | Interpretation |
|---|---|---|---:|---|
| MOT | `nlp_only` | A2-W20 | -0.0214 | Best MOT SpeechGraph-only model was still below baseline |
| MOT | `school_year_plus_nlp` | A2-W20 | -0.0274 | Combined model did not improve enough to become positive |
| MOT | `school_year_only` | A1-W10 | -0.0266 | School year alone did not predict MOT |

Therefore, the MOT finding from Run 01 should be interpreted as **partition-sensitive**. SpeechGraph metrics may still be associated with MOT in specific activities, but they did not produce stable out-of-sample prediction across repeated train/test partitions.

---

## Partial correlations adjusted by School year

Partial Spearman correlations were computed to evaluate whether individual SpeechGraph metrics were associated with Barratt dimensions after adjusting for School year.

The partial correlation was computed by residualizing both the SpeechGraph metric and the Barratt target with respect to the control variable and then correlating the residuals:

$$
r_{XY\cdot Z} = cor\left(X - \hat{X}(Z),\;Y - \hat{Y}(Z)\right)
$$

where:

- $X$ is a SpeechGraph metric,
- $Y$ is a Barratt target,
- $Z$ is School year.

Both simple and partial Spearman correlations were reported. False-discovery-rate correction was applied globally and by target.

### Strongest partial correlations

| Rank | Target | Activity-window | Metric | Partial r | Partial p | Global q | Target-level q | Interpretation |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | MOT | A1-W30 | `lsc` | -0.2438 | 0.0110 | 0.5512 | 0.3478 | Largest absolute partial correlation; exploratory only |
| 2 | TOTAL | A1-W30 | `lsc` | -0.2385 | 0.0129 | 0.5512 | 0.3698 | Similar pattern for TOTAL |
| 3 | MOT | A5-W20 | `lsc` | -0.2367 | 0.0124 | 0.5512 | 0.3478 | Activity 5 W20 signal; not FDR-significant |
| 4 | TOTAL | A1-W20 | `asp` | -0.2346 | 0.0013 | 0.2264 | 0.1038 | More compact path structure associated with higher TOTAL |
| 5 | MOT | A1-W20 | `atd` | 0.2337 | 0.0014 | 0.2264 | 0.1156 | Higher average total degree associated with MOT |
| 6 | MOT | A1-W20 | `density` | 0.2331 | 0.0015 | 0.2264 | 0.1156 | Higher graph density associated with MOT |
| 7 | TOTAL | A1-W20 | `density` | 0.2290 | 0.0018 | 0.2264 | 0.1038 | Similar direction for TOTAL |
| 8 | MOT | A1-W20 | `nodes` | -0.2289 | 0.0018 | 0.2264 | 0.1156 | More unique nodes associated with lower MOT |
| 9 | MOT | A1-W20 | `lcc` | -0.2289 | 0.0018 | 0.2264 | 0.1156 | Mirrors the node-count pattern |
| 10 | TOTAL | A1-W20 | `atd` | 0.2284 | 0.0018 | 0.2264 | 0.1038 | Similar graph-compactness pattern |

The main partial-correlation pattern was concentrated in **Activity 1, W20**, especially for MOT and TOTAL. Higher **density** and **average total degree** tended to be associated with higher impulsivity scores, whereas larger node/component/path metrics such as **nodes**, **LCC**, **diameter**, and **ASP** tended to be associated with lower impulsivity scores.

However, none of the strongest associations survived stringent global FDR correction at conventional levels. Therefore, these correlations should be interpreted as **exploratory associations**, not confirmatory evidence.

---

## Partial correlations by Barratt dimension

### TOTAL

TOTAL showed its strongest adjusted associations in Activity 1, especially W20 and W30. The most prominent metrics were:

| Direction | Metrics | Interpretation |
|---|---|---|
| Positive | `density`, `atd`, `re` | More compact or recurrent graphs tended to relate to higher TOTAL |
| Negative | `asp`, `diameter`, `nodes`, `lcc`, `lsc` | Larger or longer-path graphs tended to relate to lower TOTAL |

Although several nominal p-values were below 0.01, the corrected q-values did not support a confirmatory result.

### MOT

MOT showed the most coherent partial-correlation pattern. Activity 1 W20 was again central:

| Metric | Partial r | Interpretation |
|---|---:|---|
| `atd` | 0.2337 | Higher degree/transition compactness associated with higher MOT |
| `density` | 0.2331 | Denser lexical-transition graph associated with higher MOT |
| `nodes` | -0.2289 | More unique lexical states associated with lower MOT |
| `lcc` | -0.2289 | Larger connected component associated with lower MOT |
| `asp` | -0.2240 | Longer paths associated with lower MOT |
| `diameter` | -0.2080 | Larger graph extent associated with lower MOT |

This pattern is scientifically interpretable, but it did not translate into stable out-of-sample prediction under Monte Carlo CV.

### COG

COG showed a different pattern, mainly involving Activity 3 and some Activity 6 effects:

| Activity-window | Metric | Partial r | Interpretation |
|---|---|---:|---|
| A3-W20 | `re` | 0.1983 | Repeated edges associated with higher COG |
| A3-W30 | `re` | 0.1842 | Similar recurrence pattern |
| A3-W10 | `lsc` | 0.1580 | Strongly connected structure associated with COG |
| A1-W20 | `asp` | -0.1799 | Longer path structure associated with lower COG |
| A6-W20 | `l3` | -0.1660 | Three-step loops inversely associated with COG |
| A6-W30 | `l3` | -0.1631 | Similar direction at W30 |

These effects are compatible with weak language-organization associations, but the strongest predictive COG signal was still better explained by School year.

### NPLAN

NPLAN showed no convincing predictive signal and weaker, more diffuse partial correlations. The strongest adjusted associations involved Activity 1 W20/W30 and Activity 5 W20, but no coherent predictive pattern emerged.

---

## Exact linear SHAP interpretation

Feature interpretation was summarized using exact linear SHAP-style contributions for Ridge models. For a linear model, the contribution of feature $j$ can be understood as approximately:

$$
\phi_j \approx (x_j - E[x_j])\beta_j
$$

or, after standardization, as a product between the standardized feature value and the learned coefficient. The reported value was the mean absolute contribution across observations and Monte Carlo repetitions:

$$
Mean\;|SHAP_j| = \frac{1}{N}\sum_i |\phi_{ij}|
$$

These values quantify how much each variable contributed to the fitted linear model. They do **not** imply causality, and they should not be interpreted as substantive evidence when the corresponding model has negative or near-zero out-of-sample performance.

### COG SHAP pattern

For the strongest combined COG scheme, **COG, Activity 3, W10, `school_year_plus_nlp`**, School year was the dominant feature:

| Feature | Mean |SHAP| | Interpretation |
|---|---:|---|
| `School year` | **0.2844** | Dominant contribution in the combined model |
| `l1` | 0.0974 | Main NLP contribution |
| `cc` | 0.0895 | Clustering contribution |
| `lsc` | 0.0894 | Strongly connected structure |
| `re` | 0.0580 | Repeated edges |
| `diameter` | 0.0486 | Graph path extent |

For **COG, Activity 3, W10, `nlp_only`**, the main NLP features were:

| Feature | Mean |SHAP| | Interpretation |
|---|---:|---|
| `l1` | 0.0516 | One-step recurrence/cycle structure |
| `cc` | 0.0395 | Clustering |
| `lsc` | 0.0393 | Strongly connected component structure |
| `l3` | 0.0260 | Three-step cycles |
| `diameter` | 0.0259 | Graph extent |
| `re` | 0.0239 | Repeated edges |

The combined COG model demonstrates that School year contributed more strongly than individual SpeechGraph features.

### MOT SHAP pattern

For **MOT, Activity 2, W20, `nlp_only`**, the largest contributions were:

| Feature | Mean |SHAP| | Interpretation |
|---|---:|---|
| `pe` | 0.4960 | Parallel/repeated transition structure |
| `cc` | 0.3205 | Clustering |
| `l2` | 0.2751 | Two-step cycles |
| `lsc` | 0.2418 | Strongly connected structure |
| `diameter` | 0.1763 | Graph path extent |
| `re` | 0.1438 | Repeated edges |

However, this model had negative mean $R^2$. Therefore, these SHAP values describe model-internal weighting but do not establish robust predictive relevance.

### TOTAL SHAP pattern

For **TOTAL, Activity 2, W20, `school_year_plus_nlp`**, the largest contributions were:

| Feature | Mean |SHAP| | Interpretation |
|---|---:|---|
| `pe` | 0.7873 | Largest model contribution |
| `School year` | 0.4752 | Strong metadata contribution |
| `lsc` | 0.4737 | Strongly connected structure |
| `l2` | 0.4275 | Two-step cycles |
| `cc` | 0.4181 | Clustering |
| `re` | 0.2633 | Repeated edges |

Because the corresponding mean $R^2$ remained negative, these contributions should be interpreted cautiously.

---

## Figures generated by Run 02

The following figures summarize the main Run 02 outputs. Paths are relative to `outputs/02_run/`.

### Model-performance figures

| Figure | Purpose |
|---|---|
| `figures/models/mc_best_mean_r2_by_target_model_set.png` | Best Monte Carlo mean R² by target and model set |
| `figures/models/mc_model_comparison_nlp_vs_school_year.png` | Direct comparison between NLP-only and School-year-only models |
| `figures/models/mc_model_comparison_added_nlp_over_school_year.png` | Incremental value of NLP beyond School year |
| `figures/models/mc_model_comparison_added_school_year_over_nlp.png` | Incremental value of School year beyond NLP |
| `figures/models/mc_mean_r2_heatmap_COG_school_year_only.png` | COG performance map for School-year-only models |
| `figures/models/mc_mean_r2_heatmap_COG_school_year_plus_nlp.png` | COG performance map for combined models |
| `figures/models/mc_mean_r2_heatmap_MOT_nlp_only.png` | MOT performance map for SpeechGraph-only models |

### Partial-correlation figures

| Figure | Purpose |
|---|---|
| `figures/analysis/top_partial_correlations_TOTAL.png` | Top School-year-adjusted SpeechGraph associations with TOTAL |
| `figures/analysis/top_partial_correlations_COG.png` | Top School-year-adjusted SpeechGraph associations with COG |
| `figures/analysis/top_partial_correlations_MOT.png` | Top School-year-adjusted SpeechGraph associations with MOT |
| `figures/analysis/top_partial_correlations_NPLAN.png` | Top School-year-adjusted SpeechGraph associations with NPLAN |

### SHAP figures

| Figure | Purpose |
|---|---|
| `figures/shap/shap_top_features_TOTAL_school_year_plus_nlp.png` | Linear SHAP contributions for the best TOTAL combined scheme |
| `figures/shap/shap_top_features_COG_nlp_only.png` | Linear SHAP contributions for the best COG NLP-only scheme |
| `figures/shap/shap_top_features_COG_school_year_plus_nlp.png` | Linear SHAP contributions for the best COG combined scheme |
| `figures/shap/shap_top_features_MOT_nlp_only.png` | Linear SHAP contributions for the best MOT NLP-only scheme |
| `figures/shap/shap_top_features_NPLAN_nlp_only.png` | Linear SHAP contributions for the best NPLAN NLP-only scheme |
| `figures/shap/shap_top_features_NPLAN_school_year_plus_nlp.png` | Linear SHAP contributions for the best NPLAN combined scheme |

---

## Scientific interpretation

The second run shifts the interpretation of the project.

The first run suggested weak SpeechGraph signal for MOT and COG. The second run showed that this signal was not stable under repeated Monte Carlo train/test resampling. The best positive models in Run 02 were mainly **COG models using School year**, not SpeechGraph-only models.

This implies that part of the previously observed COG association may be confounded or mediated by grade-related variation. School year may capture developmental, educational, or language-exposure differences that are also related to cognitive impulsivity scores.

The partial correlations indicate that SpeechGraph metrics still contain exploratory associations with Barratt dimensions. The most interpretable pattern involved Activity 1 W20, where denser and more compact graphs tended to relate positively to MOT and TOTAL, while larger or longer-path graphs tended to relate negatively. However, these correlations did not survive global FDR correction and did not translate into stable predictive performance.

The SHAP results identify features that influenced the fitted Ridge models, but they should be interpreted in the context of model performance. When a model has negative mean $R^2$, high SHAP values indicate large algebraic contributions within an unstable or non-generalizing model rather than reliable psychological predictors.

---

## Main conclusions

| Claim | Supported by Run 02? | Evidence |
|---|---|---|
| SpeechGraph-only models robustly predict Barratt TOTAL | **No** | Best TOTAL NLP-only model had negative mean R² |
| SpeechGraph-only models robustly predict NPLAN | **No** | NPLAN remained negative and diffuse |
| SpeechGraph-only models robustly predict MOT | **No** | Previously positive MOT signal did not remain positive under Monte Carlo CV |
| SpeechGraph-only models robustly predict COG | **No** | Best COG NLP-only model had negative mean R² |
| School year carries weak predictive signal for COG | **Yes, weakly** | Best models were COG `school_year_only`, with mean R² around 0.015–0.023 |
| SpeechGraph adds robust incremental value beyond School year | **Not supported** | `added_nlp_over_school_year` effects were small and intervals crossed zero |
| School year adds value beyond SpeechGraph for COG | **Weakly supported** | Several COG `added_school_year_over_nlp` comparisons were positive across most splits |
| SpeechGraph metrics show exploratory partial associations with Barratt | **Yes, exploratory** | Partial correlations up to approximately |r| = 0.24, but not FDR-significant globally |
| Results support clinical or individual prediction claims | **No** | Effects were small, unstable, and frequently below baseline |

---

## Methodological cautions

| Issue | Consequence | Recommendation |
|---|---|---|
| Mean R² values are very small | Predictive utility is limited | Report as weak exploratory signal, not practical prediction |
| Many intervals cross zero | Positive mean values may not be stable | Report full Monte Carlo distributions and proportions above zero |
| Activity 5 W30 is underpowered | Large ΔR² values can be unstable | Treat A5-W30 results as exploratory only |
| SpeechGraph features are collinear | SHAP and coefficients may be unstable | Interpret feature families rather than individual metrics |
| COG signal is partly School-year-driven | NLP interpretation may be confounded by grade | Use School year as a control in future analyses |
| Correlations do not survive global FDR | Nominal associations may be false positives | Treat partial correlations as hypothesis-generating |
| Regression framing may be too restrictive | Impulsivity may not be linearly predictable from single activity-window schemes | Consider subject-level multimodal profiling in future analyses |

---

## Recommended next analytical direction

The results do not support continuing with a simple model in which each activity-window SpeechGraph feature set is used to directly predict continuous Barratt scores. A more appropriate next step is to reformulate the analysis as a **multimodal profiling problem**.

Instead of asking whether SpeechGraph directly predicts Barratt dimensions, the next analysis should ask:

> Can demographic, cognitive-linguistic, and SpeechGraph variables jointly identify subject-level profiles that differ in Barratt impulsivity dimensions?

In that framework, Barratt scores should be used as **external validation criteria** rather than as direct labels for constructing the profiles. SpeechGraph would then be evaluated for its **incremental profiling value** over demographic and cognitive metadata.

A valid Run 03 should therefore:

1. Convert activity-window SpeechGraph metrics into a **subject-level feature matrix**.
2. Integrate demographic, school, and cognitive-linguistic metadata.
3. Build profiles without using Barratt scores as clustering inputs.
4. Validate the resulting profiles against TOTAL, NPLAN, MOT, COG, and high/low impulsivity status.
5. Compare metadata-only profiles against metadata + SpeechGraph profiles.
6. Evaluate profile stability using bootstrap or repeated resampling.
7. Report effect sizes, confidence intervals, and correction for multiple testing.

This would align the analysis more closely with the empirical evidence: SpeechGraph contains weak local associations with impulsivity-related dimensions, but it does not currently support robust direct prediction.

---

## Final conclusion

Run 02 provides a more conservative and statistically robust interpretation than Run 01. Under 400 Monte Carlo train/test repetitions, the previously observed SpeechGraph signal did not remain stable as a direct predictor of Barratt scores. The strongest positive predictive pattern was instead associated with **School year**, particularly for **COG**.

SpeechGraph metrics still showed interpretable exploratory associations with impulsivity dimensions, especially involving density, degree, recurrence, connectedness, clustering, and path structure. However, these associations were small, did not survive strict multiple-comparison correction, and did not consistently improve out-of-sample prediction beyond School year.

The results should therefore be reported as follows:

> SpeechGraph features showed weak and localized associations with Barratt impulsivity dimensions, but Monte Carlo validation did not support robust direct prediction or stable incremental value beyond School year. The clearest signal in Run 02 was a small School-year-related effect for COG. These findings motivate a shift from direct prediction toward subject-level multimodal profiling, where SpeechGraph features are evaluated as one component of broader demographic and cognitive-language profiles.

