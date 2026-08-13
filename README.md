# Spacecraft Telemetry Anomaly Detection: Leakage-Controlled Evaluation on NASA SMAP and MSL

A window-based anomaly detection pipeline over NASA spacecraft telemetry, comparing supervised ensembles (Random Forest, XGBoost) against an unsupervised baseline (Isolation Forest) under two evaluation protocols.

**Headline result:** the same models score F1 0.51 under a strict channel-level split and F1 0.91 under a random window-level split. The protocol, not the model, accounts for most of the apparent performance. That gap is the finding.

---

## Why this project

Sliding windows extracted from the same telemetry channel overlap and are highly correlated. If train and test windows are drawn from the same channel pool, a model can score near 0.90 F1 while having learned almost nothing about generalising to a channel it has never seen. That is the setting a real spacecraft monitoring system operates in.

This project runs both protocols over one identical pipeline so the difference is isolated cleanly. The strict channel-level split is treated as the primary result. The random window-level split is reported as an upper bound, not as a headline.

---

## Data

NASA SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory) telemetry, introduced by Hundman et al. (2018).

Each channel is a 2D array where rows are time steps and columns are telemetry variables, making this a multivariate rather than univariate problem. Anomalies are annotated as **intervals** rather than point labels, which is what makes window-level formulation natural.

Required files:

```
data/
├── labeled_anomalies.csv    # channel id, spacecraft, anomaly intervals, length
├── train/                   # per-channel .npy telemetry sequences
└── test/                    # per-channel .npy telemetry sequences
```

Only raw telemetry and anomaly metadata are used. Artefacts from the original Telemanom implementation (saved predictions, smoothed errors, model files) are deliberately not used, so the pipeline is independent.

---

## Results

### Primary: strict channel-level split

Development and held-out test channels are fully disjoint. MSL uses 21 development and 6 test channels, SMAP uses 43 development and 11 test channels.

| Dataset | Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| MSL | **Random Forest** | 0.25 | 0.211 | 0.758 | **0.330** | **0.304** | **0.727** |
| MSL | XGBoost | 0.30 | 0.211 | 0.703 | 0.324 | 0.253 | 0.710 |
| MSL | Isolation Forest | 0.05 | 0.141 | 0.983 | 0.246 | 0.116 | 0.394 |
| SMAP | **Random Forest** | 0.45 | 0.383 | 0.783 | **0.514** | **0.641** | **0.858** |
| SMAP | XGBoost | 0.75 | 0.509 | 0.429 | 0.466 | 0.591 | 0.843 |
| SMAP | Isolation Forest | 0.05 | 0.145 | 1.000 | 0.253 | 0.108 | 0.352 |

These numbers are low in absolute terms, and that is the point. Generalising anomaly detection to telemetry channels the model has never seen is genuinely hard, and a protocol that hides the difficulty is not doing its job.

Isolation Forest illustrates why recall alone is misleading. It captures essentially every anomaly on SMAP at recall 1.000, but with precision 0.145 and ROC-AUC 0.352, meaning it ranks anomalous windows **below** normal ones. It flags almost everything rather than detecting anything.

### Secondary: random window-level split

Windows are split randomly with stratification, so windows from the same channel appear in both train and test.

| Dataset | Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| MSL | Random Forest | 0.40 | 0.927 | 0.829 | 0.875 | 0.928 | 0.979 |
| MSL | XGBoost | 0.55 | 0.869 | 0.854 | 0.862 | 0.934 | 0.980 |
| MSL | Isolation Forest | 0.05 | 0.125 | 0.911 | 0.220 | 0.124 | 0.511 |
| SMAP | Random Forest | 0.50 | 0.982 | 0.843 | 0.907 | 0.940 | 0.971 |
| SMAP | XGBoost | 0.65 | 0.967 | 0.849 | 0.904 | 0.938 | 0.977 |
| SMAP | Isolation Forest | 0.05 | 0.134 | 0.878 | 0.232 | 0.113 | 0.453 |

### The gap

| Dataset | Model | Strict F1 | Random F1 | Inflation |
|---|---|---:|---:|---:|
| MSL | Random Forest | 0.330 | 0.875 | **+0.545** |
| MSL | XGBoost | 0.324 | 0.862 | +0.538 |
| SMAP | Random Forest | 0.514 | 0.907 | **+0.393** |
| SMAP | XGBoost | 0.466 | 0.904 | +0.438 |

Same pipeline, same features, same models, same data. Only the split changed.

Two things follow. The engineered features are clearly informative enough to separate anomalous from normal windows, since the supervised models reach 0.90 when the distribution is familiar. And the real bottleneck is **cross-channel generalisation**, not window discrimination. A paper reporting only the random-split number would describe a model that does not exist in operational conditions.

Isolation Forest is the control that confirms this. Its scores barely move between protocols, because its weakness is selectivity in this feature space rather than the difficulty of the split.

---

## Method

### Preprocessing

Normalisation is fitted per channel on the training sequence only, then applied to the paired test sequence. Channels differ substantially in scale and distribution, so global scaling would both distort the data and leak test information into preprocessing.

### Windowing and labelling

Sliding windows of **30 time steps with stride 5**, preserving multivariate structure. A window index table retains each window's start and end position in the original sequence.

Labels use an **any-overlap rule**: a test window touching any part of a labelled anomaly interval is positive. All anomaly subtypes collapse into a single class, keeping the task detection rather than subtype classification.

### Features

Twelve statistics per variable, concatenated across variables into one flat vector. A window with D variables yields 12D features.

- **Distributional:** mean, standard deviation, variance, min, max, median
- **Temporal change:** linear slope, mean absolute successive difference, standard deviation of first differences
- **Shape and energy:** RMS, skewness, kurtosis

Interpretable statistics were chosen over learned embeddings deliberately, so that model behaviour stays inspectable and the same representation feeds all three models. Any performance difference reflects the learning strategy, not the input format.

### Models

Random Forest, XGBoost, and Isolation Forest, all consuming the identical feature matrix. Isolation Forest fits only on training-window features and never sees labels.

---

## Evaluation design

This is where most of the work went.

**Channel-level splitting, 80/20 within each spacecraft.** Held-out channels are touched only once, at final evaluation.

**4-fold grouped cross-validation** with channel identifier as the grouping variable. Overlapping windows from one channel are strongly correlated, so a random CV split would inflate validation performance and corrupt model selection.

**Thresholds tuned on non-test data only.** A grid from 0.05 to 0.95 in steps of 0.05, selected by F1 on development data (strict setting) or validation data (random setting), then applied once. A fixed 0.5 threshold is indefensible here, since Isolation Forest's optimal point sits at 0.05 on every run while XGBoost ranges from 0.30 to 0.75 depending on dataset and split.

**PR-AUC alongside ROC-AUC.** Anomalous windows are a small minority, and precision-recall is more informative than ROC under that imbalance. Confusion matrix counts are retained to make the precision and recall trade-off explicit.

---

## What I would take from this

**Report the protocol before the score.** An F1 of 0.91 and an F1 of 0.51 came out of the same code on the same day. A number without its split design is close to meaningless.

**Supervised ensembles beat the unsupervised baseline under both settings**, and the ranking held across datasets and protocols. Random Forest was the most consistent, with XGBoost second everywhere.

**High recall is not detection.** Isolation Forest caught every SMAP anomaly and was still useless, because it flagged nearly everything and ranked anomalies below normal windows.

**SMAP is the easier benchmark.** Both supervised models scored substantially higher on SMAP than MSL under both protocols, so results reported on one should not be assumed to transfer.

---

## Repository structure

```
.
├── src/
│   ├── preprocessing.py        # per-channel scaling, interval parsing
│   ├── windowing.py            # sliding windows, overlap labelling
│   ├── features.py             # 12 statistics per variable
│   ├── splits.py               # channel-level and random splits
│   └── evaluation.py           # metrics, threshold sweep
├── notebooks/
│   ├── 01_main_channel_level.ipynb   # primary strict experiments
│   ├── 02_smote_experiments.ipynb    # exploratory, see notes
│   └── 03_random_window_level.ipynb  # secondary optimistic split
├── data/
├── results/                    # metric tables, threshold sweeps, figures
├── requirements.txt
└── README.md
```

## Reproducing

```bash
git clone https://github.com/farhanshahriar30/nasa-spacecraft-telemetry-anomaly-detection
cd nasa-spacecraft-telemetry-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the NASA benchmark files in `data/` as shown above, then run the notebooks in order. Run them from the `notebooks/` folder, since they add the project root to `sys.path` for `src/` imports.

Everything runs on CPU. No GPU required.

---

## Notes

- **The SMOTE notebook is exploratory.** Oversampling was tested as a route to the class imbalance problem and is not part of the reported results. Under channel-level splitting the difficulty is cross-channel transfer rather than class balance, so synthetic minority oversampling does not address the actual bottleneck.
- Regenerating channel splits with different settings will shift the numbers. The reported results use the splits produced by `splits.py` as committed.
- Differences between notebooks usually trace back to split configuration or threshold settings rather than to modelling changes.

---

## Limitations and next steps

- **Cross-channel generalisation is unsolved here**, only measured. Methods explicitly targeting unseen-channel robustness are the obvious next direction.
- **No cross-spacecraft transfer.** Training on SMAP and testing on MSL would test whether detection logic generalises across platforms, not just channels.
- **No feature attribution.** SHAP over the engineered features would show which statistics drive predictions and whether the informative ones differ between spacecraft.
- **Statistical features only.** No frequency-domain descriptors, and no comparison against sequence models such as LSTM autoencoders or the original Telemanom approach.
- **Hyperparameters were not exhaustively tuned.** The emphasis was a fair comparison under one protocol rather than maximising any single model.

---

## References

- Hundman et al. (2018), Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding, introduces the SMAP and MSL benchmark
- Kim et al. (2022), Towards a Rigorous Evaluation of Time-Series Anomaly Detection
- Saito and Rehmsmeier (2015), precision-recall over ROC under imbalance
- Tatbul et al. (2018), Precision and Recall for Time Series
- Schmidl, Wenig and Papenbrock (2022), Anomaly Detection in Time Series: A Comprehensive Evaluation
- Breiman (2001), Random Forests
- Chen and Guestrin (2016), XGBoost

Full reference list is in the accompanying paper.
