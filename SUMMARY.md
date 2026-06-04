# Race Strategist Project Summary

## 1. Project Overview

Race Strategist is a machine learning project and Streamlit app for Formula 1 pit strategy prediction.

The goal is simple:

> Given the current lap situation for a driver, predict whether that driver will pit on the next lap.

This is useful because pit stops are one of the most important strategic decisions in Formula 1. A well-timed pit stop can gain track position, protect from an undercut, respond to tyre degradation, or react to race conditions. This project turns historical lap data into a supervised machine learning problem and builds models that estimate pit-stop risk lap by lap.

The project includes:

- A validated F1 strategy dataset.
- Preprocessing code for clean model inputs.
- Multiple classical machine learning models.
- Model evaluation reports and plots.
- A Streamlit app for interactive prediction and exploration.

## 2. What The Project Predicts

The target column is:

```text
WillPitNextLap
```

This is a binary classification target:

| Value | Meaning |
|---:|---|
| `0` | The driver will not pit on the next lap |
| `1` | The driver will pit on the next lap |

Each row represents a lap-level situation for one driver. The model looks at the current state of that lap, then predicts the chance of a pit stop on the next lap.

Example:

```text
Driver is on lap 24, using MEDIUM tyres, tyre life is 18 laps,
position is 5th, lap time is getting slower, race progress is 45%.

Question: will this driver pit on lap 25?
```

The model returns a probability. The app then compares that probability to a decision threshold to classify the lap as either "pit next lap" or "do not pit next lap."

## 3. Dataset

The main dataset is:

```text
data/f1_strategy_dataset_v4_validated.csv
```

Current dataset summary:

| Item | Value |
|---|---:|
| Rows | 101,371 |
| Columns | 18 |
| Seasons | 2022-2025 |
| Races | 28 |
| Drivers | 31 |
| Training rows | 79,283 |
| Test rows | 22,088 |
| Test positive rate | 3.46% |

The positive rate is very low because most laps do not happen immediately before a pit stop. In the held-out test set, only about 3.5% of rows are "pit next lap" examples.

This creates a class imbalance problem:

- The majority class is "no pit next lap."
- The minority class is "pit next lap."
- A naive model can look accurate by predicting "no pit" almost all the time.

Because of this, the project does not focus on plain accuracy. It uses ranking and classification metrics that are better for rare-event prediction, especially ROC-AUC, precision, recall, and F1. PR-AUC is still reported as a supporting metric for the rare positive class.

## 4. Main Features Used By The Model

The model uses current-lap information such as:

| Feature | Meaning |
|---|---|
| `Driver` | Driver name or code |
| `Race` | Race/session name |
| `Year` | Season year |
| `LapNumber` | Current lap number |
| `Stint` | Current tyre stint number |
| `Compound` | Tyre compound, such as SOFT, MEDIUM, HARD, WET |
| `TyreLife` | Number of laps completed on current tyres |
| `Position` | Current race position |
| `LapTime (s)` | Current lap time in seconds |
| `LapTime_Delta` | Lap time change compared with a reference |
| `Cumulative_Degradation` | Accumulated tyre or performance degradation |
| `RaceProgress` | How far the race has progressed |
| `Normalized_TyreLife` | Tyre life adjusted to a normalized scale |
| `Position_Change` | Recent or current position movement |

These features describe the race situation from several angles:

- Tyre condition.
- Race timing.
- Driver and race context.
- Track position.
- Lap pace and degradation.

The model also uses causal temporal features derived from the lap history within each (Driver, Race, Year) group:

| Feature | Meaning |
|---|---|
| `LapTime_lag1`, `LapTime_lag2`, `LapTime_lag3` | Lap times from the previous 1–3 laps |
| `LapTime_Delta_lag1/2/3` | Lap-time deltas from the previous 1–3 laps |
| `LapTime_rolling_mean_3` | Rolling mean lap time over the past 3 laps |
| `LapTime_rolling_std_3` | Rolling standard deviation of lap time over the past 3 laps |
| `LapTime_Delta_rolling_mean_3` | Rolling mean delta over the past 3 laps |
| `LapTime_Delta_rolling_std_3` | Rolling standard deviation of delta over the past 3 laps |
| `Cumulative_Degradation_rolling_mean_3` | Short-term degradation trend |
| `Position_lag1` | Position on the previous lap |
| `Position_rolling_mean_3` | Rolling mean position over the past 3 laps |
| `Position_Change_lag1` | Position change on the previous lap |
| `LapTime_diff_from_rolling_mean3` | Pace slowdown signal: current lap time minus the 3-lap rolling mean |
| `Compound_changed_last_lap` | Flag indicating whether the tyre compound changed since the prior lap |
| `Lap_history_count` | Number of prior laps available (0–3); 0 at the start of each stint |

Each temporal feature is computed strictly from laps with smaller `LapNumber` within the same (Driver, Race, Year) group, so no future information leaks into any row.

## 5. Machine Learning Problem

This project uses supervised learning.

That means the model learns from examples where the answer is already known:

```text
Input features: lap situation
Known answer: did the driver pit on the next lap?
```

After training, the model can receive a new lap situation and estimate the probability of a pit stop on the next lap.

The machine learning task is:

```text
Binary classification with rare positive events.
```

This is different from predicting a continuous number like lap time. The model is not predicting "how fast" or "which lap exactly." It is answering a yes/no question for each lap:

```text
Will this driver pit on the next lap?
```

## 6. Data Leakage Prevention

Data leakage happens when the model is accidentally given information that would not be available at prediction time. Leakage can make model results look much better than they really are.

This project removes known leakage columns before training:

```text
PitNextLap
NextStint
PitStop
```

These columns are too directly connected to the answer. If they were included, the model could cheat by learning from future information or from columns that already reveal the pit event.

The split is also race-grouped. This means:

```text
No race appears in both training and testing.
```

This matters because laps from the same race are strongly related. If some laps from a race were in training and other laps from the same race were in testing, the model could benefit from race-specific patterns it had already seen. A race-grouped split gives a more honest test of whether the model generalizes to unseen races.

The held-out test races are:

- Abu Dhabi Grand Prix
- Chinese Grand Prix
- Dutch Grand Prix
- Hungarian Grand Prix
- Pre-Season Track Session
- Spanish Grand Prix

## 7. Preprocessing Pipeline

The preprocessing code is in:

```text
src/preprocessing.py
```

The pipeline prepares raw data for machine learning.

### Missing Values

Missing `Compound` values are filled with:

```text
UNKNOWN
```

This keeps rows usable while still telling the model that the tyre compound was missing.

### Numeric Features

Numeric features include:

```text
LapNumber
Stint
TyreLife
Position
LapTime (s)
Year
LapTime_Delta
Cumulative_Degradation
RaceProgress
Normalized_TyreLife
Position_Change
```

### Categorical Features

Categorical features include:

```text
Compound
Driver
Race
```

The project uses two preprocessing styles depending on the model type.

### Scaled Preprocessor

Used for Logistic Regression.

It applies:

- `RobustScaler` for numeric features.
- `OneHotEncoder` for `Compound`.
- `TargetEncoder` for high-cardinality categorical features: `Driver` and `Race`.

Logistic Regression is a linear model, so scaling helps keep numeric columns comparable. One-hot encoding makes tyre compound readable to the model without implying a false order like `SOFT < MEDIUM < HARD`.

### Tree Preprocessor

Used for tree-based models.

It applies:

- Numeric passthrough for numeric features.
- `OrdinalEncoder` for categorical features.

Tree models can handle integer-coded categories more naturally than linear models, so the tree pipeline is simpler and faster.

## 8. Models Trained

The training code is in:

```text
src/train.py
```

The project trains seven model types:

| Model | Purpose |
|---|---|
| Majority baseline | Simple reference model that predicts the majority class |
| Logistic Regression | Interpretable linear baseline |
| Decision Tree | Simple tree-based model |
| Random Forest | Ensemble of many decision trees |
| Gradient Boosting | Sequential tree ensemble |
| LightGBM | Fast gradient boosting with native imbalance weighting (`scale_pos_weight`) |
| XGBoost | Histogram-based gradient boosting with `scale_pos_weight` |

The majority baseline is important because it shows what performance looks like when the model does not really learn pit-stop patterns. The real models should perform much better than this baseline.

## 9. Training Method

The training process has three main steps.

### Step 1: Grouped Cross-Validation

The project uses:

```text
GroupKFold(5)
```

The group is:

```text
Race
```

This means each validation fold contains whole races that are not in the training portion of that fold. This keeps evaluation closer to the real challenge: predicting on races the model has not already seen.

### Step 2: Hyperparameter Search

The project uses `GridSearchCV` to try different model settings.

The cross-validation scoring metric used during training is:

```text
average_precision
```

Average precision is the same idea as PR-AUC. It is useful for imbalanced classification because it focuses on how well the model finds the rare positive class. In this summary, the main headline metric is ROC-AUC because it shows how well the model ranks pit-risk laps above no-pit laps across thresholds.

### Step 3: Calibration and Threshold Tuning

Most classifiers output a probability, for example:

```text
Pit next lap probability = 0.72
```

But the final prediction needs to be:

```text
0 or 1
```

So the project chooses a threshold. If the calibrated probability is greater than or equal to the threshold, the model predicts `1`.

Example:

```text
Probability = 0.72
Threshold = 0.488
Prediction = 1
```

Before threshold tuning, the raw out-of-fold probabilities are passed through an isotonic regression calibrator trained on the OOF labels. Calibration aligns the raw probability distribution closer to the true positive rate, which makes threshold selection more reliable.

The threshold is then selected using a recall-targeted policy:

```text
Policy: recall_target_0.60
```

This picks the highest-precision threshold whose OOF recall is at least 0.60. If no threshold on the PR curve reaches 0.60, the fallback selects the threshold that maximizes F-beta with beta=2 (recall-weighted).

This approach is better than optimizing for F1 or using a fixed 0.5 threshold, because rare-event pit prediction benefits from catching more true pit stops even at the cost of some additional false alerts.

## 10. Evaluation Metrics Explained Simply

The evaluation code is in:

```text
src/evaluate.py
```

### ROC-AUC

ROC-AUC measures how well the model ranks positive examples above negative examples.

It answers:

```text
How well does the model rank pit-next-lap cases above no-pit cases?
```

This is the main metric focus in this summary because the project is primarily evaluating whether the model can separate high-risk pit laps from normal laps.

### PR-AUC

PR-AUC means area under the precision-recall curve.

It answers:

```text
How well does the model find rare pit-stop events while avoiding false alarms?
```

PR-AUC is still useful because pit stops are rare, but it is treated here as a supporting diagnostic rather than the headline metric.

### Precision

Precision answers:

```text
When the model predicts "pit next lap," how often is it correct?
```

High precision means fewer false pit alerts.

### Recall

Recall answers:

```text
Out of all real pit-next-lap events, how many did the model catch?
```

Low recall means the model misses many actual pit stops.

### F1

F1 combines precision and recall into one score.

It is useful when both false positives and missed positives matter.

### Brier Score

Brier score measures probability calibration.

It answers:

```text
Are the predicted probabilities close to reality?
```

Lower is better.

## 11. Current Model Results

Current held-out test results are:

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9880 | 0.8571 | 0.7523 | 0.9140 | 0.6392 | 0.0122 | 0.488 |
| Random Forest | 0.9874 | 0.8506 | 0.7028 | 0.9000 | 0.5765 | 0.0119 | 0.561 |
| LightGBM | 0.9877 | 0.8440 | 0.7338 | 0.9109 | 0.6144 | 0.0125 | 0.511 |
| Gradient Boosting | 0.9859 | 0.7859 | 0.5806 | 0.8612 | 0.4379 | 0.0164 | 0.487 |
| Decision Tree | 0.9319 | 0.7504 | 0.7208 | 0.8285 | 0.6379 | 0.0151 | 0.392 |
| Logistic Regression | 0.9156 | 0.3163 | 0.3875 | 0.3649 | 0.4131 | 0.0284 | 0.104 |
| Majority Baseline | n/a | 0.0346 | 0.0000 | 0.0000 | 0.0000 | 0.0346 | 1.100 |

All thresholds use the `recall_target_0.60` policy on isotonic-calibrated OOF probabilities.

The best model by ROC-AUC is:

```text
XGBoost
```

Its test performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9880 |
| PR-AUC | 0.8571 |
| F1 | 0.7523 |
| Precision | 0.9140 |
| Recall | 0.6392 |
| Threshold | 0.488 |

## 12. How To Interpret The Results

The XGBoost model is much better than random guessing and substantially stronger than the earlier Random Forest baseline.

The main result is the XGBoost ROC-AUC:

```text
0.9880
```

This means the model almost always ranks actual pit-next-lap situations above no-pit situations. In other words, the model has learned strong pit-stop signal from both the current-lap state and recent lap history.

As a supporting rare-event metric, the XGBoost PR-AUC is also far above the random PR-AUC floor.

The random PR-AUC floor is approximately the positive rate:

```text
0.0346
```

The XGBoost PR-AUC is:

```text
0.8571
```

This is more than 24 times the random floor, and roughly double the previous best (0.4270 for Random Forest without temporal features).

Recall has improved substantially:

```text
Recall = 0.6392
```

This means the model catches roughly 2 out of every 3 actual next-lap pit stops at the selected threshold. Compared with the earlier baseline (recall 0.2523), the model now misses far fewer real pit stops.

Precision remains high:

```text
Precision = 0.9140
```

This means that when the model does predict a pit stop, it is correct 91% of the time. The threshold policy (`recall_target_0.60`) deliberately trades a small amount of precision for much higher recall, which is the right tradeoff for a strategy assistant that should flag genuine pit windows without overwhelming false alerts.

## 13. Per-Race Recall

The XGBoost model does not perform equally across all held-out races, but recall is much more consistent than the earlier Random Forest baseline.

| Race | Actual pit-next-lap events | XGBoost Recall | Previous RF Recall |
|---|---:|---:|---:|
| Abu Dhabi Grand Prix | 92 | 0.435 | 0.359 |
| Chinese Grand Prix | 65 | 0.738 | 0.015 |
| Dutch Grand Prix | 222 | 0.676 | 0.347 |
| Hungarian Grand Prix | 147 | 0.544 | 0.259 |
| Pre-Season Track Session | 47 | 0.468 | 0.021 |
| Spanish Grand Prix | 192 | 0.776 | 0.224 |

The most dramatic improvements are in races where the earlier model almost completely failed. Chinese GP recall improved from 0.015 to 0.738, and the Pre-Season Track Session from 0.021 to 0.468. These gains come from the temporal lag and rolling features, which give the model visibility into recent pace trends and stint history.

Some variation across races remains. Abu Dhabi (0.435) and Hungarian GP (0.544) still lag behind Spanish GP (0.776) and Chinese GP (0.738). This suggests that some races have pit strategy patterns that are harder to learn from historical data alone, possibly due to unique track characteristics or race conditions.

## 14. Streamlit App

The Streamlit app is located at:

```text
app/app.py
```

The app loads:

```text
models/best_model_candidate.joblib
```

or falls back to:

```text
models/random_forest.joblib
```

It also reads:

```text
data/f1_strategy_dataset_v4_validated.csv
data/sample_prediction_input.json
reports/
```

The app provides an interactive interface for:

- Exploring the project and model behavior.
- Entering or adjusting lap-level input values.
- Generating a pit-next-lap probability.
- Showing model outputs in a user-friendly way.

For Streamlit deployment, the project needs the app file, requirements file, model artifacts, and data files to stay available in the repository or deployment environment.

## 15. Project File Map

Important files and directories:

```text
app/
  app.py                         Streamlit application

src/
  preprocessing.py               Data loading, leakage removal, split, preprocessing
  train.py                       Model training and threshold tuning
  evaluate.py                    Test-set evaluation and plots
  test_preprocessing.py          Preprocessing tests

data/
  f1_strategy_dataset_v4_validated.csv
  sample_prediction_input.json
  processed/
    X_train.csv
    X_test.csv
    y_train.csv
    y_test.csv
    test_races.json
    preprocessing_metadata.json

models/
  best_model_candidate.joblib
  random_forest.joblib
  gradient_boosting.joblib
  decision_tree.joblib
  logistic_regression.joblib
  majority_baseline.joblib
  preprocessor_scaled.joblib
  preprocessor_tree.joblib

reports/
  RESULT.md
  training_summary.md
  training_summary.json
  model_comparison.csv
  cv_results.json
  best_model_per_race_recall.csv
  figures/
```

## 16. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python -m src.train
```

Evaluate models:

```bash
python -m src.evaluate
```

Run preprocessing tests:

```bash
pytest src/test_preprocessing.py -q
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

The project uses:

```text
random_state = 42
```

This helps keep splits and model training reproducible.

## 17. Current Strengths

The project has several strong engineering and machine learning choices:

- Uses a race-grouped train/test split to reduce overly optimistic results.
- Removes direct leakage columns before training.
- Adds causal lag and rolling temporal features per (Driver, Race, Year) group, giving the model visibility into recent pace trends and stint history without leaking future information.
- Compares seven model families (including LightGBM and XGBoost) instead of relying on one model.
- Uses native `scale_pos_weight` in LightGBM and XGBoost for direct handling of class imbalance.
- Applies isotonic probability calibration on out-of-fold predictions before threshold selection.
- Uses a recall-targeted threshold policy (recall ≥ 0.60) rather than F1-max, making the operating point appropriate for real strategy decisions.
- Uses ROC-AUC as the main ranking metric for separating pit-risk laps from normal laps.
- Still reports PR-AUC as a supporting metric for the rare pit-next-lap class.
- Saves model artifacts and evaluation reports for reproducibility.
- Includes preprocessing tests for leakage, split correctness, missing values, temporal feature causality, and transformations.
- Provides a Streamlit app for interactive use.

## 18. Current Limitations

The model is a strong prototype, but it has remaining limitations:

- The Streamlit app sends a single-lap input with no prior history, so all lag and rolling features fall back to sentinel values at inference time. A live strategy tool would need to pass the previous N laps through the pipeline to use real temporal context.
- Performance still varies across races: Abu Dhabi recall (0.435) and Hungarian GP recall (0.544) lag behind Spanish GP (0.776) and Chinese GP (0.738).
- The model does not include live race context such as weather, safety car state, gaps to nearby cars, track status, or undercut window estimates.
- There are no SHAP or feature-importance explanations in the app to show why the model predicted a high pit risk on a given lap.

## 19. Recommended Next Improvements

The most useful next improvements are:

1. Add rolling inference context for the app.

   The current app sends a single lap with no history, so temporal features fall back to sentinel values. A rolling inference mode would pass the previous N laps for the selected driver and race through the pipeline before generating the prediction for the current lap:

   ```text
   laps [L-3, L-2, L-1] → temporal features → prediction for lap L
   ```

   This would unlock the full value of the lag and rolling features at inference time.

2. Add live race context features.

   ```text
   gap to car ahead
   gap to car behind
   safety car state
   weather
   track status
   pit window estimates
   ```

3. Add SHAP explanations.

   Add feature importance or SHAP values to show why the model predicted high pit risk on a specific lap. This would make the app more useful as an explanation and teaching tool.

4. Ensemble the top models.

   XGBoost, LightGBM, and Random Forest all achieve ROC-AUC above 0.987. Combining their calibrated probabilities (e.g. via a simple average or a stacked meta-learner) may push performance further.

## 20. Final Assessment

Race Strategist is a complete and reproducible F1 pit-next-lap prediction system with strong held-out performance.

The machine learning pipeline is well structured:

- The target is clearly defined.
- Leakage is handled carefully.
- Temporal features are computed causally per (Driver, Race, Year) group.
- Evaluation uses race-grouped splits.
- Metrics are appropriate for imbalanced classification.
- Probabilities are isotonic-calibrated before threshold selection.
- The best model significantly beats the majority baseline on every metric.

The current XGBoost model catches roughly 2 out of every 3 actual pit stops on the held-out test set (recall 0.6392) while still being correct 91% of the time when it does predict a stop (precision 0.9140). PR-AUC reached 0.8571, more than double the earlier baseline of 0.4270 and 24 times above the random floor.

The most impactful remaining improvement is a rolling inference mode in the app so that lag and rolling features reflect real prior-lap history rather than sentinel values.
