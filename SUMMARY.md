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

The project trains five model types:

| Model | Purpose |
|---|---|
| Majority baseline | Simple reference model that predicts the majority class |
| Logistic Regression | Interpretable linear baseline |
| Decision Tree | Simple tree-based model |
| Random Forest | Ensemble of many decision trees |
| Gradient Boosting | Sequential tree ensemble |

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

### Step 3: Threshold Tuning

Most classifiers output a probability, for example:

```text
Pit next lap probability = 0.72
```

But the final prediction needs to be:

```text
0 or 1
```

So the project chooses a threshold. If the probability is greater than or equal to the threshold, the model predicts `1`.

Example:

```text
Probability = 0.72
Threshold = 0.647
Prediction = 1
```

The threshold is tuned using out-of-fold predictions from cross-validation. The selected threshold maximizes F1 on the precision-recall curve.

This is better than blindly using `0.5`, because rare-event classification often needs a different threshold.

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
| Random Forest | 0.8938 | 0.4270 | 0.3979 | 0.9415 | 0.2523 | 0.0361 | 0.647 |
| Gradient Boosting | 0.8741 | 0.4185 | 0.3380 | 0.9573 | 0.2052 | 0.1564 | 0.966 |
| Logistic Regression | 0.8559 | 0.2942 | 0.2443 | 0.7368 | 0.1464 | 0.0956 | 0.870 |
| Decision Tree | 0.8252 | 0.3935 | 0.4130 | 0.8333 | 0.2745 | 0.1255 | 0.922 |
| Majority Baseline | n/a | 0.0346 | 0.0000 | 0.0000 | 0.0000 | 0.0346 | 1.100 |

The best model by ROC-AUC is:

```text
Random Forest
```

Its test performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8938 |
| PR-AUC | 0.4270 |
| F1 | 0.3979 |
| Precision | 0.9415 |
| Recall | 0.2523 |
| Threshold | 0.647 |

## 12. How To Interpret The Results

The Random Forest model is much better than random guessing.

The main result is the Random Forest ROC-AUC:

```text
0.8938
```

This means the model usually ranks actual pit-next-lap situations above no-pit situations. In other words, the model has learned real pit-stop signal from the data.

As a supporting rare-event metric, the Random Forest PR-AUC is also much higher than the random PR-AUC floor.

The random PR-AUC floor is approximately the positive rate:

```text
0.0346
```

The Random Forest PR-AUC is:

```text
0.4270
```

However, the model is not production-ready.

The most important issue is recall:

```text
Recall = 0.2523
```

This means the model catches only about 25% of actual next-lap pit stops at the selected threshold. In practical terms, it misses around 3 out of every 4 real pit-next-lap events.

Precision is very high:

```text
Precision = 0.9415
```

This means that when the model does predict a pit stop, it is usually correct. The model is conservative: it avoids false alarms, but it misses many true pit stops.

For a real race strategy assistant, this tradeoff may not be ideal. A strategist might prefer more warnings, even if some are false alarms. That would require a lower threshold or a recall-targeted thresholding strategy.

## 13. Per-Race Recall

The Random Forest does not perform equally across all held-out races.

| Race | Actual pit-next-lap events | Recall |
|---|---:|---:|
| Abu Dhabi Grand Prix | 92 | 0.359 |
| Chinese Grand Prix | 65 | 0.015 |
| Dutch Grand Prix | 222 | 0.347 |
| Hungarian Grand Prix | 147 | 0.259 |
| Pre-Season Track Session | 47 | 0.021 |
| Spanish Grand Prix | 192 | 0.224 |

This shows a generalization problem.

The model performs reasonably on some races, such as Abu Dhabi and Dutch GP, but almost fails to catch pit stops in Chinese GP and the Pre-Season Track Session. This suggests that pit strategy patterns can vary strongly by race, track, and session type.

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
- Compares multiple model families instead of relying on one model.
- Uses ROC-AUC as the main ranking metric for separating pit-risk laps from normal laps.
- Still reports PR-AUC as a supporting metric for the rare pit-next-lap class.
- Tunes decision thresholds instead of assuming a default threshold of 0.5.
- Saves model artifacts and evaluation reports for reproducibility.
- Includes preprocessing tests for leakage, split correctness, missing values, and transformations.
- Provides a Streamlit app for interactive use.

## 18. Current Limitations

The model is a solid baseline, but it has clear limitations:

- Recall is low, so many real pit stops are missed.
- The model scores each lap mostly as an isolated row.
- It does not yet use rolling or lag features from previous laps.
- It does not include live race context such as weather, safety car, gaps, traffic, or undercut windows.
- Performance varies strongly across races.
- The current best model is useful for detecting high-confidence pit signals, not for catching every pit stop.

## 19. Recommended Next Improvements

The most useful next improvements are:

1. Add lag and rolling features.

   Examples:

   ```text
   LapTime_Delta over last 3 laps
   rolling tyre degradation
   stint progress by compound
   recent position changes
   ```

   Pit strategy is sequential, so recent lap history should improve the model.

2. Tune for higher recall.

   Instead of maximizing F1, choose a threshold that reaches a target recall, such as:

   ```text
   recall >= 0.60
   ```

   This would create more alerts but miss fewer real pit stops.

3. Try stronger tabular models.

   LightGBM or XGBoost may perform better than sklearn Gradient Boosting on this type of structured racing data.

4. Improve race context.

   Add features such as:

   ```text
   gap to car ahead
   gap to car behind
   safety car state
   weather
   track status
   pit window estimates
   ```

5. Explain predictions.

   Add feature importance or SHAP explanations to show why the model predicted high pit risk.

## 20. Final Assessment

Race Strategist is a complete and reproducible baseline for F1 pit-next-lap prediction.

The machine learning pipeline is well structured:

- The target is clearly defined.
- Leakage is handled carefully.
- Evaluation uses race-grouped splits.
- Metrics are appropriate for imbalanced classification.
- The best model significantly beats the majority baseline.

The current Random Forest model is best understood as a high-precision pit-risk detector. When it predicts a pit stop, it is often correct, but it still misses many actual pit stops. The next major step is to improve recall with better temporal features and thresholding.
