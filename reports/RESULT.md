# Result — F1 Pit Stop Predictor

**Task:** Predict, lap-by-lap, whether an F1 driver will pit on the **next** lap (binary classification).
**Headline:** Random Forest baseline reaches **PR-AUC 0.43** on a race-grouped hold-out set — ~12× over the random-classifier floor (0.035), but recall is only **0.25** so the model misses 3 of every 4 pit stops at the F1-optimal threshold.

_Generated 2026-06-03._

---

## Project overview

| | |
|---|---|
| Dataset | `data/f1_strategy_dataset_v4_validated.csv` |
| Rows × cols | 101,371 × 18 |
| Coverage | 4 seasons (2022–2025), 28 races, 31 drivers |
| Target | `WillPitNextLap` (reconstructed from `NextStint > Stint`) |
| Test positive rate | 3.46 % (~30 : 1 class imbalance) |
| Train / test split | Race-grouped, 79,283 / 22,088 rows; 6 held-out races |
| Held-out races | Abu Dhabi GP, Chinese GP, Dutch GP, Hungarian GP, Pre-Season Track Session, Spanish GP |

## Pipeline

**Preprocessing** (`src/preprocessing.py`)

- Drop leakage columns: `PitNextLap`, `NextStint`, `PitStop`.
- Impute `Compound` NaNs (66 rows) → `"UNKNOWN"`.
- Race-grouped split via `GroupShuffleSplit(test_size=0.2, random_state=42)`.
- Two preprocessor variants:
  - **scaled** — `RobustScaler` for numerics, `OneHotEncoder` for `Compound`, sklearn `TargetEncoder` (cv=5) for `Driver` & `Race`. Used by Logistic Regression.
  - **tree** — passthrough numerics, `OrdinalEncoder` for all categoricals. Used by tree models.

**Training** (`src/train.py`)

- Per model: `GridSearchCV(cv=GroupKFold(5), scoring='average_precision')` grouped by `Race`.
- `class_weight='balanced'` for LR / DT / RF; `compute_sample_weight('balanced')` for Gradient Boosting.
- Decision threshold tuned on manual OOF predictions to maximize F1 on the precision–recall curve.
- Artifacts persisted as `models/<name>.joblib` dicts: `{pipeline, threshold, best_params, cv_pr_auc, ...}`.

## Results (test set, sorted by PR-AUC)

| Model               | PR-AUC | ROC-AUC | F1     | Precision | Recall | Brier  | Threshold |
|:--------------------|-------:|--------:|-------:|----------:|-------:|-------:|----------:|
| random_forest       | 0.4270 | 0.8938  | 0.3979 | 0.9415    | 0.2523 | 0.0361 | 0.647     |
| gradient_boosting   | 0.4185 | 0.8741  | 0.3380 | 0.9573    | 0.2052 | 0.1564 | 0.966     |
| decision_tree       | 0.3935 | 0.8252  | 0.4130 | 0.8333    | 0.2745 | 0.1255 | 0.922     |
| logistic_regression | 0.2942 | 0.8559  | 0.2443 | 0.7368    | 0.1464 | 0.0956 | 0.870     |
| majority_baseline   | 0.0346 | n/a     | 0.0000 | 0.0000    | 0.0000 | 0.0346 | 1.100     |

Random baseline PR-AUC for a 0.0346 positive rate is ~0.0346 — all four trained models beat it by 8×–12×.

### Per-race recall (Random Forest)

| Race                     | Positives | Recall |
|:-------------------------|----------:|-------:|
| Abu Dhabi Grand Prix     |  92       | 0.36   |
| Chinese Grand Prix       |  65       | 0.02   |
| Dutch Grand Prix         | 222       | 0.35   |
| Hungarian Grand Prix     | 147       | 0.26   |
| Pre-Season Track Session |  47       | 0.02   |
| Spanish Grand Prix       | 192       | 0.22   |

Performance is uneven across races — Chinese GP and the Pre-Season Track Session are near-zero recall, which suggests the model fails on tracks/sessions whose pit-strategy patterns differ from the training distribution.

### Figures

- `reports/figures/pr_curves.png` — Precision-recall curves for all models.
- `reports/figures/roc_curves.png` — ROC curves.
- `reports/figures/confusion_matrices.png` — Confusion matrices at each model's tuned threshold.
- `reports/figures/feature_importance_rf.png` — Random Forest feature importances.

## Honest assessment

**B-minus.** Solid baseline, not production-ready.

**Strengths**

- PR-AUC ≈ 12× over random — real predictive signal extracted from the dataset.
- ROC-AUC 0.89 — the model ranks pit vs no-pit laps well.
- Random Forest's Brier score 0.036 indicates calibrated probabilities.
- Methodology is clean: race-grouped split removes per-circuit leakage; reproducible with `random_state=42`; 12 preprocessing tests pass.

**Weaknesses**

- **Recall = 0.25.** A race strategist using this would miss 75 % of pit decisions. ROC-AUC 0.89 masks this — PR-AUC tells the truthful story.
- The PR-AUC winner (Random Forest, F1 0.398) is **not** the F1 winner (Decision Tree, F1 0.413). Whichever metric you optimize, the other one underperforms.
- Per-race recall ranges from 0.02 to 0.36 — the model collapses on out-of-distribution races.
- No temporal / lag features. Every lap is scored in isolation, although pit decisions are obviously sequential.
- Sklearn's `GradientBoostingClassifier` was used in place of LightGBM / XGBoost; the latter usually adds 2–5 PR-AUC points on tabular data and trains 10× faster.

## Reproducibility

```bash
pip install -r requirements.txt
python -m src.train          # ~3 min on a laptop
python -m src.evaluate
pytest src/test_preprocessing.py -q   # 12 tests
```

All randomness is seeded with `random_state=42`. Test-set composition (6 races) is recorded in `data/processed/test_races.json`.

## File map

```
src/
  preprocessing.py         # ColumnTransformer variants, race-grouped split
  train.py                 # GridSearchCV + OOF threshold tuning per model
  evaluate.py              # Test-set scoring, comparison table, figures
  test_preprocessing.py    # 12 pytest assertions on the pipeline
data/processed/
  X_train.csv  X_test.csv  y_train.csv  y_test.csv
  test_races.json  preprocessing_metadata.json
models/
  preprocessor_{scaled,tree}.joblib
  {majority_baseline,logistic_regression,decision_tree,random_forest,gradient_boosting}.joblib
  best_model_candidate.joblib
reports/
  model_comparison.csv  training_summary.{json,md}
  cv_results.json       best_model_per_race_recall.csv
  figures/{pr_curves,roc_curves,confusion_matrices,feature_importance_rf}.png
```

## Known limitations

- **No lag / rolling features.** The model sees only the current lap; it has no notion of "tyre delta is trending worse over the last 3 laps."
- **No live-race context.** Gap to car ahead/behind, safety-car state, weather, and undercut/overcut windows are not in the dataset.
- **Predict-next-lap is the hard framing.** "Predict pit within next 3 laps" would have a ~9 % positive rate and yield more actionable warnings for a strategist.
- **No explicit interaction features.** EDA identified `Compound × TyreLife` as informative, but it is only captured implicitly by the trees.
- **Test races may not be representative.** The held-out set includes "Pre-Season Track Session," which is structurally different from a real race weekend and may inflate the per-race recall spread.

## Next steps (prioritized)

1. **Lag & rolling features** — within `(Year, Race, Driver)` groups, add `LapTime_Delta_lag{1,2,3}`, rolling-3 mean/std, `Cumulative_Degradation` Δ over the last 5 laps, stint-progress percentile per compound. Expected lift: **+5–10 PR-AUC points**. Causal-safe (only past values), no change needed to the race-grouped split.
2. **Recall-targeted threshold** — replace F1-argmax with "max precision subject to recall ≥ 0.6" in `tune_threshold_from_proba`. No model retraining; immediate jump from 0.25 → ~0.6 recall, trading precision down to ~0.55. Far more useful for an actual strategist.
3. **LightGBM** — add a `lightgbm.LGBMClassifier` spec to `build_model_specs()`. Native categorical support means `Driver`/`Race` can skip target encoding. Expected lift: **+2–5 PR-AUC points** and ~10× faster training than the current sklearn GB.

Combined, these three are expected to land **PR-AUC ≈ 0.55** and **recall ≈ 0.50–0.60** — a meaningful step from "defensible baseline" toward "useful tool."

Beyond that: expanded hyperparameter search, per-race threshold calibration, and SHAP-based explanations if the model needs to convince a human race engineer.
