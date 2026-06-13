# F1 Pit Stop Predictor — Model Training Summary

**Test set:** 22,088 rows, positive rate 0.0346
**Threshold policy:** `recall_target_0.60` (target recall = 0.60)
**Best model:** `xgboost` — PR-AUC 0.8571, recall 0.6392, precision 0.9140, F1 0.7523, threshold 0.488

## Comparison (sorted by PR-AUC)

| model               |   pr_auc |   roc_auc |     f1 |   precision |   recall |   brier |   threshold | policy             | calibrated   |   fit_time_sec |   inference_time_sec |
|:--------------------|---------:|----------:|-------:|------------:|---------:|--------:|------------:|:-------------------|:-------------|---------------:|---------------------:|
| xgboost             |   0.8571 |    0.9880 | 0.7523 |      0.9140 |   0.6392 |  0.0122 |      0.4885 | recall_target_0.60 | True         |        16.7599 |               0.0241 |
| random_forest       |   0.8506 |    0.9874 | 0.7028 |      0.9000 |   0.5765 |  0.0119 |      0.5612 | recall_target_0.60 | True         |        73.3210 |               0.0542 |
| lightgbm            |   0.8440 |    0.9877 | 0.7338 |      0.9109 |   0.6144 |  0.0125 |      0.5112 | recall_target_0.60 | True         |        96.3746 |               0.0630 |
| gradient_boosting   |   0.7859 |    0.9859 | 0.5806 |      0.8612 |   0.4379 |  0.0164 |      0.4868 | recall_target_0.60 | True         |       623.3061 |               0.0295 |
| decision_tree       |   0.7504 |    0.9319 | 0.7208 |      0.8285 |   0.6379 |  0.0151 |      0.3919 | recall_target_0.60 | True         |        10.8952 |               0.0091 |
| logistic_regression |   0.3163 |    0.9156 | 0.3875 |      0.3649 |   0.4131 |  0.0284 |      0.1044 | recall_target_0.60 | True         |        18.7609 |               0.0118 |
| majority_baseline   |   0.0346 |  nan      | 0.0000 |      0.0000 |   0.0000 |  0.0346 |      1.1000 | recall_target_0.60 | False        |         0.0261 |               0.0218 |

## Figures

- `figures/pr_curves.png`
- `figures/roc_curves.png`
- `figures/confusion_matrices.png`
- `figures/feature_importance_rf.png`

## Notes

- Splits are race-grouped: no race in `test_races.json` appears in training.
- Per-model probabilities are isotonic-calibrated on OOF predictions from GroupKFold(5).
- Decision thresholds were tuned on calibrated OOF predictions using the `recall_target_0.60` policy: pick the highest-precision threshold whose recall meets the target.
- Best-model selection prefers models that hit the target recall on test, then ranks by PR-AUC.
- Random baseline PR-AUC for a 0.0346 positive rate is ~0.0346. All non-baseline models exceed this by >5x.