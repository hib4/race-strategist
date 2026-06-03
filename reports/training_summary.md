# F1 Race Strategist — Model Training Summary

**Test set:** 22,088 rows, positive rate 0.0346
**Best model:** `random_forest` — PR-AUC 0.4270, F1 0.3979, threshold 0.647

## Comparison (sorted by PR-AUC)

| model               |   pr_auc |   roc_auc |     f1 |   precision |   recall |   brier |   threshold |   fit_time_sec |   inference_time_sec |
|:--------------------|---------:|----------:|-------:|------------:|---------:|--------:|------------:|---------------:|---------------------:|
| random_forest       |   0.4270 |    0.8938 | 0.3979 |      0.9415 |   0.2523 |  0.0361 |      0.6472 |        40.1903 |               0.0536 |
| gradient_boosting   |   0.4185 |    0.8741 | 0.3380 |      0.9573 |   0.2052 |  0.1564 |      0.9663 |        94.1640 |               0.0191 |
| decision_tree       |   0.3935 |    0.8252 | 0.4130 |      0.8333 |   0.2745 |  0.1255 |      0.9221 |         3.1852 |               0.0067 |
| logistic_regression |   0.2942 |    0.8559 | 0.2443 |      0.7368 |   0.1464 |  0.0956 |      0.8698 |         3.6896 |               0.0077 |
| majority_baseline   |   0.0346 |  nan      | 0.0000 |      0.0000 |   0.0000 |  0.0346 |      1.1000 |         0.0157 |               0.0136 |

## Figures

- `figures/pr_curves.png`
- `figures/roc_curves.png`
- `figures/confusion_matrices.png`
- `figures/feature_importance_rf.png`

## Notes

- Splits are race-grouped: no race in `test_races.json` appears in training.
- Decision thresholds were tuned on OOF predictions from GroupKFold(5) to maximize F1 on the PR curve.
- Random baseline PR-AUC for a 0.0346 positive rate is ~0.0346. All non-baseline models exceed this by >5x.