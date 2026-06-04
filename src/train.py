"""Train classical ML models for F1 pit-prediction.

Pipeline per non-baseline model:
  1. GridSearchCV with GroupKFold(5) on (X_trainval, y_trainval), grouped by Race,
     scoring=average_precision (PR-AUC).
  2. Manual OOF probabilities via the same GroupKFold splits using the tuned
     hyperparameters.
  3. Fit an isotonic calibrator on those OOF probabilities and the true labels,
     then tune the decision threshold on the *calibrated* OOF probabilities to
     hit a target recall.
  4. Refit the best estimator on all trainval data, wrap it with the calibrator
     (CalibratedPipeline), and persist the calibrated wrapper plus the tuned
     threshold.

Threshold policy:
  - recall_target_0.60: pick the *highest precision* threshold whose
    OOF recall >= 0.60. Falls back to F-beta=2 (recall-weighted) if no
    threshold reaches the target.

Best-model selection ranks by (achieved OOF recall >= target, then CV PR-AUC).

Artifacts written:
  data/processed/{X_train,y_train,X_test,y_test}.csv
  data/processed/test_races.json
  data/processed/preprocessing_metadata.json
  models/preprocessor_scaled.joblib
  models/preprocessor_tree.joblib
  models/<model_name>.joblib       (each is a dict: {pipeline, threshold, ...})
  models/best_model_candidate.joblib
  reports/cv_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import (
    PROJECT_ROOT,
    build_preprocessor,
    feature_names,
    load_validated,
    make_race_grouped_split,
)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET_RECALL = 0.60
THRESHOLD_POLICY = f"recall_target_{TARGET_RECALL:.2f}"

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _ensure_dirs() -> None:
    for d in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "figures"]:
        d.mkdir(parents=True, exist_ok=True)


class CalibratedPipeline:
    """Wraps a fitted sklearn Pipeline with an isotonic probability calibrator.

    Exposes the same `predict`, `predict_proba`, `named_steps`, and
    `feature_names_in_` surface the eval/app code touches, so it is a
    drop-in replacement for the underlying Pipeline.
    """

    def __init__(self, pipeline: Pipeline, calibrator: IsotonicRegression):
        self.pipeline = pipeline
        self.calibrator = calibrator

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    @property
    def feature_names_in_(self):
        return self.pipeline.feature_names_in_

    def predict_proba(self, X) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def tune_threshold_f1(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float, float, float]:
    """Pick the threshold maximizing F1 on the PR curve.
    Returns (threshold, precision, recall, f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    prec = precision[:-1]
    rec = recall[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if len(f1) == 0:
        return 0.5, 0.0, 0.0, 0.0
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(prec[i]), float(rec[i]), float(f1[i])


def tune_threshold_recall_target(
    y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = TARGET_RECALL
) -> tuple[float, float, float, float]:
    """Pick the highest-precision threshold whose recall meets the target.

    Falls back to F-beta=2 (recall-weighted) if no threshold achieves
    `target_recall`. Returns (threshold, precision, recall, f1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    prec = precision[:-1]
    rec = recall[:-1]
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0

    mask = rec >= target_recall
    if mask.any():
        candidate_prec = prec[mask]
        candidate_rec = rec[mask]
        candidate_thr = thresholds[mask]
        i = int(np.argmax(candidate_prec))
        p, r, t = float(candidate_prec[i]), float(candidate_rec[i]), float(candidate_thr[i])
        f1 = 2 * p * r / (p + r + 1e-12)
        return t, p, r, f1

    # Fallback: F-beta with beta=2 (favors recall) over the whole PR curve.
    beta_sq = 4.0
    fbeta = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec + 1e-12)
    i = int(np.argmax(fbeta))
    p, r, t = float(prec[i]), float(rec[i]), float(thresholds[i])
    f1 = 2 * p * r / (p + r + 1e-12)
    return t, p, r, f1


def build_model_specs(y_trainval: np.ndarray) -> list[dict]:
    """Return spec dicts: name, preprocessor kind, estimator, param_grid.

    `scale_pos_weight` is the negative/positive ratio of `y_trainval` and is
    baked into LightGBM/XGBoost at construction time.
    """
    n_pos = max(1, int((y_trainval == 1).sum()))
    n_neg = int((y_trainval == 0).sum())
    pos_weight = float(n_neg / n_pos)

    return [
        {
            "name": "majority_baseline",
            "kind": "tree",
            "estimator": DummyClassifier(strategy="most_frequent"),
            "param_grid": {},
        },
        {
            "name": "logistic_regression",
            "kind": "scaled",
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            },
        },
        {
            "name": "decision_tree",
            "kind": "tree",
            "estimator": DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "param_grid": {
                "model__max_depth": [6, 10, 16, None],
                "model__min_samples_leaf": [10, 50, 200],
            },
        },
        {
            "name": "random_forest",
            "kind": "tree",
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__n_estimators": [200],
                "model__max_depth": [12, 20, None],
                "model__min_samples_leaf": [5, 20],
                "model__max_features": ["sqrt"],
            },
        },
        {
            "name": "gradient_boosting",
            "kind": "tree",
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "model__n_estimators": [200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
            },
        },
        {
            "name": "lightgbm",
            "kind": "tree",
            "estimator": LGBMClassifier(
                objective="binary",
                scale_pos_weight=pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            "param_grid": {
                "model__n_estimators": [400],
                "model__learning_rate": [0.05],
                "model__num_leaves": [31, 63],
                "model__min_child_samples": [20, 50],
                "model__reg_lambda": [0.0, 1.0],
            },
        },
        {
            "name": "xgboost",
            "kind": "tree",
            "estimator": XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                scale_pos_weight=pos_weight,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            "param_grid": {
                "model__n_estimators": [400],
                "model__learning_rate": [0.05],
                "model__max_depth": [6, 8],
                "model__min_child_weight": [1, 5],
                "model__subsample": [0.8, 1.0],
            },
        },
    ]


def build_pipeline(kind: str, estimator) -> Pipeline:
    pre = build_preprocessor(kind)
    return Pipeline(
        steps=[("preprocess", pre.named_steps["preprocess"]), ("model", estimator)]
    )


def _needs_sample_weight(name: str) -> bool:
    return name == "gradient_boosting"


def _sample_weight_for(name: str, y_train: np.ndarray) -> np.ndarray | None:
    if _needs_sample_weight(name):
        from sklearn.utils.class_weight import compute_sample_weight
        return compute_sample_weight("balanced", y_train)
    return None


def manual_oof_proba(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    cv: GroupKFold,
    use_sample_weight: bool,
) -> np.ndarray:
    """Manual OOF probability via cloning the pipeline per fold."""
    oof = np.zeros(len(y), dtype=float)
    has_proba = hasattr(pipeline.named_steps["model"], "predict_proba")
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]
        model = clone(pipeline)
        if use_sample_weight:
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", y_tr)
            model.fit(X_tr, y_tr, model__sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        if has_proba:
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
        else:
            oof[val_idx] = float(y_tr.mean())
    return oof


def train_one(
    spec: dict,
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    groups_trainval: pd.Series,
) -> dict:
    """Grid-search + calibrate + recall-target threshold-tune + refit one model."""
    name = spec["name"]
    kind = spec["kind"]
    grid = spec["param_grid"]
    estimator = spec["estimator"]

    pipeline = build_pipeline(kind, estimator)
    cv = GroupKFold(n_splits=N_SPLITS)

    y_arr = y_trainval.to_numpy()
    use_sw = _needs_sample_weight(name)

    t0 = time.time()
    if grid:
        gs = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        if use_sw:
            sw = _sample_weight_for(name, y_arr)
            gs.fit(X_trainval, y_arr, groups=groups_trainval, model__sample_weight=sw)
        else:
            gs.fit(X_trainval, y_arr, groups=groups_trainval)
        best_pipeline = gs.best_estimator_
        best_params = gs.best_params_
        cv_pr_auc = float(gs.best_score_)
    else:
        if use_sw:
            sw = _sample_weight_for(name, y_arr)
            pipeline.fit(X_trainval, y_arr, model__sample_weight=sw)
        else:
            pipeline.fit(X_trainval, y_arr)
        best_pipeline = pipeline
        best_params = {}
        cv_pr_auc = float("nan")

    fit_time = time.time() - t0

    # OOF probabilities from the chosen hyperparameters.
    oof_proba_raw = manual_oof_proba(
        best_pipeline, X_trainval, y_arr, groups_trainval, cv, use_sw
    )
    if not grid:
        cv_pr_auc = float(average_precision_score(y_arr, oof_proba_raw))

    # The majority baseline gets no calibration: its proba is constant and the
    # PR-curve threshold collapses. Force a "predict all negatives" model.
    if name == "majority_baseline":
        calibrator = None
        final_pipeline = best_pipeline
        oof_proba_cal = np.zeros_like(y_arr, dtype=float)
        threshold = 1.1
        prec_at_thr, rec_at_thr, f1_at_thr = 0.0, 0.0, 0.0
        f1_threshold = 1.1
        f1_recall = 0.0
        f1_precision = 0.0
        f1_f1 = 0.0
    else:
        # Fit isotonic calibrator on OOF predictions, then re-apply to OOF for
        # threshold tuning. The final saved pipeline uses the same calibrator
        # on top of a base estimator refit on the full trainval data.
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(oof_proba_raw, y_arr)
        oof_proba_cal = np.clip(calibrator.predict(oof_proba_raw), 0.0, 1.0)

        threshold, prec_at_thr, rec_at_thr, f1_at_thr = tune_threshold_recall_target(
            y_arr, oof_proba_cal, TARGET_RECALL
        )
        # Also record what the best-F1 threshold would have been, for diagnostics.
        f1_threshold, f1_precision, f1_recall, f1_f1 = tune_threshold_f1(
            y_arr, oof_proba_cal
        )
        final_pipeline = CalibratedPipeline(best_pipeline, calibrator)

    # Recompute calibrated CV PR-AUC for honest reporting.
    cv_pr_auc_cal = (
        float(average_precision_score(y_arr, oof_proba_cal))
        if name != "majority_baseline"
        else float(average_precision_score(y_arr, np.full_like(y_arr, y_arr.mean(), dtype=float)))
    )

    achieved_target = bool(rec_at_thr >= TARGET_RECALL) and name != "majority_baseline"

    payload = {
        "pipeline": final_pipeline,
        "threshold": threshold,
        "best_params": best_params,
        "cv_pr_auc": cv_pr_auc,
        "cv_pr_auc_calibrated": cv_pr_auc_cal,
        "oof_precision_at_threshold": prec_at_thr,
        "oof_recall_at_threshold": rec_at_thr,
        "oof_f1_at_threshold": f1_at_thr,
        "policy": THRESHOLD_POLICY,
        "target_recall": TARGET_RECALL,
        "achieved_target_recall": achieved_target,
        "f1_threshold": f1_threshold,
        "f1_precision": f1_precision,
        "f1_recall": f1_recall,
        "f1_f1": f1_f1,
        "fit_time_sec": fit_time,
        "preprocessor_kind": kind,
        "calibrated": calibrator is not None,
    }
    joblib.dump(payload, MODELS_DIR / f"{name}.joblib")
    print(
        f"  [{name}] CV PR-AUC={cv_pr_auc:.4f} (cal={cv_pr_auc_cal:.4f}) "
        f"thr={threshold:.3f} OOF P={prec_at_thr:.3f} R={rec_at_thr:.3f} "
        f"F1={f1_at_thr:.3f} target_hit={achieved_target} fit={fit_time:.1f}s"
    )
    return {
        "name": name,
        "cv_pr_auc": cv_pr_auc,
        "cv_pr_auc_calibrated": cv_pr_auc_cal,
        "threshold": threshold,
        "oof_precision": prec_at_thr,
        "oof_recall": rec_at_thr,
        "oof_f1": f1_at_thr,
        "achieved_target_recall": achieved_target,
        "best_params": best_params,
        "fit_time_sec": fit_time,
        "preprocessor_kind": kind,
    }


def main() -> None:
    _ensure_dirs()
    print("Loading data...")
    df = load_validated()
    split = make_race_grouped_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"  trainval={split.X_train.shape}  test={split.X_test.shape}")
    print(f"  trainval pos rate={split.y_train.mean():.4f}  test pos rate={split.y_test.mean():.4f}")

    split.X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    split.X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    split.y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    split.y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)
    (DATA_PROCESSED / "test_races.json").write_text(json.dumps(split.test_races, indent=2))

    print("Fitting standalone preprocessors...")
    for kind in ("scaled", "tree"):
        pre = build_preprocessor(kind)
        pre.fit(split.X_train, split.y_train)
        joblib.dump(pre, MODELS_DIR / f"preprocessor_{kind}.joblib")
    meta = {
        "scaled_feature_names": feature_names(build_preprocessor("scaled").fit(split.X_train, split.y_train)),
        "tree_feature_names": feature_names(build_preprocessor("tree").fit(split.X_train, split.y_train)),
        "target": "WillPitNextLap",
        "test_races": split.test_races,
        "n_train_rows": int(len(split.X_train)),
        "n_test_rows": int(len(split.X_test)),
        "train_pos_rate": float(split.y_train.mean()),
        "test_pos_rate": float(split.y_test.mean()),
        "random_state": RANDOM_STATE,
        "threshold_policy": THRESHOLD_POLICY,
        "target_recall": TARGET_RECALL,
    }
    (DATA_PROCESSED / "preprocessing_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nTraining models (threshold policy: {THRESHOLD_POLICY})...")
    specs = build_model_specs(split.y_train.to_numpy())
    results = []
    for spec in specs:
        print(f"\n>>> {spec['name']}")
        results.append(train_one(spec, split.X_train, split.y_train, split.groups_train))

    # Rank: hit-target first, then by calibrated CV PR-AUC, tie-break by OOF F1.
    ranked = sorted(
        results,
        key=lambda r: (
            not r["achieved_target_recall"],
            -r["cv_pr_auc_calibrated"],
            -r["oof_f1"],
        ),
    )
    best_name = ranked[0]["name"]
    print(f"\nBest model: {best_name} "
          f"(achieved_target_recall={ranked[0]['achieved_target_recall']}, "
          f"calibrated PR-AUC={ranked[0]['cv_pr_auc_calibrated']:.4f})")
    best_payload = joblib.load(MODELS_DIR / f"{best_name}.joblib")
    joblib.dump(best_payload, MODELS_DIR / "best_model_candidate.joblib")

    (REPORTS_DIR / "cv_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote artifacts to {MODELS_DIR}/ and {REPORTS_DIR}/cv_results.json")


if __name__ == "__main__":
    main()
