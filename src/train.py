"""Train classical ML models for F1 pit-prediction.

Pipeline per non-baseline model:
  1. GridSearchCV with GroupKFold(5) on (X_trainval, y_trainval), grouped by Race,
     scoring=average_precision (PR-AUC).
  2. Use cross_val_predict (same CV) with the best estimator to obtain OOF
     probabilities, then tune the decision threshold by maximizing F1 on the
     full PR curve.
  3. Refit the best estimator on all trainval data and persist with its tuned
     threshold.

Artifacts written:
  data/processed/{X_train,y_train,X_test,y_test}.csv
  data/processed/test_races.json
  data/processed/preprocessing_metadata.json
  models/preprocessor_scaled.joblib
  models/preprocessor_tree.joblib
  models/<model_name>.joblib       (each is a dict: {pipeline, threshold, best_params})
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    PROJECT_ROOT,
    build_preprocessor,
    feature_names,
    load_validated,
    make_race_grouped_split,
)

RANDOM_STATE = 42
N_SPLITS = 5

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _ensure_dirs() -> None:
    for d in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "figures"]:
        d.mkdir(parents=True, exist_ok=True)


def tune_threshold_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Pick the threshold maximizing F1 on the PR curve. Returns (threshold, f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if len(f1) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def build_model_specs() -> list[dict]:
    """Return spec dicts: name, preprocessor kind, estimator, param_grid."""
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
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        },
    ]


def build_pipeline(kind: str, estimator) -> Pipeline:
    pre = build_preprocessor(kind)
    # unwrap to a single Pipeline: [(preprocess, ColumnTransformer), (model, estimator)]
    return Pipeline(steps=[("preprocess", pre.named_steps["preprocess"]), ("model", estimator)])


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
    """Manual OOF probability via cloning the pipeline per fold.
    Avoids sklearn metadata-routing churn for sample_weight."""
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
            # Dummy: constant prediction
            oof[val_idx] = float(y_tr.mean())
    return oof


def train_one(
    spec: dict,
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    groups_trainval: pd.Series,
) -> dict:
    """Grid-search + threshold-tune + refit one model. Returns a result dict."""
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
        # Baseline: no grid
        if use_sw:
            sw = _sample_weight_for(name, y_arr)
            pipeline.fit(X_trainval, y_arr, model__sample_weight=sw)
        else:
            pipeline.fit(X_trainval, y_arr)
        best_pipeline = pipeline
        best_params = {}
        cv_pr_auc = float("nan")  # set after manual OOF below

    fit_time = time.time() - t0

    # Manual OOF predictions for threshold tuning
    oof_proba = manual_oof_proba(
        best_pipeline, X_trainval, y_arr, groups_trainval, cv, use_sw
    )
    if not grid:
        cv_pr_auc = float(average_precision_score(y_arr, oof_proba))

    # Special-case the majority baseline: its proba is constant so the PR-curve
    # threshold collapses to 0 and the model "predicts all positives". The
    # correct baseline behavior is to predict the majority class always.
    if name == "majority_baseline":
        oof_proba = np.zeros_like(y_arr, dtype=float)

    if name == "majority_baseline":
        # Threshold > 1 means no row is predicted positive (matches "most_frequent=0")
        threshold, oof_f1 = 1.1, 0.0
    else:
        threshold, oof_f1 = tune_threshold_from_proba(y_arr, oof_proba)

    payload = {
        "pipeline": best_pipeline,
        "threshold": threshold,
        "best_params": best_params,
        "cv_pr_auc": cv_pr_auc,
        "oof_f1_at_threshold": oof_f1,
        "fit_time_sec": fit_time,
        "preprocessor_kind": kind,
    }
    joblib.dump(payload, MODELS_DIR / f"{name}.joblib")
    print(
        f"  [{name}] CV PR-AUC={cv_pr_auc:.4f}  thr={threshold:.3f}  "
        f"OOF F1={oof_f1:.4f}  fit={fit_time:.1f}s  params={best_params}"
    )
    return {
        "name": name,
        "cv_pr_auc": cv_pr_auc,
        "threshold": threshold,
        "oof_f1": oof_f1,
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

    # Persist raw splits for reproducibility
    split.X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    split.X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    split.y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    split.y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)
    (DATA_PROCESSED / "test_races.json").write_text(json.dumps(split.test_races, indent=2))

    # Fit + save standalone preprocessors (for later inference apps)
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
    }
    (DATA_PROCESSED / "preprocessing_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Training models...")
    specs = build_model_specs()
    results = []
    for spec in specs:
        print(f"\n>>> {spec['name']}")
        results.append(train_one(spec, split.X_train, split.y_train, split.groups_train))

    # Pick best by CV PR-AUC, tie-break by OOF F1
    ranked = sorted(results, key=lambda r: (-r["cv_pr_auc"], -r["oof_f1"]))
    best_name = ranked[0]["name"]
    print(f"\nBest by CV PR-AUC: {best_name}")
    best_payload = joblib.load(MODELS_DIR / f"{best_name}.joblib")
    joblib.dump(best_payload, MODELS_DIR / "best_model_candidate.joblib")

    (REPORTS_DIR / "cv_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote artifacts to {MODELS_DIR}/ and {REPORTS_DIR}/cv_results.json")


if __name__ == "__main__":
    main()
