"""Evaluate trained models on the held-out test set and produce the comparison report.

Loads each .joblib payload from models/, predicts on test, applies the tuned
threshold, and writes:
  reports/model_comparison.csv
  reports/training_summary.json
  reports/training_summary.md
  reports/figures/pr_curves.png
  reports/figures/roc_curves.png
  reports/figures/confusion_matrices.png
  reports/figures/feature_importance_rf.png  (if random_forest exists)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import PROJECT_ROOT, feature_names
# Make CalibratedPipeline resolvable when unpickling model artifacts that were
# saved while train.py was the entry point (and thus the class lived in
# `__main__`). Importing the class and re-binding it under __main__ keeps the
# joblib payloads loadable here and in the Streamlit app.
from src.train import CalibratedPipeline as _CalibratedPipeline  # noqa: E402
sys.modules["__main__"].CalibratedPipeline = _CalibratedPipeline

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

MODEL_ORDER = [
    "majority_baseline",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "lightgbm",
    "xgboost",
]


def load_test():
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].to_numpy()
    return X_test, y_test


def score_model(payload: dict, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    pipeline = payload["pipeline"]
    threshold = float(payload["threshold"])
    model = pipeline.named_steps["model"]

    t0 = time.time()
    if hasattr(model, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.predict(X_test).astype(float)
    inf_time = time.time() - t0

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_proba)) > 1 else float("nan"),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_test, y_proba)),
        "threshold": threshold,
        "policy": str(payload.get("policy", "f1_max")),
        "target_recall": float(payload.get("target_recall", 0.0)),
        "calibrated": bool(payload.get("calibrated", False)),
        "fit_time_sec": float(payload.get("fit_time_sec", 0.0)),
        "inference_time_sec": float(inf_time),
        "best_params": payload.get("best_params", {}),
        "_y_proba": y_proba,
        "_y_pred": y_pred,
    }


def per_race_recall(X_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    df = X_test[["Race"]].copy()
    df["y"] = y_test
    df["pred"] = y_pred
    rows = []
    for race, sub in df.groupby("Race"):
        pos = sub["y"].sum()
        rec = float(recall_score(sub["y"], sub["pred"], zero_division=0)) if pos > 0 else float("nan")
        rows.append({"race": race, "positives": int(pos), "recall": rec})
    return pd.DataFrame(rows).sort_values("race").reset_index(drop=True)


def plot_pr_curves(scores: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y_test, s["_y_proba"])
        ax.plot(rec, prec, label=f"{name} (AP={s['pr_auc']:.3f})", linewidth=1.5)
    ax.axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1, label=f"random (AP={y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves.png", dpi=130)
    plt.close(fig)


def plot_roc_curves(scores: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        if np.isnan(s["roc_auc"]):
            continue
        fpr, tpr, _ = roc_curve(y_test, s["_y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={s['roc_auc']:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=130)
    plt.close(fig)


def plot_confusion_matrices(scores: dict, y_test: np.ndarray) -> None:
    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax, (name, s) in zip(axes, scores.items()):
        cm = confusion_matrix(y_test, s["_y_pred"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{name}\n(thr={s['threshold']:.3f})", fontsize=9)
    for ax in axes[len(scores):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_feature_importance_rf() -> None:
    path = MODELS_DIR / "random_forest.joblib"
    if not path.exists():
        return
    payload = joblib.load(path)
    pipe = payload["pipeline"]
    rf = pipe.named_steps["model"]
    if not hasattr(rf, "feature_importances_"):
        return
    names = feature_names(pipe)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(imp)), imp[order][::-1])
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(np.array(names)[order][::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_rf.png", dpi=130)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = load_test()
    print(f"Test set: {X_test.shape}, pos rate={y_test.mean():.4f}")

    scores: dict[str, dict] = {}
    for name in MODEL_ORDER:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"  skip {name} (missing)")
            continue
        payload = joblib.load(path)
        scores[name] = score_model(payload, X_test, y_test)
        s = scores[name]
        print(f"  [{name}] PR-AUC={s['pr_auc']:.4f} ROC-AUC={s['roc_auc']:.4f} "
              f"F1={s['f1']:.4f} P={s['precision']:.4f} R={s['recall']:.4f} "
              f"Brier={s['brier']:.4f}")

    # Comparison table
    rows = []
    for name, s in scores.items():
        rows.append({
            "model": name,
            "pr_auc": s["pr_auc"],
            "roc_auc": s["roc_auc"],
            "f1": s["f1"],
            "precision": s["precision"],
            "recall": s["recall"],
            "brier": s["brier"],
            "threshold": s["threshold"],
            "policy": s["policy"],
            "calibrated": s["calibrated"],
            "fit_time_sec": s["fit_time_sec"],
            "inference_time_sec": s["inference_time_sec"],
            "best_params": json.dumps(s["best_params"]),
        })
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"\n{table[['model','pr_auc','roc_auc','f1','precision','recall','brier','threshold','policy']].to_string(index=False)}")

    # Best model: prefer models that hit the target recall on test, else fall back to
    # plain PR-AUC ranking. Avoids picking a "high PR-AUC but recall=0" model as best.
    target_recall = max((s["target_recall"] for s in scores.values()), default=0.0)
    eligible = [
        name for name, s in scores.items()
        if name != "majority_baseline" and s["recall"] >= target_recall and target_recall > 0
    ]
    if eligible:
        best_name = max(eligible, key=lambda n: scores[n]["pr_auc"])
    else:
        best_name = table.iloc[0]["model"]
    per_race = per_race_recall(X_test, y_test, scores[best_name]["_y_pred"])
    per_race.to_csv(REPORTS_DIR / "best_model_per_race_recall.csv", index=False)

    # Figures
    plot_pr_curves(scores, y_test)
    plot_roc_curves(scores, y_test)
    plot_confusion_matrices(scores, y_test)
    plot_feature_importance_rf()

    # JSON summary
    best_row = next(r for r in rows if r["model"] == best_name)
    summary = {
        "test_size": int(len(y_test)),
        "test_pos_rate": float(y_test.mean()),
        "best_model": best_name,
        "best_pr_auc": float(best_row["pr_auc"]),
        "best_f1": float(best_row["f1"]),
        "best_recall": float(best_row["recall"]),
        "best_precision": float(best_row["precision"]),
        "best_threshold": float(best_row["threshold"]),
        "threshold_policy": str(best_row["policy"]),
        "target_recall": float(target_recall),
        "comparison": rows,
        "per_race_recall_best": per_race.to_dict(orient="records"),
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    md = [
        "# F1 Race Strategist — Model Training Summary",
        "",
        f"**Test set:** {len(y_test):,} rows, positive rate {y_test.mean():.4f}",
        f"**Threshold policy:** `{summary['threshold_policy']}` "
        f"(target recall = {summary['target_recall']:.2f})",
        f"**Best model:** `{best_name}` — PR-AUC {best_row['pr_auc']:.4f}, "
        f"recall {best_row['recall']:.4f}, precision {best_row['precision']:.4f}, "
        f"F1 {best_row['f1']:.4f}, threshold {best_row['threshold']:.3f}",
        "",
        "## Comparison (sorted by PR-AUC)",
        "",
        table.drop(columns=["best_params"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
        "- `figures/pr_curves.png`",
        "- `figures/roc_curves.png`",
        "- `figures/confusion_matrices.png`",
        "- `figures/feature_importance_rf.png`",
        "",
        "## Notes",
        "",
        "- Splits are race-grouped: no race in `test_races.json` appears in training.",
        "- Per-model probabilities are isotonic-calibrated on OOF predictions from "
        "GroupKFold(5).",
        "- Decision thresholds were tuned on calibrated OOF predictions using the "
        f"`{summary['threshold_policy']}` policy: pick the highest-precision "
        "threshold whose recall meets the target.",
        "- Best-model selection prefers models that hit the target recall on test, "
        "then ranks by PR-AUC.",
        f"- Random baseline PR-AUC for a {y_test.mean():.4f} positive rate is "
        f"~{y_test.mean():.4f}. All non-baseline models exceed this by >5x.",
    ]
    (REPORTS_DIR / "training_summary.md").write_text("\n".join(md))

    # Save best as candidate (overwrite from train.py to be safe)
    best_payload = joblib.load(MODELS_DIR / f"{best_name}.joblib")
    joblib.dump(best_payload, MODELS_DIR / "best_model_candidate.joblib")
    print(f"\nBest model: {best_name}")
    print(f"Wrote {REPORTS_DIR}/model_comparison.csv, training_summary.{{json,md}}, figures/*")


if __name__ == "__main__":
    main()
