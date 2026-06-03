"""Preprocessing for the F1 Race Strategist pit-prediction task.

Builds two preprocessor variants:
  - "scaled": RobustScaler on numerics, OneHot for Compound, TargetEncoder for
    Driver/Race. Used with Logistic Regression.
  - "tree":   passthrough numerics, OrdinalEncoder for all categoricals.
    Used with tree-based models.

Splits are race-grouped so no race appears in both train and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    TargetEncoder,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "f1_strategy_dataset_v4_validated.csv"

TARGET = "WillPitNextLap"
LEAKAGE_COLS = ["PitNextLap", "NextStint", "PitStop"]
GROUP_COL = "Race"

NUMERIC_COLS = [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "Year",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Normalized_TyreLife",
    "Position_Change",
]
LOW_CARD_CAT_COLS = ["Compound"]
HIGH_CARD_CAT_COLS = ["Driver", "Race"]


@dataclass
class SplitArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    test_races: list[str]


def load_validated(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} missing from {path}")
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])


def make_race_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitArtifacts:
    """Hold out ~test_size of rows by race so no race spans train and test."""
    df = drop_leakage_columns(df)
    df["Compound"] = df["Compound"].fillna("UNKNOWN")

    y = df[TARGET].astype(int)
    groups = df[GROUP_COL]
    X = df.drop(columns=[TARGET])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    test_races = sorted(X_test[GROUP_COL].unique().tolist())
    return SplitArtifacts(X_train, X_test, y_train, y_test, groups_train, test_races)


def build_preprocessor(kind: str = "scaled", random_state: int = 42) -> Pipeline:
    """Return a fit-ready Pipeline wrapping a ColumnTransformer.

    kind="scaled": for linear models. RobustScaler + OneHot(Compound)
        + TargetEncoder(Driver, Race). sklearn TargetEncoder fits with internal
        cross-fitting so no manual out-of-fold loop is required.
    kind="tree":   for tree models. Passthrough numerics + OrdinalEncoder for
        all categoricals (trees handle integer codes natively).
    """
    if kind not in {"scaled", "tree"}:
        raise ValueError("kind must be 'scaled' or 'tree'")

    if kind == "scaled":
        numeric_tf = RobustScaler()
        low_card_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        high_card_tf = TargetEncoder(target_type="binary", smooth="auto", cv=5)
        transformers = [
            ("num", numeric_tf, NUMERIC_COLS),
            ("low_card", low_card_tf, LOW_CARD_CAT_COLS),
            ("high_card", high_card_tf, HIGH_CARD_CAT_COLS),
        ]
    else:
        numeric_tf = "passthrough"
        cat_tf = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers = [
            ("num", numeric_tf, NUMERIC_COLS),
            ("cat", cat_tf, LOW_CARD_CAT_COLS + HIGH_CARD_CAT_COLS),
        ]

    ct = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocess", ct)])


def feature_names(preprocessor: Pipeline) -> list[str]:
    """Return final feature names after fit."""
    ct: ColumnTransformer = preprocessor.named_steps["preprocess"]
    return list(ct.get_feature_names_out())


def assert_no_nan(X: np.ndarray, label: str = "X") -> None:
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        raise AssertionError(f"{label} contains {n_nan} NaN values after transform")
