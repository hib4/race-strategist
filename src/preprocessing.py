"""Preprocessing for the F1 pit-prediction task.

Builds two preprocessor variants:
  - "scaled": RobustScaler on numerics, OneHot for Compound, TargetEncoder for
    Driver/Race. Used with Logistic Regression.
  - "tree":   passthrough numerics, OrdinalEncoder for all categoricals.
    Used with tree-based models.

Splits are race-grouped so no race appears in both train and test.

Also exposes `add_temporal_features` which adds causal per-(Driver, Race)
lag / rolling features. All new feature values for row i in a (Driver, Race)
group depend strictly on rows with LapNumber < i.LapNumber for the same
(Driver, Race), so they are leakage-free across laps. Used both during
training (called inside `make_race_grouped_split`) and at inference time
(called by the Streamlit app on a single-row input).
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

BASE_NUMERIC_COLS = [
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

# Causal lag / rolling features computed inside add_temporal_features.
TEMPORAL_NUMERIC_COLS = [
    "LapTime_lag1",
    "LapTime_lag2",
    "LapTime_lag3",
    "LapTime_Delta_lag1",
    "LapTime_Delta_lag2",
    "LapTime_Delta_lag3",
    "LapTime_rolling_mean_3",
    "LapTime_rolling_std_3",
    "LapTime_Delta_rolling_mean_3",
    "LapTime_Delta_rolling_std_3",
    "Cumulative_Degradation_rolling_mean_3",
    "Position_lag1",
    "Position_rolling_mean_3",
    "Position_Change_lag1",
    "LapTime_diff_from_rolling_mean3",
    "Compound_changed_last_lap",
    "Lap_history_count",
]

NUMERIC_COLS = BASE_NUMERIC_COLS + TEMPORAL_NUMERIC_COLS
LOW_CARD_CAT_COLS = ["Compound"]
HIGH_CARD_CAT_COLS = ["Driver", "Race"]

# Fill value used for lag/rolling features when a row has no prior history.
# `Lap_history_count` carries the "how much history is available" signal.
TEMPORAL_FILL_VALUE = 0.0


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


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-(Driver, Race) causal lag/rolling features.

    For each row, every new feature depends strictly on rows with smaller
    `LapNumber` within the same (Driver, Race, Year) group. The (Year)
    coordinate matters because the same `Race` name (e.g. "Abu Dhabi Grand
    Prix") appears in multiple seasons and we must not let one year's lap
    history bleed into another. The function is safe to call on the full
    historical dataset before the train/test split and on single-row inference
    inputs (which have no history, so all lag/rolling values fall back to the
    `TEMPORAL_FILL_VALUE` sentinel while `Lap_history_count = 0` carries the
    no-history signal).

    Returns a new DataFrame sorted by (Driver, Race, Year, LapNumber) with a
    fresh RangeIndex. The caller is responsible for not relying on the
    original row order (the race-grouped split downstream uses positional
    indices, so this is fine).
    """
    required = ["Driver", "Race", "Year", "LapNumber", "Compound"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_temporal_features missing columns: {missing}")

    out = (
        df.sort_values(["Driver", "Race", "Year", "LapNumber"], kind="mergesort")
        .reset_index(drop=True)
    )

    grouped = out.groupby(["Driver", "Race", "Year"], sort=False, group_keys=False)

    def lag(col: str, k: int) -> pd.Series:
        return grouped[col].shift(k)

    def rolling_past_mean(col: str) -> pd.Series:
        return grouped[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    def rolling_past_std(col: str) -> pd.Series:
        return grouped[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )

    out["LapTime_lag1"] = lag("LapTime (s)", 1)
    out["LapTime_lag2"] = lag("LapTime (s)", 2)
    out["LapTime_lag3"] = lag("LapTime (s)", 3)

    out["LapTime_Delta_lag1"] = lag("LapTime_Delta", 1)
    out["LapTime_Delta_lag2"] = lag("LapTime_Delta", 2)
    out["LapTime_Delta_lag3"] = lag("LapTime_Delta", 3)

    out["Position_lag1"] = lag("Position", 1)
    out["Position_Change_lag1"] = lag("Position_Change", 1)

    out["LapTime_rolling_mean_3"] = rolling_past_mean("LapTime (s)")
    out["LapTime_rolling_std_3"] = rolling_past_std("LapTime (s)")
    out["LapTime_Delta_rolling_mean_3"] = rolling_past_mean("LapTime_Delta")
    out["LapTime_Delta_rolling_std_3"] = rolling_past_std("LapTime_Delta")
    out["Cumulative_Degradation_rolling_mean_3"] = rolling_past_mean(
        "Cumulative_Degradation"
    )
    out["Position_rolling_mean_3"] = rolling_past_mean("Position")

    # Slowdown signal: how much the current lap differs from the recent average.
    out["LapTime_diff_from_rolling_mean3"] = (
        out["LapTime (s)"] - out["LapTime_rolling_mean_3"]
    )

    prev_compound = grouped["Compound"].shift(1)
    out["Compound_changed_last_lap"] = (
        prev_compound.notna() & (out["Compound"].astype(str) != prev_compound.astype(str))
    ).astype(int)

    # Number of past laps available, clipped to [0, 3] — the window we look back.
    out["Lap_history_count"] = grouped.cumcount().clip(upper=3).astype(int)

    # Fill NaN created by lag/rolling on the leading laps of each group.
    fill_cols = [
        c for c in TEMPORAL_NUMERIC_COLS
        if c not in {"Compound_changed_last_lap", "Lap_history_count"}
    ]
    for col in fill_cols:
        out[col] = out[col].fillna(TEMPORAL_FILL_VALUE).astype(float)

    return out


def make_race_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitArtifacts:
    """Hold out ~test_size of rows by race so no race spans train and test."""
    df = drop_leakage_columns(df)
    df["Compound"] = df["Compound"].fillna("UNKNOWN")
    df = add_temporal_features(df)

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
