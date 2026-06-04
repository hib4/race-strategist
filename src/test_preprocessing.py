"""Sanity checks for the preprocessing pipeline.

Verifies leakage controls, NaN handling, race-grouped split correctness, and
that the preprocessors round-trip train -> test without errors.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    GROUP_COL,
    HIGH_CARD_CAT_COLS,
    LEAKAGE_COLS,
    LOW_CARD_CAT_COLS,
    NUMERIC_COLS,
    TARGET,
    TEMPORAL_FILL_VALUE,
    TEMPORAL_NUMERIC_COLS,
    add_temporal_features,
    build_preprocessor,
    drop_leakage_columns,
    load_validated,
    make_race_grouped_split,
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_validated()


@pytest.fixture(scope="module")
def split(raw_df):
    return make_race_grouped_split(raw_df, test_size=0.2, random_state=42)


def test_target_present_in_raw(raw_df):
    assert TARGET in raw_df.columns


def test_drop_leakage_removes_all_known_columns(raw_df):
    cleaned = drop_leakage_columns(raw_df)
    for col in LEAKAGE_COLS:
        assert col not in cleaned.columns


def test_split_drops_leakage_columns_from_X(split):
    for col in LEAKAGE_COLS:
        assert col not in split.X_train.columns
        assert col not in split.X_test.columns


def test_split_drops_target_from_X(split):
    assert TARGET not in split.X_train.columns
    assert TARGET not in split.X_test.columns


def test_no_race_overlap_between_train_and_test(split):
    train_races = set(split.X_train[GROUP_COL].unique())
    test_races = set(split.X_test[GROUP_COL].unique())
    assert not (train_races & test_races), "Race(s) appear in both splits"


def test_test_races_list_matches_test_frame(split):
    assert set(split.test_races) == set(split.X_test[GROUP_COL].unique())


def test_compound_nans_filled(split):
    assert split.X_train["Compound"].isna().sum() == 0
    assert split.X_test["Compound"].isna().sum() == 0


def test_class_balance_preserved_roughly(split):
    train_rate = split.y_train.mean()
    test_rate = split.y_test.mean()
    assert 0.01 < train_rate < 0.06
    assert 0.01 < test_rate < 0.06


@pytest.mark.parametrize("kind", ["scaled", "tree"])
def test_preprocessor_no_nan_after_transform(split, kind):
    pre = build_preprocessor(kind)
    X_train_tx = pre.fit_transform(split.X_train, split.y_train)
    X_test_tx = pre.transform(split.X_test)
    assert not np.isnan(np.asarray(X_train_tx, dtype=float)).any()
    assert not np.isnan(np.asarray(X_test_tx, dtype=float)).any()


def test_target_encoder_does_not_see_test_during_fit(split):
    """The fit must depend only on (X_train, y_train); transforming X_test
    after fit must not change the encoder's internal mapping."""
    pre = build_preprocessor("scaled")
    pre.fit(split.X_train, split.y_train)
    ct = pre.named_steps["preprocess"]
    te = dict(ct.named_transformers_)["high_card"]
    before = [arr.copy() for arr in te.encodings_]
    _ = pre.transform(split.X_test)
    after = te.encodings_
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b, a)


def test_expected_columns_exist(split):
    expected = set(NUMERIC_COLS) | set(LOW_CARD_CAT_COLS) | set(HIGH_CARD_CAT_COLS)
    missing = expected - set(split.X_train.columns)
    assert not missing, f"Missing input columns: {missing}"


# --------------------------- Temporal feature tests ---------------------------


def _synthetic_lap_frame() -> pd.DataFrame:
    """Two drivers, one race, four laps each. Mixed compound for change test."""
    rows = []
    for driver, laptimes, compounds, deltas in [
        ("NOR", [90.0, 91.0, 92.0, 93.0], ["MEDIUM", "MEDIUM", "MEDIUM", "HARD"], [0.0, 1.0, 1.0, 1.0]),
        ("VER", [89.0, 89.5, 90.0, 90.5], ["SOFT", "SOFT", "MEDIUM", "MEDIUM"], [0.0, 0.5, 0.5, 0.5]),
    ]:
        for i, lap in enumerate([1, 2, 3, 4], start=0):
            rows.append({
                "Driver": driver,
                "Race": "Test GP",
                "Compound": compounds[i],
                "LapNumber": lap,
                "LapTime (s)": laptimes[i],
                "LapTime_Delta": deltas[i],
                "Position": 5 + i,
                "Position_Change": 0 if i == 0 else 1,
                "Cumulative_Degradation": -1.0 * i,
                "Stint": 1 if compounds[i] == compounds[0] else 2,
                "TyreLife": i + 1,
                "Year": 2024,
                "RaceProgress": 0.1 * (i + 1),
                "Normalized_TyreLife": 0.05 * (i + 1),
                TARGET: 0,
            })
    return pd.DataFrame(rows)


def test_add_temporal_features_adds_all_expected_columns():
    df = _synthetic_lap_frame()
    out = add_temporal_features(df)
    for col in TEMPORAL_NUMERIC_COLS:
        assert col in out.columns, f"Missing temporal column: {col}"


def test_add_temporal_features_first_lap_has_no_history():
    df = _synthetic_lap_frame()
    out = add_temporal_features(df)
    first_laps = out[out["LapNumber"] == 1]
    assert (first_laps["Lap_history_count"] == 0).all()
    for col in [
        "LapTime_lag1", "LapTime_lag2", "LapTime_lag3",
        "LapTime_Delta_lag1", "LapTime_Delta_lag2", "LapTime_Delta_lag3",
        "Position_lag1", "Position_Change_lag1",
        "LapTime_rolling_mean_3", "LapTime_rolling_std_3",
        "LapTime_Delta_rolling_mean_3", "LapTime_Delta_rolling_std_3",
        "Cumulative_Degradation_rolling_mean_3", "Position_rolling_mean_3",
    ]:
        assert (first_laps[col] == TEMPORAL_FILL_VALUE).all(), f"{col} not filled on lap 1"
    # Compound never changes "from nothing" on lap 1.
    assert (first_laps["Compound_changed_last_lap"] == 0).all()


def test_add_temporal_features_lag_matches_prior_lap():
    df = _synthetic_lap_frame()
    out = add_temporal_features(df)
    nor = out[out["Driver"] == "NOR"].sort_values("LapNumber").reset_index(drop=True)
    # Lap 2 sees lap 1
    assert nor.loc[1, "LapTime_lag1"] == 90.0
    assert nor.loc[1, "Lap_history_count"] == 1
    # Lap 3 sees laps 1, 2
    assert nor.loc[2, "LapTime_lag1"] == 91.0
    assert nor.loc[2, "LapTime_lag2"] == 90.0
    assert nor.loc[2, "Lap_history_count"] == 2
    # Lap 4 sees laps 1, 2, 3
    assert nor.loc[3, "LapTime_lag1"] == 92.0
    assert nor.loc[3, "LapTime_lag2"] == 91.0
    assert nor.loc[3, "LapTime_lag3"] == 90.0
    assert nor.loc[3, "Lap_history_count"] == 3


def test_add_temporal_features_rolling_mean_excludes_current():
    df = _synthetic_lap_frame()
    out = add_temporal_features(df)
    nor = out[out["Driver"] == "NOR"].sort_values("LapNumber").reset_index(drop=True)
    # Lap 4 rolling mean over (lap1=90, lap2=91, lap3=92) = 91.0
    assert nor.loc[3, "LapTime_rolling_mean_3"] == pytest.approx(91.0)
    # Slowdown signal = current (93) minus rolling mean (91) = 2.0
    assert nor.loc[3, "LapTime_diff_from_rolling_mean3"] == pytest.approx(2.0)


def test_add_temporal_features_compound_change_detected():
    df = _synthetic_lap_frame()
    out = add_temporal_features(df)
    nor = out[out["Driver"] == "NOR"].sort_values("LapNumber").reset_index(drop=True)
    # NOR switched MEDIUM -> HARD between lap 3 and lap 4
    assert nor.loc[3, "Compound_changed_last_lap"] == 1
    assert nor.loc[0, "Compound_changed_last_lap"] == 0  # no previous
    assert nor.loc[1, "Compound_changed_last_lap"] == 0  # same compound


def test_add_temporal_features_does_not_cross_driver_boundary():
    df = _synthetic_lap_frame()
    out = add_temporal_features(df)
    # VER's first lap must not see NOR's laps
    ver_first = out[(out["Driver"] == "VER") & (out["LapNumber"] == 1)].iloc[0]
    assert ver_first["Lap_history_count"] == 0
    assert ver_first["LapTime_lag1"] == TEMPORAL_FILL_VALUE


def test_add_temporal_features_single_row_inference():
    """Streamlit app sends a single-row frame with no history. The helper
    must return that row with sentinel values for every temporal feature."""
    row = pd.DataFrame([{
        "Driver": "NOR",
        "Race": "Azerbaijan Grand Prix",
        "Compound": "MEDIUM",
        "LapNumber": 25,
        "LapTime (s)": 90.0,
        "LapTime_Delta": 0.0,
        "Position": 5,
        "Position_Change": 0,
        "Cumulative_Degradation": -10.0,
        "Stint": 2,
        "TyreLife": 10,
        "Year": 2024,
        "RaceProgress": 0.3,
        "Normalized_TyreLife": 0.5,
        TARGET: 0,
    }])
    out = add_temporal_features(row)
    assert len(out) == 1
    for col in TEMPORAL_NUMERIC_COLS:
        assert col in out.columns
    assert out.iloc[0]["Lap_history_count"] == 0
    assert out.iloc[0]["LapTime_lag1"] == TEMPORAL_FILL_VALUE
    assert out.iloc[0]["Compound_changed_last_lap"] == 0


def test_temporal_features_causal_on_real_data(raw_df):
    """For every row in a representative (Driver, Race) group, verify that
    LapTime_lag1 equals the prior lap's LapTime (or the fill sentinel)."""
    df = drop_leakage_columns(raw_df).copy()
    df["Compound"] = df["Compound"].fillna("UNKNOWN")
    feat = add_temporal_features(df)

    # Pick the (Driver, Race, Year) group with the most laps. Race name alone
    # spans multiple seasons in this dataset, so the (Year) coordinate must be
    # part of the temporal grouping key.
    sizes = feat.groupby(["Driver", "Race", "Year"]).size()
    driver, race, year = sizes.idxmax()
    sub = (
        feat[(feat["Driver"] == driver) & (feat["Race"] == race) & (feat["Year"] == year)]
        .sort_values("LapNumber")
        .reset_index(drop=True)
    )
    assert len(sub) >= 5, "Need a group with multiple laps for a meaningful test"

    # First lap: sentinel.
    assert sub.loc[0, "LapTime_lag1"] == TEMPORAL_FILL_VALUE
    assert sub.loc[0, "Lap_history_count"] == 0

    # Every subsequent lap's lag1 equals the previous row's LapTime.
    for i in range(1, len(sub)):
        prev = sub.loc[i - 1, "LapTime (s)"]
        actual = sub.loc[i, "LapTime_lag1"]
        if pd.isna(prev):
            assert actual == TEMPORAL_FILL_VALUE
        else:
            assert actual == pytest.approx(prev), (
                f"Lap {sub.loc[i, 'LapNumber']} lag1 mismatch: got {actual}, expected {prev}"
            )

    # Rolling mean at lap with full history matches the manual past-3 average.
    if len(sub) >= 4:
        i = 3  # row index, has 3 past laps
        past_three = sub.loc[i - 3:i - 1, "LapTime (s)"].dropna()
        if len(past_three) >= 1:
            assert sub.loc[i, "LapTime_rolling_mean_3"] == pytest.approx(past_three.mean())


def test_temporal_features_present_in_split(split):
    for col in TEMPORAL_NUMERIC_COLS:
        assert col in split.X_train.columns, f"Missing {col} in X_train"
        assert col in split.X_test.columns, f"Missing {col} in X_test"


def test_temporal_features_have_no_nan_after_split(split):
    for col in TEMPORAL_NUMERIC_COLS:
        assert split.X_train[col].isna().sum() == 0, f"NaN in {col} (train)"
        assert split.X_test[col].isna().sum() == 0, f"NaN in {col} (test)"
