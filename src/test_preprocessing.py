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
