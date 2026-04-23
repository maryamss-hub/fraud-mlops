"""Unit tests for `src.data_utils` using synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_utils import (
    encode_categoricals,
    get_feature_target_split,
    handle_missing_values,
    load_data,
    time_based_train_test_split,
)


def test_load_data_merges_on_transaction_id(tmp_path):
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
            "TransactionAmt": [100.0, 200.0, 150.0],
            "isFraud": [0, 1, 0],
        }
    )
    identity = pd.DataFrame({"TransactionID": [2, 3], "DeviceInfo": ["A", "B"]})

    t_path = tmp_path / "transactions.csv"
    i_path = tmp_path / "identity.csv"
    transactions.to_csv(t_path, index=False)
    identity.to_csv(i_path, index=False)

    merged = load_data(str(t_path), str(i_path))
    assert merged.shape[0] == 3
    assert "DeviceInfo" in merged.columns
    assert merged.loc[merged["TransactionID"] == 1, "DeviceInfo"].isna().all()


def test_handle_missing_values_drops_high_missing_and_fills(tmp_path):
    df = pd.DataFrame(
        {
            "num": [1.0, None, 3.0, None],
            "cat": ["x", None, "y", None],
            "mostly_missing": [None, None, None, 1.0],
        }
    )
    cleaned = handle_missing_values(df)
    assert "mostly_missing" not in cleaned.columns
    assert cleaned["num"].isna().sum() == 0
    assert cleaned["cat"].isna().sum() == 0
    assert (cleaned["cat"] == "Unknown").sum() == 2


def test_encode_categoricals_encodes_high_and_low_cardinality():
    df = pd.DataFrame(
        {
            "TransactionDT": [1, 2, 3, 4],
            "card1": ["a", "b", "a", "c"],
            "lowcat": ["x", "y", "x", "x"],
            "isFraud": [0, 1, 0, 1],
        }
    )
    enc = encode_categoricals(df, target_col="isFraud")
    assert enc["card1"].dtype.kind in ("i", "u", "f")
    assert enc["lowcat"].dtype.kind in ("i", "u")
    assert enc.isna().sum().sum() == 0


def test_get_feature_target_split():
    df = pd.DataFrame({"a": [1, 2], "isFraud": [0, 1]})
    X, y = get_feature_target_split(df)
    assert "isFraud" not in X.columns
    assert list(y.values) == [0, 1]


def test_time_based_train_test_split_chronological():
    df = pd.DataFrame({"TransactionDT": [3, 1, 2, 4], "x": [30, 10, 20, 40]})
    train, test = time_based_train_test_split(df, train_ratio=0.5)
    assert train["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(train) == 2
    assert len(test) == 2


def test_time_based_train_test_split_requires_column():
    with pytest.raises(ValueError):
        _ = time_based_train_test_split(pd.DataFrame({"x": [1, 2, 3]}))
