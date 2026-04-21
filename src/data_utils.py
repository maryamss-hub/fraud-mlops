"""Utility functions for loading and preparing IEEE-CIS style fraud data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder


HIGH_CARDINALITY_COLS = {
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
}


def load_data(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """Load and merge transaction + identity CSVs on `TransactionID`.

    Args:
        transaction_path: Path to the transactions CSV.
        identity_path: Path to the identity CSV.

    Returns:
        Merged dataframe. If the identity file has no matching IDs, missing
        identity fields will be NaN.
    """
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    if "TransactionID" not in transactions.columns:
        raise ValueError("transactions CSV must contain 'TransactionID'")
    if "TransactionID" not in identity.columns:
        raise ValueError("identity CSV must contain 'TransactionID'")
    merged = transactions.merge(identity, on="TransactionID", how="left")
    return merged


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with simple, explainable rules.

    Rules:
    - Drop columns with > 50% missing values
    - Fill numeric columns with median
    - Fill categorical columns with "Unknown"

    Args:
        df: Input dataframe.

    Returns:
        Cleaned dataframe.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    missing_frac = out.isna().mean()
    drop_cols = missing_frac[missing_frac > 0.50].index.tolist()
    if drop_cols:
        out = out.drop(columns=drop_cols)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in out.columns if c not in numeric_cols]

    for col in numeric_cols:
        if out[col].isna().any():
            median = out[col].median()
            out[col] = out[col].fillna(median)

    for col in cat_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna("Unknown")
        out[col] = out[col].astype(str)

    return out


def encode_categoricals(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Encode categorical columns using target-encoding or label-encoding.

    High-cardinality columns (IEEE-CIS specific) are target-encoded.
    Remaining object/category columns are label-encoded (integer codes).
    As a final safety step, any remaining object columns are factorized so that
    no object dtypes remain after encoding.

    Args:
        df: Input dataframe containing features and the target.
        target_col: Name of the binary target column.

    Returns:
        Dataframe with encoded categorical columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    out = df.copy()
    y = out[target_col].astype(int)
    out[target_col] = y

    cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return out

    high_cols = [c for c in cat_cols if c in HIGH_CARDINALITY_COLS]
    low_cols = [c for c in cat_cols if c not in HIGH_CARDINALITY_COLS]

    if high_cols:
        encoder = TargetEncoder(cols=high_cols, smoothing=10.0)
        out[high_cols] = encoder.fit_transform(out[high_cols], y)
        for col in high_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for col in low_cols:
        series = out[col].astype(str).fillna("Unknown")
        codes, _uniques = pd.factorize(series, sort=True)
        out[col] = codes.astype(np.int32)

    # Final catch-all: ensure *all* remaining object/category columns are numeric.
    for col in out.columns:
        if col == target_col:
            continue
        is_cat = isinstance(out[col].dtype, pd.CategoricalDtype)
        if pd.api.types.is_object_dtype(out[col]) or is_cat:
            codes, _uniques = pd.factorize(out[col].astype(str).fillna("Unknown"), sort=True)
            out[col] = codes.astype(np.int32)

    return out


def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target.

    Expects IEEE-CIS target column name `isFraud`.

    Args:
        df: Input dataframe.

    Returns:
        (X, y) where X is a dataframe of features and y is a Series.
    """
    if "isFraud" not in df.columns:
        raise ValueError("Expected target column 'isFraud' not found")

    encoded = encode_categoricals(df, target_col="isFraud")
    X = encoded.drop(columns=["isFraud"])
    y = encoded["isFraud"].astype(int)

    # Safety: ensure X has no object columns.
    for col in X.columns:
        is_cat = isinstance(X[col].dtype, pd.CategoricalDtype)
        if pd.api.types.is_object_dtype(X[col]) or is_cat:
            codes, _uniques = pd.factorize(X[col].astype(str).fillna("Unknown"), sort=True)
            X[col] = codes.astype(np.int32)
    return X, y


def time_based_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically using `TransactionDT`.

    Args:
        df: Input dataframe with a `TransactionDT` column.
        train_ratio: Fraction of rows to use for training (chronologically first).

    Returns:
        (df_train, df_test) dataframes.
    """
    if "TransactionDT" not in df.columns:
        raise ValueError("Expected column 'TransactionDT' for time-based split")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    sorted_df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(sorted_df):
        raise ValueError("train_ratio results in an empty train or test split")
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()

