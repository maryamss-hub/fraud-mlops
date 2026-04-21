"""Feature engineering component for the fraud MLOps pipeline."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data_utils import encode_categoricals


def feature_engineering(input_path: str, output_path: str, target_col: str = "isFraud") -> str:
    """Apply simple feature engineering and encoding.

    Adds a few derived features and encodes categoricals.

    Args:
        input_path: CSV or parquet path.
        output_path: Output CSV or parquet path.
        target_col: Target column name (default: isFraud).

    Returns:
        Output path written.
    """
    if input_path.lower().endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype(float))

    if target_col in df.columns:
        df = encode_categoricals(df, target_col=target_col)
    else:
        # Encode without target by converting categoricals to codes.
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            df[col] = pd.factorize(df[col].astype(str), sort=True)[0].astype(np.int32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path

