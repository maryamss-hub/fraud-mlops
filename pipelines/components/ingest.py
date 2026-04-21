"""Data ingestion component for the fraud MLOps pipeline."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def ingest(transaction_path: str, identity_path: str, output_path: str) -> str:
    """Ingest and merge raw CSV files, writing a merged parquet/csv.

    Args:
        transaction_path: Path to transaction CSV.
        identity_path: Path to identity CSV.
        output_path: Output file path (csv or parquet).

    Returns:
        Output path written.
    """
    df_t = pd.read_csv(transaction_path)
    df_i = pd.read_csv(identity_path)
    if "TransactionID" not in df_t.columns or "TransactionID" not in df_i.columns:
        raise ValueError("Both inputs must include 'TransactionID'")
    df = df_t.merge(df_i, on="TransactionID", how="left")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path

