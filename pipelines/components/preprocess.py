"""Preprocessing component for the fraud MLOps pipeline."""

from __future__ import annotations

import os

import pandas as pd

from src.data_utils import handle_missing_values


def preprocess(input_path: str, output_path: str) -> str:
    """Preprocess a dataset and write the cleaned output.

    Args:
        input_path: CSV or parquet path.
        output_path: Output CSV or parquet path.

    Returns:
        Output path written.
    """
    if input_path.lower().endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    df = handle_missing_values(df)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.lower().endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path

