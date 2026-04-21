"""Evaluation component wrapper for the fraud MLOps pipeline."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate_model


def evaluate_saved_model(
    model_path: str, data_path: str, target_col: str = "isFraud"
) -> Dict[str, float]:
    """Evaluate a saved model against a held-out split and return metrics.

    Args:
        model_path: Path to a joblib model.
        data_path: CSV or parquet containing features + target.
        target_col: Target column name.

    Returns:
        Metrics dict (precision/recall/f1/auc_roc).
    """
    model = joblib.load(model_path)
    if data_path.lower().endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    _X_train, X_test, _y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return evaluate_model(model, X_test, y_test, model_name="pipeline_model")

