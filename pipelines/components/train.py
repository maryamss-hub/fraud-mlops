"""Training component wrapper for the fraud MLOps pipeline."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.train import train_lightgbm, train_xgboost


def train_models(input_path: str, target_col: str = "isFraud") -> Dict[str, str]:
    """Train baseline models from a prepared dataset file.

    Args:
        input_path: CSV or parquet containing features + target.
        target_col: Target column name.

    Returns:
        Dict mapping model name to saved model path.
    """
    if input_path.lower().endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    xgb = train_xgboost(X_train, y_train, cost_sensitive=False)
    lgbm = train_lightgbm(X_train, y_train, cost_sensitive=False)

    os.makedirs("models", exist_ok=True)
    paths = {
        "xgboost": os.path.join("models", "xgboost.joblib"),
        "lightgbm": os.path.join("models", "lightgbm.joblib"),
    }
    joblib.dump(xgb, paths["xgboost"])
    joblib.dump(lgbm, paths["lightgbm"])
    return paths

