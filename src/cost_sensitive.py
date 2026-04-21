"""Cost-sensitive learning utilities for fraud detection."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.evaluate import business_impact_analysis


def _train_xgb(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    scale_pos_weight: float,
) -> XGBClassifier:
    """Train an XGBoost model with a specified `scale_pos_weight`."""
    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=0,
    )
    model.fit(np.asarray(X_train), np.asarray(y_train))
    return model


def standard_vs_costsensitive_comparison(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compare standard XGBoost vs cost-sensitive XGBoost and save results.

    Trains:
    - Standard XGBoost (`scale_pos_weight=1`)
    - Cost-sensitive XGBoost (`scale_pos_weight=50`)

    Computes recall/precision/F1/AUC and business impact for each approach,
    and saves a CSV to `reports/cost_sensitive_comparison.csv`.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dataframe of results.
    """
    os.makedirs("reports", exist_ok=True)

    y_true = np.asarray(y_test).astype(int)

    standard = _train_xgb(X_train, y_train, scale_pos_weight=1.0)
    cost_sensitive = _train_xgb(X_train, y_train, scale_pos_weight=50.0)

    def _row(name: str, model: XGBClassifier) -> Dict[str, float]:
        prob = model.predict_proba(np.asarray(X_test))[:, 1]
        pred = (prob >= 0.5).astype(int)
        impact = business_impact_analysis(
            y_true,
            pred,
            fn_cost=500.0,
            fp_cost=10.0,
        )
        return {
            "model": name,
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, prob)),
            "total_cost": float(impact["total_cost"]),
            "fraud_loss": float(impact["fraud_loss"]),
            "false_alarm_cost": float(impact["false_alarm_cost"]),
        }

    df = pd.DataFrame(
        [
            _row("xgboost_standard", standard),
            _row("xgboost_cost_sensitive_spw50", cost_sensitive),
        ]
    ).set_index("model")

    df.to_csv(os.path.join("reports", "cost_sensitive_comparison.csv"))
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n=== Standard vs Cost-Sensitive Comparison ===")
        print(df)
    return df

