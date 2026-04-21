"""Evaluation utilities for fraud detection models."""

from __future__ import annotations

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """Evaluate a model and save key plots.

    Saves:
    - `reports/{model_name}_confusion_matrix.png`
    - `reports/{model_name}_roc_curve.png`

    Args:
        model: A fitted model with `predict` and optionally `predict_proba`.
        X_test: Test features.
        y_test: True labels.
        model_name: Name used for reporting and filenames.

    Returns:
        Dictionary of metrics.
    """
    _ensure_dir("reports")

    y_true = np.asarray(y_test).astype(int)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.close()

    fpr, tpr, _thr = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", f"{model_name}_roc_curve.png"), dpi=150)
    plt.close()

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(auc),
    }
    return metrics


def compare_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Evaluate multiple models and print a comparison table.

    Args:
        models_dict: Mapping from model name to model object.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dataframe of metrics per model.
    """
    rows = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append({"model": name, **metrics})
    table = pd.DataFrame(rows).set_index("model").sort_values("auc_roc", ascending=False)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n=== Model Comparison ===")
        print(table)
    return table


def business_impact_analysis(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    fn_cost: float = 500.0,
    fp_cost: float = 10.0,
) -> Dict[str, float]:
    """Compute simple business impact based on false negatives and false positives.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (0/1).
        fn_cost: Cost per missed fraud (false negative).
        fp_cost: Cost per false alert (false positive).

    Returns:
        Dictionary containing counts and total cost.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fraud_loss = float(fn * fn_cost)
    false_alarm_cost = float(fp * fp_cost)
    total = fraud_loss + false_alarm_cost

    table = pd.DataFrame(
        [
            {"item": "False Negatives (missed fraud)", "count": int(fn), "unit_cost": fn_cost, "total_cost": fraud_loss},
            {"item": "False Positives (false alarms)", "count": int(fp), "unit_cost": fp_cost, "total_cost": false_alarm_cost},
            {"item": "Total", "count": int(fp + fn), "unit_cost": np.nan, "total_cost": total},
        ]
    )
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n=== Business Impact ===")
        print(table)

    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "fraud_loss": fraud_loss,
        "false_alarm_cost": false_alarm_cost,
        "total_cost": total,
    }

