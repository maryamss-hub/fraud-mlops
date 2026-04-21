"""Intelligent retraining strategies for a fraud detection system."""

from __future__ import annotations

import csv
import os
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple


def threshold_based_retrain(current_recall: float, threshold: float = 0.80) -> bool:
    """Trigger retraining if recall drops below a threshold."""
    return float(current_recall) < float(threshold)


def periodic_retrain(last_retrain_date: date, interval_days: int = 7) -> bool:
    """Trigger retraining if `interval_days` have passed since last retrain."""
    if not isinstance(last_retrain_date, date):
        raise TypeError("last_retrain_date must be a datetime.date")
    return date.today() >= (last_retrain_date + timedelta(days=int(interval_days)))


def hybrid_retrain(
    current_recall: float,
    last_retrain_date: date,
    emergency_threshold: float = 0.75,
    periodic_days: int = 7,
) -> bool:
    """Hybrid strategy: emergency retrain on severe drop, else periodic."""
    if float(current_recall) < float(emergency_threshold):
        return True
    return periodic_retrain(last_retrain_date, interval_days=periodic_days)


def _train_candidates(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Dict[str, object]:
    """Train a small set of candidate models."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=0,
    )
    xgb.fit(np.asarray(X_train), np.asarray(y_train))

    lgbm = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgbm.fit(np.asarray(X_train), np.asarray(y_train))

    return {"xgboost": xgb, "lightgbm": lgbm}


def run_retrain_pipeline(
    X_new: pd.DataFrame | np.ndarray,
    y_new: pd.Series | np.ndarray,
) -> Tuple[object, Dict[str, float]]:
    """Retrain and promote the best model, logging metrics.

    The best model is selected by validation AUC-ROC. The promoted model is
    written to:
    - `models/best_model_<timestamp>.joblib`
    - `models/best_model.joblib` (latest pointer)
    Metrics are appended to `logs/retrain_log.csv`.

    Args:
        X_new: New training features.
        y_new: New training labels.

    Returns:
        (best_model, metrics_dict)
    """
    import joblib
    import numpy as np
    from sklearn.metrics import recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(
        np.asarray(X_new),
        np.asarray(y_new).astype(int),
        test_size=0.25,
        random_state=42,
        stratify=np.asarray(y_new).astype(int),
    )

    candidates = _train_candidates(X_train, y_train)
    best_name = ""
    best_auc = -1.0
    best_model: Optional[object] = None
    best_recall = 0.0

    for name, model in candidates.items():
        prob = model.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_val, prob))
        rec = float(recall_score(y_val, pred, zero_division=0))
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model
            best_recall = rec

    if best_model is None:
        raise RuntimeError("No candidate models were trained successfully")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    versioned_path = os.path.join("models", f"best_model_{ts}.joblib")
    latest_path = os.path.join("models", "best_model.joblib")
    joblib.dump(best_model, versioned_path)
    joblib.dump(best_model, latest_path)

    metrics = {
        "timestamp_utc": ts,
        "best_model": best_name,
        "auc_roc": best_auc,
        "recall": best_recall,
    }
    log_path = os.path.join("logs", "retrain_log.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"Promoted model: {best_name} (AUC={best_auc:.4f}, Recall={best_recall:.4f})")
    print(f"Saved: {versioned_path}")
    return best_model, {"auc_roc": best_auc, "recall": best_recall}


def compare_strategies_report() -> list[dict[str, str]]:
    """Print a simple qualitative comparison of retraining strategies.

    Returns:
        List of row dicts describing each strategy.
    """
    rows: list[dict[str, str]] = [
        {
            "strategy": "threshold-based",
            "stability": "medium",
            "cost": "medium",
            "performance": "high when monitored",
            "notes": "Retrains only on metric degradation; needs reliable monitoring.",
        },
        {
            "strategy": "periodic",
            "stability": "high",
            "cost": "high",
            "performance": "medium",
            "notes": "Simple schedule; may retrain unnecessarily when data is stable.",
        },
        {
            "strategy": "hybrid",
            "stability": "high",
            "cost": "medium",
            "performance": "high",
            "notes": "Emergency retrain on severe drop plus periodic refresh.",
        },
    ]

    headers = ["strategy", "stability", "cost", "performance", "notes"]
    widths = {h: max(len(h), max(len(r[h]) for r in rows)) for h in headers}

    print("\n=== Retraining Strategy Comparison ===")
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(r[h].ljust(widths[h]) for h in headers))
    return rows

