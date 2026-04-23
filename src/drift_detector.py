"""Data drift detection utilities (PSI-based) for fraud pipelines."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI).

    PSI compares how a feature distribution shifts between a reference (expected)
    and a new sample (actual). Common interpretation:
    - < 0.1: no significant shift
    - 0.1–0.2: moderate shift
    - > 0.2: significant shift

    Args:
        expected: Reference sample values.
        actual: New sample values.
        buckets: Number of buckets to use.

    Returns:
        PSI value (non-negative).
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)

    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if exp.size == 0 or act.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(exp, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts, _ = np.histogram(exp, bins=breakpoints)
    act_counts, _ = np.histogram(act, bins=breakpoints)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1.0)
    act_perc = np.clip(act_perc, eps, 1.0)

    psi = np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))
    return float(max(psi, 0.0))


def detect_feature_drift(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """Compute PSI for all numeric features and flag drifted ones.

    Args:
        df_train: Reference (training) dataframe.
        df_test: New (testing/serving) dataframe.
        threshold: PSI threshold used to flag drift.

    Returns:
        Drift report dict with per-feature PSI and a `drifted` flag.
    """
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    report: Dict[str, Dict[str, float]] = {}
    for col in num_cols:
        if col not in df_test.columns:
            continue
        psi = calculate_psi(df_train[col].values, df_test[col].values, buckets=10)
        report[col] = {"psi": float(psi), "drifted": float(psi > threshold)}

    drifted_features = [c for c, v in report.items() if v["psi"] > threshold]
    report["_summary"] = {
        "num_features_checked": float(
            len([c for c in num_cols if c in df_test.columns])
        ),
        "num_drifted": float(len(drifted_features)),
        "max_psi": float(
            max(
                (v["psi"] for k, v in report.items() if k != "_summary"),
                default=0.0,
            )
        ),
    }
    return report


def simulate_time_drift(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a simple time-based drift scenario for demonstration.

    Steps:
    - Split the full dataframe chronologically using `TransactionDT`.
    - In the test split, for 30% of fraud cases: multiply `TransactionAmt` by 3.
    - In the test split, for 20% of fraud cases: simulate a new device type.

    Args:
        df: Full dataframe with `TransactionDT`, `isFraud`, and optionally
            `TransactionAmt` and `DeviceType`.

    Returns:
        (df_train, df_test_drifted)
    """
    if "TransactionDT" not in df.columns:
        raise ValueError("simulate_time_drift requires 'TransactionDT'")
    if "isFraud" not in df.columns:
        raise ValueError("simulate_time_drift requires 'isFraud'")

    sorted_df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.7)
    if split_idx <= 0 or split_idx >= len(sorted_df):
        raise ValueError("Not enough rows to split chronologically")

    df_train = sorted_df.iloc[:split_idx].copy()
    df_test = sorted_df.iloc[split_idx:].copy()

    rng = np.random.default_rng(42)
    fraud_mask = df_test["isFraud"].astype(int) == 1
    fraud_idx = df_test.index[fraud_mask].to_numpy()

    if fraud_idx.size > 0 and "TransactionAmt" in df_test.columns:
        n_amt = int(np.ceil(0.30 * fraud_idx.size))
        chosen = rng.choice(fraud_idx, size=min(n_amt, fraud_idx.size), replace=False)
        df_test.loc[chosen, "TransactionAmt"] = (
            df_test.loc[chosen, "TransactionAmt"].astype(float) * 3.0
        )

    device_col = "DeviceType"
    if device_col not in df_test.columns:
        df_test[device_col] = "Unknown"

    if fraud_idx.size > 0:
        n_dev = int(np.ceil(0.20 * fraud_idx.size))
        chosen = rng.choice(fraud_idx, size=min(n_dev, fraud_idx.size), replace=False)
        df_test.loc[chosen, device_col] = "new_device_type"

    return df_train, df_test


def should_retrain(
    drift_report: Dict[str, Dict[str, float]],
    recall_score: float,
    recall_threshold: float = 0.80,
    drift_threshold: float = 0.2,
) -> bool:
    """Decide if retraining should be triggered.

    Triggers when:
    - Recall is below `recall_threshold`, OR
    - Any feature PSI exceeds `drift_threshold`

    Args:
        drift_report: Output from `detect_feature_drift`.
        recall_score: Current recall score (0..1).
        recall_threshold: Minimum acceptable recall.
        drift_threshold: PSI threshold for significant drift.

    Returns:
        True if retraining should be triggered, else False.
    """
    if recall_score < recall_threshold:
        return True

    for feat, values in drift_report.items():
        if feat == "_summary":
            continue
        if float(values.get("psi", 0.0)) > drift_threshold:
            return True
    return False
