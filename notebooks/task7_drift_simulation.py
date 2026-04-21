"""Task 7: Drift simulation + PSI visualization.

Run:
    python notebooks/task7_drift_simulation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    print("Starting Task 7 drift simulation...", flush=True)
    _ensure_repo_on_path()

    import pandas as pd

    from src.data_utils import get_feature_target_split, load_data
    from src.drift_detector import (
        detect_feature_drift,
        should_retrain,
        simulate_time_drift,
    )

    repo_root = Path(__file__).resolve().parents[1]
    transaction_path = repo_root / "data" / "raw" / "train_transaction.csv"
    identity_path = repo_root / "data" / "raw" / "train_identity.csv"

    print("Loading merged dataset...", flush=True)
    df = load_data(str(transaction_path), str(identity_path))

    # Keep the demo runnable on modest machines.
    if "TransactionDT" in df.columns and len(df) > 200_000:
        df = df.sort_values("TransactionDT").reset_index(drop=True).iloc[:200_000].copy()
        print(
            f"Downsampled to {len(df):,} rows for drift simulation.",
            flush=True,
        )

    print("Simulating time drift...", flush=True)
    df_train_raw, df_test_raw = simulate_time_drift(df)

    print("Encoding features and computing PSI...", flush=True)
    X_train, _y_train = get_feature_target_split(df_train_raw)
    X_test, _y_test = get_feature_target_split(df_test_raw)

    # Align columns to ensure drift is computed on the same feature set.
    all_cols = sorted(set(X_train.columns).union(set(X_test.columns)))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)

    drift_report = detect_feature_drift(X_train, X_test)
    psi_rows = [
        {"feature": feat, "psi": float(values["psi"])}
        for feat, values in drift_report.items()
        if feat != "_summary"
    ]
    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)

    print("\n=== Top 10 drifted features (by PSI) ===")
    print(psi_df.head(10).to_string(index=False))

    os.makedirs(repo_root / "reports", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = psi_df.head(30).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["psi"], color="#1f77b4")
    plt.axvline(0.2, color="red", linestyle="--", linewidth=2)
    plt.xlabel("PSI")
    plt.title("Top drifted features (PSI)")
    plt.tight_layout()
    out_path = repo_root / "reports" / "drift_simulation.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved drift plot to: {out_path}")

    retrain = should_retrain(drift_report=drift_report, recall_score=0.85)
    print(f"\nshould_retrain(): {retrain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

