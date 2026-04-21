"""Data validation component using Great Expectations."""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def validate_basic_schema(df: pd.DataFrame) -> Dict[str, object]:
    """Run lightweight validation checks without requiring a GE project.

    Checks:
    - `TransactionID` exists and is unique-like (no strict uniqueness enforced)
    - `TransactionDT` exists
    - If `isFraud` exists, it is binary (0/1) in non-null rows

    Args:
        df: Input dataframe.

    Returns:
        Dict containing validation status and any failures.
    """
    failures = []
    for col in ("TransactionID", "TransactionDT"):
        if col not in df.columns:
            failures.append(f"Missing required column: {col}")

    if "isFraud" in df.columns:
        vals = df["isFraud"].dropna().unique().tolist()
        if any(v not in (0, 1) for v in vals):
            failures.append("Target column 'isFraud' must be binary (0/1)")

    return {"success": len(failures) == 0, "failures": failures}


def validate_file(input_path: str) -> Dict[str, object]:
    """Load a dataset file and run basic validations.

    Args:
        input_path: CSV or parquet.

    Returns:
        Validation report dict.
    """
    if input_path.lower().endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    report = validate_basic_schema(df)
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame({"failure": report["failures"]}).to_csv(
        os.path.join("reports", "data_validation_failures.csv"), index=False
    )
    return report

