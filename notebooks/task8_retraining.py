"""Task 8: Retraining strategy comparison report.

Run:
    python notebooks/task8_retraining.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    print("Starting Task 8 retraining strategy report...", flush=True)
    _ensure_repo_on_path()

    from src.retrain import compare_strategies_report

    _ = compare_strategies_report()

    rows = [
        {
            "Strategy": "Threshold-based",
            "Trigger Condition": "Metric (e.g., recall) drops below threshold",
            "Compute Cost": "Medium",
            "Stability": "Medium",
            "When to Use": (
                "Strong monitoring + stable traffic; retrain only when needed"
            ),
        },
        {
            "Strategy": "Periodic",
            "Trigger Condition": "Fixed schedule (e.g., weekly/monthly)",
            "Compute Cost": "High",
            "Stability": "High",
            "When to Use": "Regulated/strict cadence or predictable drift patterns",
        },
        {
            "Strategy": "Hybrid",
            "Trigger Condition": "Emergency threshold OR periodic schedule",
            "Compute Cost": "Medium",
            "Stability": "High",
            "When to Use": "Best default: combines safety net with cadence",
        },
    ]

    print("\n=== Retraining strategy comparison (manual table) ===")
    headers = [
        "Strategy",
        "Trigger Condition",
        "Compute Cost",
        "Stability",
        "When to Use",
    ]
    widths = {h: max(len(h), max(len(r[h]) for r in rows)) for h in headers}
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(r[h].ljust(widths[h]) for h in headers))

    repo_root = Path(__file__).resolve().parents[1]
    os.makedirs(repo_root / "reports", exist_ok=True)
    out_path = repo_root / "reports" / "retraining_strategy_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

