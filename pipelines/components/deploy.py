"""Deployment component for the fraud MLOps pipeline."""

from __future__ import annotations

import os
import shutil


def deploy_model(model_path: str, destination_path: str = "models/best_model.joblib") -> str:
    """Deploy a model artifact by copying it to the serving location.

    Args:
        model_path: Path to the model artifact to deploy.
        destination_path: Target path for the serving model.

    Returns:
        Deployed artifact path.
    """
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    shutil.copyfile(model_path, destination_path)
    return destination_path

