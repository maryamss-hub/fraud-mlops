"""Model training utilities for fraud detection."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def _ensure_models_dir(models_dir: str = "models") -> str:
    """Ensure the models directory exists and return its path."""
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert input features to a numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute core binary classification metrics for comparison tables."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cost_sensitive: bool = False,
) -> XGBClassifier:
    """Train an XGBoost classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        cost_sensitive: If True, uses a high `scale_pos_weight` to penalize FN.

    Returns:
        Fitted XGBClassifier.
    """
    scale_pos_weight = 50.0 if cost_sensitive else 1.0
    model = XGBClassifier(
        n_estimators=300,
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
    model.fit(_to_numpy(X_train), np.asarray(y_train))
    _ensure_models_dir()
    joblib.dump(model, os.path.join("models", "xgboost.joblib"))
    joblib.dump(model, os.path.join("models", "xgboost_model.joblib"))
    return model


def train_lightgbm(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cost_sensitive: bool = False,
) -> LGBMClassifier:
    """Train a LightGBM classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        cost_sensitive: If True, uses balanced class weights.

    Returns:
        Fitted LGBMClassifier.
    """
    class_weight = "balanced" if cost_sensitive else None
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(_to_numpy(X_train), np.asarray(y_train))
    _ensure_models_dir()
    joblib.dump(model, os.path.join("models", "lightgbm.joblib"))
    return model


@dataclass
class HybridModel:
    """A simple wrapper for feature-selected XGBoost inference."""

    selected_features: List[str]
    selector: SelectFromModel
    model: XGBClassifier

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probabilities for class 1."""
        if isinstance(X, pd.DataFrame):
            X_sel = X[self.selected_features].values
        else:
            X_sel = self.selector.transform(np.asarray(X))
        return self.model.predict_proba(X_sel)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def train_hybrid(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
) -> HybridModel:
    """Train a hybrid model: RandomForest feature selection + XGBoost.

    Args:
        X_train: Training features as a dataframe (to preserve column names).
        y_train: Training labels.

    Returns:
        HybridModel with selected features and fitted XGBoost model.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("train_hybrid expects X_train as a pandas DataFrame")

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train.values, np.asarray(y_train))

    selector = SelectFromModel(rf, prefit=True, threshold="median")
    mask = selector.get_support()
    selected_features = X_train.columns[mask].tolist()
    if not selected_features:
        selected_features = X_train.columns.tolist()

    X_sel = X_train[selected_features].values
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=0,
    )
    xgb.fit(X_sel, np.asarray(y_train))

    hybrid = HybridModel(
        selected_features=selected_features,
        selector=selector,
        model=xgb,
    )
    _ensure_models_dir()
    joblib.dump(hybrid, os.path.join("models", "hybrid.joblib"))
    return hybrid


def compare_imbalance_strategies(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compare SMOTE vs class-weighted strategy using XGBoost.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        A dataframe with Precision/Recall/F1/AUC-ROC for each strategy.
    """
    if isinstance(X_train, pd.DataFrame):
        X_tr_df = X_train.copy()
    else:
        X_tr_df = pd.DataFrame(np.asarray(X_train))

    if isinstance(X_test, pd.DataFrame):
        X_te_df = X_test.copy()
    else:
        X_te_df = pd.DataFrame(np.asarray(X_test))

    if isinstance(y_train, pd.Series):
        y_tr_s = y_train.astype(int).copy()
    else:
        y_tr_s = pd.Series(np.asarray(y_train).astype(int), index=X_tr_df.index)

    y_te = np.asarray(y_test).astype(int)

    minority = int(min((y_tr_s == 0).sum(), (y_tr_s == 1).sum()))
    if minority >= 2:
        k_neighbors = max(1, min(5, minority - 1))
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        # Sample for SMOTE only — use stratified sample of 20000 rows
        sample_idx = (
            X_tr_df.groupby(y_tr_s)
            .apply(
                lambda x: x.sample(
                    min(len(x), 10000),
                    random_state=42,
                )
            )
            .index.get_level_values(1)
        )
        X_tr_sample = X_tr_df.loc[sample_idx]
        y_tr_sample = y_tr_s.loc[sample_idx]

        # Apply SMOTE on the sample only
        X_sm, y_sm = smote.fit_resample(X_tr_sample, y_tr_sample)
    else:
        # Too few minority samples for SMOTE; fall back to the original data.
        X_sm, y_sm = X_tr_df, y_tr_s

    model_smote = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=0,
    )
    model_smote.fit(np.asarray(X_sm), np.asarray(y_sm))
    prob_smote = model_smote.predict_proba(np.asarray(X_te_df))[:, 1]
    m_smote = _binary_metrics(y_te, prob_smote)

    # Class-weight strategy uses the full dataset
    y_tr = y_tr_s.to_numpy()
    X_tr = X_tr_df.to_numpy()
    X_te = X_te_df.to_numpy()
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    scale_pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    model_weighted = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=0,
    )
    model_weighted.fit(X_tr, y_tr)
    prob_weighted = model_weighted.predict_proba(X_te)[:, 1]
    m_weighted = _binary_metrics(y_te, prob_weighted)

    table = pd.DataFrame(
        [
            {"strategy": "SMOTE+XGBoost", **m_smote},
            {"strategy": "ClassWeight+XGBoost", **m_weighted},
        ]
    ).set_index("strategy")

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(table[["precision", "recall", "f1", "auc_roc"]])

    _ensure_models_dir()
    joblib.dump(model_smote, os.path.join("models", "xgboost_smote.joblib"))
    joblib.dump(model_weighted, os.path.join("models", "xgboost_classweight.joblib"))
    return table


def _synthetic_dataset(
    n_rows: int = 500,
    n_features: int = 20,
    fraud_rate: float = 0.05,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic dataset for local runs without the real data."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_rows, n_features))
    y = (rng.random(n_rows) < fraud_rate).astype(int)
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="isFraud")


def main(argv: Optional[List[str]] = None) -> int:
    """Train models from CSV paths (or synthetic data) and save artifacts."""
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data-csv", default="", help="Path to a prepared tabular CSV.")
    parser.add_argument("--target-col", default="isFraud", help="Target column name.")
    args = parser.parse_args(argv)

    if args.data_csv:
        df = pd.read_csv(args.data_csv)
        if args.target_col not in df.columns:
            raise ValueError(
                f"Target column '{args.target_col}' not found in {args.data_csv}"
            )
        X = df.drop(columns=[args.target_col])
        y = df[args.target_col].astype(int)
    else:
        X, y = _synthetic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    _ = compare_imbalance_strategies(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, cost_sensitive=False)
    train_lightgbm(X_train, y_train, cost_sensitive=False)
    train_hybrid(
        pd.DataFrame(X_train, columns=getattr(X, "columns", None)),
        y_train,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

