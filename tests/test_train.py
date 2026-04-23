"""Unit tests for `src.train` using synthetic data."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.train import (
    compare_imbalance_strategies,
    train_hybrid,
    train_lightgbm,
    train_xgboost,
)


def _make_data(n_rows: int = 100, n_features: int = 10):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series((rng.random(n_rows) < 0.12).astype(int))
    return X, y


def test_train_xgboost_saves_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _make_data()
    model = train_xgboost(X, y, cost_sensitive=True)
    assert os.path.exists(os.path.join("models", "xgboost.joblib"))
    prob = model.predict_proba(X.values)[:, 1]
    assert prob.shape[0] == X.shape[0]


def test_train_lightgbm_saves_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _make_data()
    model = train_lightgbm(X, y, cost_sensitive=True)
    assert os.path.exists(os.path.join("models", "lightgbm.joblib"))
    prob = model.predict_proba(X.values)[:, 1]
    assert prob.shape[0] == X.shape[0]


def test_train_hybrid_saves_model_and_predicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _make_data()
    model = train_hybrid(X, y)
    assert os.path.exists(os.path.join("models", "hybrid.joblib"))
    prob = model.predict_proba(X)[:, 1]
    assert prob.shape[0] == X.shape[0]


def test_compare_imbalance_strategies_returns_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, y = _make_data()
    X_train, X_test = X.iloc[:70], X.iloc[70:]
    y_train, y_test = y.iloc[:70], y.iloc[70:]
    table = compare_imbalance_strategies(X_train, y_train, X_test, y_test)
    assert "precision" in table.columns
    assert "recall" in table.columns
    assert table.shape[0] == 2
