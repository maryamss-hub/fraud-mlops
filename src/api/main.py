"""FastAPI inference service for fraud detection with Prometheus metrics."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sklearn.dummy import DummyClassifier

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse


fraud_api_requests_total = Counter(
    "fraud_api_requests_total",
    "Total requests to the fraud API",
    labelnames=("endpoint", "status"),
)

fraud_api_latency_seconds = Histogram(
    "fraud_api_latency_seconds",
    "Latency of fraud API requests in seconds",
    labelnames=("endpoint",),
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)

fraud_model_recall = Gauge(
    "fraud_model_recall",
    "Latest observed model recall (from monitoring job)",
)

feature_drift_score = Gauge(
    "feature_drift_score",
    "Latest observed drift score (max PSI across monitored features)",
)


def _load_model(path: str) -> Tuple[Any, bool]:
    """Load a model from disk or create a safe fallback."""
    if os.path.exists(path):
        return joblib.load(path), True

    # Fallback keeps the service usable for tests/demo environments.
    X = np.zeros((20, 5))
    y = np.zeros(20, dtype=int)
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X, y)
    return dummy, False


def _align_features(model: Any, features: Dict[str, Any]) -> pd.DataFrame:
    """Build a 1-row dataframe aligned to the model's expected feature set."""
    row = pd.DataFrame([features])
    if hasattr(model, "feature_names_in_"):
        cols = list(getattr(model, "feature_names_in_"))
        for c in cols:
            if c not in row.columns:
                row[c] = 0
        row = row[cols]
    return row


app = FastAPI(title="Fraud Detection API", version="1.0.0")

MODEL_PATH = os.path.join("models", "best_model.joblib")
MODEL: Optional[Any] = None
MODEL_LOADED: bool = False
MODEL_VERSION: Optional[str] = None

# Ensure the model is available even when startup events aren't triggered
# (e.g., some test runners / client configurations).
MODEL, MODEL_LOADED = _load_model(MODEL_PATH)
MODEL_VERSION = os.path.basename(MODEL_PATH) if MODEL_LOADED else "fallback_dummy"
fraud_model_recall.set(1.0 if MODEL_LOADED else 0.0)
feature_drift_score.set(0.0)


@app.on_event("startup")
def _startup() -> None:
    """Load the model at startup and initialize gauges."""
    global MODEL, MODEL_LOADED, MODEL_VERSION
    MODEL, MODEL_LOADED = _load_model(MODEL_PATH)
    MODEL_VERSION = os.path.basename(MODEL_PATH) if MODEL_LOADED else "fallback_dummy"
    fraud_model_recall.set(1.0 if MODEL_LOADED else 0.0)
    feature_drift_score.set(0.0)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Collect request counters and latency histograms."""
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status_label = "ok" if response.status_code < 400 else "error"
        fraud_api_requests_total.labels(endpoint=endpoint, status=status_label).inc()
        return response
    except Exception:
        fraud_api_requests_total.labels(endpoint=endpoint, status="error").inc()
        raise
    finally:
        fraud_api_latency_seconds.labels(endpoint=endpoint).observe(
            time.perf_counter() - start
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health endpoint for liveness/readiness checks."""
    return HealthResponse(status="ok", model_loaded=bool(MODEL_LOADED))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict fraud probability for a single transaction."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    X = _align_features(MODEL, req.features)

    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)
        if proba.shape[1] >= 2:
            prob = float(proba[:, 1][0])
        else:
            prob = 0.0
    else:
        pred = float(MODEL.predict(X)[0])
        prob = float(max(0.0, min(1.0, pred)))

    prediction = int(prob >= 0.5)
    return PredictResponse(
        fraud_probability=prob,
        prediction=prediction,
        model_version=MODEL_VERSION,
    )


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
