"""API tests for FastAPI inference service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_endpoint_returns_probability():
    client = TestClient(app)
    payload = {"features": {"f0": 0.1, "f1": -0.2, "TransactionAmt": 123.45}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_metrics_endpoint_exposes_prometheus():
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "fraud_api_requests_total" in r.text

