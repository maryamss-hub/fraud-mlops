## Fraud MLOps (IEEE-CIS)

This repository is a complete (assignment-friendly) MLOps fraud detection system built around the **IEEE-CIS Fraud Detection** dataset. It includes:

- **Training + evaluation** (XGBoost, LightGBM, Hybrid model)
- **Imbalance handling** (SMOTE vs class weight)
- **Cost-sensitive learning** and **business impact** analysis
- **Drift detection** (PSI) and **intelligent retraining**
- **FastAPI inference API** with **Prometheus metrics**
- **MLflow** experiment tracking and pipeline orchestration
- **CI/CD** via GitHub Actions
- **Monitoring assets** (Prometheus rules + Grafana dashboards)
- **Notebooks** for modeling and explainability (SHAP)

---

## Setup

### Install dependencies

From the project root (`fraud-mlops/`):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### Run unit tests

```bash
pytest -q tests
```

### Lint

```bash
flake8 src
```

---

## Training

The training entrypoint is `src/train.py`. It can run on:

- **Your prepared CSV** (features + target), or
- **Synthetic data** (default, for demo/testing)

Example (synthetic):

```bash
python -m src.train
```

Artifacts are saved to:

- `models/xgboost.joblib`
- `models/lightgbm.joblib`
- `models/hybrid.joblib`
- `models/xgboost_smote.joblib`
- `models/xgboost_classweight.joblib`

---

## Evaluation

Use `src/evaluate.py` helpers to evaluate any fitted model.

Saved plots:

- `reports/<model>_confusion_matrix.png`
- `reports/<model>_roc_curve.png`

---

## Cost-sensitive learning

Run a side-by-side comparison:

```bash
python -c "import numpy as np, pandas as pd; from sklearn.model_selection import train_test_split; from src.cost_sensitive import standard_vs_costsensitive_comparison; rng=np.random.default_rng(42); X=pd.DataFrame(rng.normal(size=(500,10))); y=pd.Series((rng.random(500)<0.07).astype(int)); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y); standard_vs_costsensitive_comparison(Xtr,ytr,Xte,yte)"
```

Output:

- `reports/cost_sensitive_comparison.csv`

---

## Drift detection + retraining

### Drift detection

`src/drift_detector.py` implements PSI-based drift detection:

- `calculate_psi()`
- `detect_feature_drift()`
- `simulate_time_drift()`
- `should_retrain()`

### Retraining

`src/retrain.py` implements:

- Threshold-based retraining
- Periodic retraining
- Hybrid retraining

Retraining logs:

- `logs/retrain_log.csv`

Promoted model:

- `models/best_model.joblib`

---

## Inference API (FastAPI + Prometheus)

Start the API locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict` (body: `{ "features": { ... } }`)
- `GET /metrics` (Prometheus scrape)

Prometheus metrics exposed:

- `fraud_api_requests_total{endpoint,status}`
- `fraud_api_latency_seconds_bucket` / `fraud_api_latency_seconds_count` / `fraud_api_latency_seconds_sum`
- `fraud_model_recall`
- `feature_drift_score`

---

## MLflow Pipeline

We use **MLflow** for experiment tracking and pipeline orchestration.

Run the pipeline:

```bash
python mlflow_pipeline/mlflow_run.py
```

The pipeline includes a **conditional deploy** step that only runs when recall \(>\) 0.85.

---

## CI/CD (GitHub Actions)

Workflow:

- `.github/workflows/mlops-pipeline.yml`

Jobs:

- `lint-and-test`: flake8 + pytest + validation smoke test
- `build-docker`: builds and pushes Docker images to GHCR
- `deploy-mlflow`: runs the MLflow pipeline (requires secrets, if configured)
- `intelligent-retrain`: manual trigger that runs drift + retraining (synthetic demo)

---

## Docker

### Training image

- `docker/training/Dockerfile`

### Inference image

- `docker/inference/Dockerfile`

---

## Monitoring

Prometheus alerting rules:

- `monitoring/prometheus_rules.yaml`

Grafana dashboards (importable JSON):

- `monitoring/grafana_dashboards/system_health.json`
- `monitoring/grafana_dashboards/model_performance.json`
- `monitoring/grafana_dashboards/data_drift.json`

---

## Notebooks

- `notebooks/01_eda.ipynb`
- `notebooks/02_modeling.ipynb`
- `notebooks/03_explainability.ipynb`


